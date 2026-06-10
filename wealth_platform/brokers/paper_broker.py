"""Paper broker: full simulation with realistic Indian transaction costs.

State persists to data/paper_portfolio.json so the simulated portfolio
survives restarts. This is the DEFAULT broker — the platform always starts
here until you explicitly switch to a live broker in wealth_config.json.
"""

import json
import logging
import os
import uuid
from typing import Dict, Optional

import yfinance as yf

from wealth_platform.brokers.base_broker import BaseBroker, Funds, OrderResult, Position, Quote

logger = logging.getLogger("wealth_platform.brokers.paper")

STATE_PATH = os.path.join("data", "paper_portfolio.json")

# Approximate all-in cost for an NSE delivery round trip at a discount broker
BROKERAGE_FLAT = 20.0  # per executed order
STT_RATE = 0.001  # 0.1% on delivery sell side (approximated both ways for safety)
OTHER_CHARGES_RATE = 0.0005


class PaperBroker(BaseBroker):
    name = "paper"

    def __init__(self, starting_cash: float = 15000.0, state_path: str = STATE_PATH):
        self.state_path = state_path
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.trade_log = []
        self._load()

    def connect(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        yf_symbol = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
        if data.empty:
            data = yf.Ticker(yf_symbol).history(period="5d")
        if data.empty:
            raise RuntimeError(f"No quote available for {symbol}")
        return Quote(symbol=symbol, last_price=float(data["Close"].iloc[-1]))

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: str = "CNC",
    ) -> OrderResult:
        if quantity <= 0:
            return OrderResult(success=False, message="Quantity must be positive")
        fill_price = price if (order_type == "LIMIT" and price) else self.get_quote(symbol).last_price
        gross = fill_price * quantity
        costs = BROKERAGE_FLAT + gross * (STT_RATE + OTHER_CHARGES_RATE)

        if side == "BUY":
            total = gross + costs
            if total > self.cash:
                return OrderResult(success=False, message=f"Insufficient paper cash: need {total:.2f}, have {self.cash:.2f}")
            self.cash -= total
            existing = self.positions.get(symbol)
            if existing:
                new_qty = existing.quantity + quantity
                existing.average_price = (existing.average_price * existing.quantity + gross) / new_qty
                existing.quantity = new_qty
            else:
                self.positions[symbol] = Position(symbol=symbol, quantity=quantity, average_price=fill_price)
        elif side == "SELL":
            existing = self.positions.get(symbol)
            if not existing or existing.quantity < quantity:
                return OrderResult(success=False, message=f"No sufficient paper position in {symbol}")
            self.cash += gross - costs
            existing.quantity -= quantity
            if existing.quantity == 0:
                del self.positions[symbol]
        else:
            return OrderResult(success=False, message=f"Unknown side {side}")

        order_id = f"PAPER-{uuid.uuid4().hex[:8]}"
        self.trade_log.append(
            {"order_id": order_id, "symbol": symbol, "side": side, "qty": quantity, "price": fill_price, "costs": round(costs, 2)}
        )
        self._save()
        logger.info("Paper %s %d %s @ %.2f (costs %.2f)", side, quantity, symbol, fill_price, costs)
        return OrderResult(success=True, order_id=order_id, filled_price=fill_price, message="Paper fill")

    def get_positions(self) -> Dict[str, Position]:
        for symbol, pos in self.positions.items():
            try:
                pos.last_price = self.get_quote(symbol).last_price
            except Exception:  # noqa: BLE001 - keep stale price rather than crash
                pos.last_price = pos.last_price or pos.average_price
        return self.positions

    def get_funds(self) -> Funds:
        return Funds(available_cash=self.cash)

    def portfolio_value(self) -> float:
        return self.cash + sum(p.value for p in self.get_positions().values())

    # ------------------------------------------------------------------

    def _save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(
                {
                    "cash": self.cash,
                    "starting_cash": self.starting_cash,
                    "positions": {
                        s: {"quantity": p.quantity, "average_price": p.average_price}
                        for s, p in self.positions.items()
                    },
                    "trade_log": self.trade_log[-500:],
                },
                f,
                indent=2,
            )

    def _load(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self.cash = state.get("cash", self.starting_cash)
            self.starting_cash = state.get("starting_cash", self.starting_cash)
            self.trade_log = state.get("trade_log", [])
            for symbol, p in state.get("positions", {}).items():
                self.positions[symbol] = Position(symbol=symbol, quantity=p["quantity"], average_price=p["average_price"])
        except Exception:  # noqa: BLE001
            logger.exception("Could not load paper portfolio state; starting fresh")
