"""Zerodha (Kite Connect) adapter. Wraps the existing kiteconnect integration
behind the common BaseBroker interface.

Requires: ZERODHA_API_KEY, ZERODHA_SECRET_KEY env vars and a valid access
token (run `python zerodha_auth.py` to refresh — tokens expire daily).
NOTE: Kite Connect is a paid API (Rs.500/month for order placement).
Use Groww (free API) or paper mode if you want zero broker-API cost.
"""

import json
import logging
import os
from typing import Dict, Optional

from wealth_platform.brokers.base_broker import BaseBroker, Funds, OrderResult, Position, Quote

logger = logging.getLogger("wealth_platform.brokers.zerodha")


class ZerodhaBroker(BaseBroker):
    name = "zerodha"

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.kite = None

    def connect(self) -> bool:
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            logger.error("kiteconnect not installed: pip install kiteconnect")
            return False
        api_key = os.getenv("ZERODHA_API_KEY")
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        if not (api_key and access_token) and os.path.exists(self.config_path):
            with open(self.config_path) as f:
                cfg = json.load(f).get("zerodha", {})
            api_key = api_key or cfg.get("api_key")
            access_token = access_token or cfg.get("access_token")
        if not (api_key and access_token):
            logger.error("Zerodha credentials missing (env or config.json)")
            return False
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        try:
            profile = self.kite.profile()
            logger.info("Zerodha connected as %s", profile.get("user_name"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Zerodha token invalid (run zerodha_auth.py): %s", exc)
            return False

    def get_quote(self, symbol: str) -> Quote:
        data = self.kite.ltp(f"NSE:{symbol}")
        return Quote(symbol=symbol, last_price=data[f"NSE:{symbol}"]["last_price"])

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: str = "CNC",
    ) -> OrderResult:
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=side,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
            )
            return OrderResult(success=True, order_id=str(order_id), message="Order placed")
        except Exception as exc:  # noqa: BLE001
            logger.error("Zerodha order failed: %s", exc)
            return OrderResult(success=False, message=str(exc))

    def get_positions(self) -> Dict[str, Position]:
        result = {}
        for item in self.kite.positions().get("net", []):
            if item["quantity"] != 0:
                result[item["tradingsymbol"]] = Position(
                    symbol=item["tradingsymbol"],
                    quantity=item["quantity"],
                    average_price=item["average_price"],
                    last_price=item["last_price"],
                )
        return result

    def get_holdings(self):
        return [
            Position(
                symbol=h["tradingsymbol"],
                quantity=h["quantity"],
                average_price=h["average_price"],
                last_price=h["last_price"],
            )
            for h in self.kite.holdings()
        ]

    def get_funds(self) -> Funds:
        margins = self.kite.margins()
        equity = margins.get("equity", {})
        return Funds(
            available_cash=equity.get("available", {}).get("cash", 0.0),
            used_margin=equity.get("utilised", {}).get("debits", 0.0),
        )
