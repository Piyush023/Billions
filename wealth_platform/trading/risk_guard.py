"""Code-enforced risk rules — the supervisor layer over LLM proposals."""

from typing import Dict, Optional, Tuple

from wealth_platform.trading.entry_filters import atr_stop_price, normalize_stops


class RiskGuard:
    MIN_TRADE_VALUE = 1000
    MAX_POSITION_PCT = 0.25
    MAX_PORTFOLIO_RISK_PCT = 0.025
    COST_HEADROOM = 1.005

    def __init__(self, config: dict, broker):
        self.config = config
        self.broker = broker
        self.capital = config.get("capital", 15000)

    def portfolio_snapshot(self) -> Dict:
        """Rich portfolio context for agents."""
        funds = self.broker.get_funds()
        positions = self.broker.get_positions()
        pos_lines = []
        total_equity = funds.available_cash
        for sym, p in positions.items():
            val = p.last_price * p.quantity
            total_equity += val
            pct = (p.last_price / p.average_price - 1) * 100 if p.average_price else 0
            pos_lines.append(
                f"  {sym}: qty={p.quantity}, avg=Rs.{p.average_price:.2f}, ltp=Rs.{p.last_price:.2f}, "
                f"P&L Rs.{p.pnl:+.0f} ({pct:+.1f}%), value=Rs.{val:.0f}"
            )
        return {
            "cash": funds.available_cash,
            "capital": self.capital,
            "total_equity": total_equity,
            "positions": positions,
            "position_lines": pos_lines,
            "position_count": len(positions),
        }

    def portfolio_context_text(self, desk_briefing: str = "") -> str:
        snap = self.portfolio_snapshot()
        lines = [
            f"Total capital: Rs.{self.capital:,.0f}",
            f"Available cash: Rs.{snap['cash']:,.2f}",
            f"Equity value (cash + holdings): Rs.{snap['total_equity']:,.2f}",
            f"Open positions ({snap['position_count']}):",
        ]
        lines.extend(snap["position_lines"] or ["  none"])
        if desk_briefing:
            lines.append("\n" + desk_briefing)
        return "\n".join(lines)

    def risk_reward_ok(self, entry: float, stop: float, target: float) -> Tuple[bool, str]:
        min_rr = self.config.get("min_risk_reward_ratio", 2.0)
        if entry <= 0 or stop <= 0 or target <= 0:
            return False, "missing entry/stop/target prices"
        risk = entry - stop
        reward = target - entry
        if risk <= 0:
            return False, "stop must be below entry"
        if reward <= 0:
            return False, "target must be above entry"
        rr = reward / risk
        if rr < min_rr:
            return False, f"risk/reward {rr:.1f}:1 below minimum {min_rr}:1"
        return True, f"R:R {rr:.1f}:1 OK"

    def fee_edge_ok(self, notional: float, entry: float, target: float) -> Tuple[bool, str]:
        """Block trades where expected gain cannot cover round-trip fees."""
        min_edge = self.config.get("min_edge_after_fees_pct", 1.5) / 100
        fees = self.config.get("round_trip_fee_rs", 120)
        if entry <= 0 or target <= entry or notional <= 0:
            return False, "invalid prices for fee check"
        gross_gain_pct = (target / entry - 1)
        fee_pct = fees / notional
        net_edge = gross_gain_pct - fee_pct
        if net_edge < min_edge:
            return False, (
                f"net edge after Rs.{fees:.0f} fees ({fee_pct*100:.1f}%) "
                f"is {net_edge*100:.1f}% — need {min_edge*100:.1f}%+"
            )
        return True, "fee-adjusted edge OK"

    def rotation_allowed(self, rotate_symbol: str) -> Tuple[bool, str]:
        """Only rotate out losers or clearly stagnant names — not working positions."""
        max_pnl = self.config.get("rotation_max_hold_pnl_pct", 2.0)
        pos = self.broker.get_positions().get(rotate_symbol)
        if not pos or not pos.average_price:
            return False, f"no position in {rotate_symbol} to rotate"
        pnl_pct = (pos.last_price / pos.average_price - 1) * 100
        if pnl_pct > max_pnl:
            return False, (
                f"rotation blocked: {rotate_symbol} at {pnl_pct:+.1f}% "
                f"(max {max_pnl:+.1f}% — won't sell winners to fund new trades)"
            )
        return True, f"rotation OK ({rotate_symbol} at {pnl_pct:+.1f}%)"

    def apply_stop_target(self, symbol: str, proposal: dict, last_price: float) -> dict:
        """Normalize and optionally ATR-adjust stop/target on proposals."""
        entry = float(proposal.get("entry_price") or last_price)
        stop = float(proposal.get("stop_loss") or 0)
        target = float(proposal.get("take_profit") or 0)
        atr_stop = atr_stop_price(symbol, entry, self.config)
        if atr_stop:
            stop = atr_stop if stop <= 0 else min(stop, atr_stop)  # wider of agent vs ATR floor
        stop, target = normalize_stops(entry, stop, target, self.config)
        proposal = dict(proposal)
        proposal["stop_loss"] = stop
        proposal["take_profit"] = target
        proposal["entry_price"] = entry
        return proposal

    def validate_and_resize(
        self,
        symbol: str,
        proposal: dict,
        pm_decision: dict,
        last_price: float,
    ) -> Tuple[bool, int, str]:
        """Enforce hard rules; return (approved, quantity, reason)."""
        if pm_decision.get("decision") != "APPROVE":
            return False, 0, pm_decision.get("reasoning", "PM rejected")

        action = proposal.get("action", "HOLD")
        if action not in ("BUY", "SELL"):
            return False, 0, "not a trade action"

        qty = int(pm_decision.get("adjusted_quantity") or proposal.get("quantity") or 0)
        if qty <= 0:
            return False, 0, "zero quantity"

        notional = qty * last_price * self.COST_HEADROOM
        funds = self.broker.get_funds()

        if action == "BUY":
            proposal = self.apply_stop_target(symbol, proposal, last_price)
            ok, msg = self.risk_reward_ok(
                proposal["entry_price"], proposal["stop_loss"], proposal["take_profit"],
            )
            if not ok:
                return False, 0, msg
            ok, msg = self.fee_edge_ok(notional, proposal["entry_price"], proposal["take_profit"])
            if not ok:
                return False, 0, msg

            if notional < self.MIN_TRADE_VALUE:
                return False, 0, f"trade value Rs.{notional:.0f} below min Rs.{self.MIN_TRADE_VALUE}"
            max_notional = self.capital * self.MAX_POSITION_PCT
            if notional > max_notional:
                qty = max(1, int(max_notional / (last_price * self.COST_HEADROOM)))
                notional = qty * last_price * self.COST_HEADROOM
            if notional > funds.available_cash:
                qty = max(0, int(funds.available_cash / (last_price * self.COST_HEADROOM)))
                notional = qty * last_price * self.COST_HEADROOM
            if notional < self.MIN_TRADE_VALUE:
                return False, 0, "insufficient cash for minimum trade after resize"

            stop = float(proposal.get("stop_loss") or last_price * (1 - self.config.get("stop_loss_pct", 6) / 100))
            risk_per_share = max(last_price - stop, last_price * 0.01)
            portfolio_risk = risk_per_share * qty
            max_risk = self.capital * self.MAX_PORTFOLIO_RISK_PCT
            if portfolio_risk > max_risk:
                qty = max(1, int(max_risk / risk_per_share))
                notional = qty * last_price * self.COST_HEADROOM

        elif action == "SELL":
            held = self.broker.get_positions().get(symbol)
            if not held or held.quantity < qty:
                return False, 0, "insufficient shares to sell"

        return True, qty, "approved"

    def price_drift_ok(self, old_price: float, new_price: float, max_pct: float = 1.5) -> Tuple[bool, str]:
        if old_price <= 0:
            return True, ""
        drift = abs(new_price / old_price - 1) * 100
        if drift > max_pct:
            return False, f"price drifted {drift:.1f}% since analysis (limit {max_pct}%)"
        return True, ""
