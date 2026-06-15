"""Code-enforced risk rules — the supervisor layer over LLM proposals."""

from typing import Dict, Optional, Tuple


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
        sector_exposure: Dict[str, float] = {}
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

            stop = float(proposal.get("stop_loss") or last_price * (1 - self.config.get("stop_loss_pct", 4) / 100))
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
