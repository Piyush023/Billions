"""Trader agent: converts the research verdict into a concrete trade proposal."""

from wealth_platform.agents.base_agent import BaseAgent
from wealth_platform.llm.llm_client import extract_json


class TraderAgent(BaseAgent):
    name = "trader"
    role = "Trader"
    system_prompt = (
        "You are a disciplined POSITIONAL trader at an Indian equity desk trading small retail capital "
        "(Rs.10,000-50,000 total), where transaction costs are roughly Rs.60 per round trip. "
        "This desk does NOT day trade: every position is a delivery (CNC) trade intended to be held "
        "for 2-12 weeks. Only propose BUY when the multi-week thesis is strong — quality business or "
        "clear technical setup with room to the +12-20% range. Never propose a trade for an intraday "
        "or few-day move; if the edge is short-lived, the action is HOLD. "
        "Given the research manager's verdict, analyst reports, current price, and available capital, "
        "produce a concrete trade proposal. Skip the trade (action HOLD) if expected edge cannot "
        "clear costs comfortably. Position size must respect: max 25% of capital per stock, "
        "minimum trade value Rs.1,000. Short selling is impossible — you may only propose SELL for "
        "stocks currently held in open positions. A bearish view on a stock we do not own means "
        "action HOLD (avoid), not SELL. "
        "Respond with a JSON object: "
        '{"action": "BUY"|"SELL"|"HOLD", "quantity": <int>, "entry_price": <float>, '
        '"stop_loss": <float>, "take_profit": <float>, "trade_type": "positional", '
        '"reasoning": "<2-4 sentences>", "confidence": 0-100}'
    )
    max_tokens = 800

    def propose(
        self,
        symbol: str,
        research_decision: dict,
        analyst_summary: str,
        current_price: float,
        available_cash: float,
        open_positions: dict,
    ) -> dict:
        context = (
            f"Stock: {symbol}\n"
            f"Current price: Rs.{current_price:.2f}\n"
            f"Available cash: Rs.{available_cash:.2f}\n"
            f"Open positions: {open_positions or 'none'}\n\n"
            f"RESEARCH MANAGER VERDICT: {research_decision}\n\n"
            f"ANALYST SUMMARY:\n{analyst_summary}"
        )
        output = self.run(context)
        try:
            proposal = extract_json(output.report)
        except ValueError:
            return {"action": "HOLD", "reasoning": "Trader output unparseable; no trade.", "confidence": 0}
        proposal.setdefault("action", "HOLD")
        proposal.setdefault("quantity", 0)
        proposal.setdefault("confidence", 0)
        return proposal
