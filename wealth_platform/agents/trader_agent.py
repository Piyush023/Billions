"""Trader agent: converts the research verdict into a concrete trade proposal."""

from wealth_platform.agents.base_agent import BaseAgent
from wealth_platform.llm.llm_client import extract_json


class TraderAgent(BaseAgent):
    name = "trader"
    role = "Trader"
    system_prompt = (
        "You are a disciplined POSITIONAL trader at an Indian equity desk trading small retail capital "
        "(Rs.10,000-50,000 total), where transaction costs are roughly Rs.120 per round trip. "
        "Hold delivery (CNC) trades for 2-8 weeks targeting +10-15%. Only propose BUY when the "
        "multi-week thesis is strong with at least 2:1 reward/risk vs your stop. "
        "Stop-loss should be 5-8% below entry (volatility-aware); take-profit 10-15% above entry. "
        "Skip the trade (action HOLD) if expected edge cannot clear Rs.120 fees plus 1.5% buffer. "
        "Position size: max 25% of capital per stock, minimum trade value Rs.1,000. "
        "Short selling is impossible — only propose SELL for stocks currently held. "
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
