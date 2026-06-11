"""Portfolio Manager: the final LLM gatekeeper that approves or rejects trades.

This is the only agent that prefers the paid Anthropic provider (Claude Haiku)
when ANTHROPIC_API_KEY is set; otherwise it runs on the free tier like everyone
else. It also receives lessons from past decisions via AgentMemory.
"""

from wealth_platform.agents.base_agent import BaseAgent
from wealth_platform.llm.llm_client import extract_json


class PortfolioManagerAgent(BaseAgent):
    name = "portfolio_manager"
    role = "Portfolio Manager"
    prefer_provider = "anthropic"
    max_tokens = 900
    system_prompt = (
        "You are the portfolio manager with final authority over every trade at a small Indian "
        "retail desk. You receive the research verdict, the trader's proposal, the full risk debate, "
        "current portfolio state, and lessons from past decisions. Approve only trades where the "
        "evidence, sizing, and stops are sound AND the portfolio can absorb the loss if the stop is "
        "hit. Hard rules you must enforce: never approve if quantity*entry_price exceeds available "
        "cash; never approve more than 25% of total capital in one stock; never approve if the "
        "stop-loss implies losing more than 2.5% of total capital. "
        "CAPITAL ROTATION: if cash is insufficient but this opportunity is clearly stronger than an "
        "existing holding (better momentum, fresher catalyst, the holding is stagnant or its thesis "
        "is weakening per the sentinel's notes), you may approve the trade funded by selling that "
        "holding — set fund_by_selling to its symbol. Only rotate when the new setup is decisively "
        "better; rotation costs ~Rs.120 in fees. Respond with a JSON object: "
        '{"decision": "APPROVE"|"REJECT", "adjusted_quantity": <int or null>, '
        '"fund_by_selling": "<symbol or null>", '
        '"reasoning": "<3-5 sentences>", "confidence": 0-100}'
    )

    def decide(
        self,
        symbol: str,
        research_decision: dict,
        trade_proposal: dict,
        risk_debate: str,
        portfolio_context: str,
        memory_lessons: str,
    ) -> dict:
        context = (
            f"Stock: {symbol}\n\n"
            f"RESEARCH VERDICT: {research_decision}\n\n"
            f"TRADE PROPOSAL: {trade_proposal}\n\n"
            f"RISK DEBATE:\n{risk_debate}\n\n"
            f"PORTFOLIO STATE:\n{portfolio_context}\n\n"
            f"LESSONS FROM PAST DECISIONS:\n{memory_lessons or 'No history yet.'}"
        )
        output = self.run(context)
        try:
            decision = extract_json(output.report)
        except ValueError:
            return {"decision": "REJECT", "reasoning": "PM output unparseable; rejecting for safety.", "confidence": 0}
        decision.setdefault("decision", "REJECT")
        return decision
