"""Researcher team: bull/bear debate plus the Research Manager synthesis."""

from wealth_platform.agents.base_agent import BaseAgent


class BullResearcher(BaseAgent):
    name = "bull_researcher"
    role = "Bull Researcher"
    system_prompt = (
        "You are the bullish researcher in an investment debate. Using the four analyst reports, "
        "build the strongest evidence-based case FOR buying this stock. Emphasise growth potential, "
        "favourable technicals, and positive catalysts. If a bear argument is provided, rebut its "
        "strongest points directly. Be persuasive but never invent facts not present in the reports."
    )
    max_tokens = 1200


class BearResearcher(BaseAgent):
    name = "bear_researcher"
    role = "Bear Researcher"
    system_prompt = (
        "You are the bearish researcher in an investment debate. Using the four analyst reports, "
        "build the strongest evidence-based case AGAINST buying this stock. Emphasise risks, weak "
        "fundamentals, negative catalysts, and unfavourable technicals. Rebut the bull's strongest "
        "points directly. Be persuasive but never invent facts not present in the reports."
    )
    max_tokens = 1200


class ResearchManager(BaseAgent):
    name = "research_manager"
    role = "Research Manager"
    system_prompt = (
        "You are the research manager. You have the analyst reports and a bull-vs-bear debate. "
        "Weigh the arguments and commit to a clear stance — do not default to HOLD unless evidence "
        "is genuinely balanced. Respond with a JSON object: "
        "This desk trades positionally (2-12 week holds, delivery only) — judge the stock on its "
        "multi-week outlook, not intraday momentum. Respond with a JSON object: "
        '{"rating": "BUY"|"OVERWEIGHT"|"HOLD"|"UNDERWEIGHT"|"SELL", '
        '"confidence": 0-100, "rationale": "<3-5 sentences>", '
        '"key_risks": ["..."], "time_horizon": "swing"|"positional"}'
    )
    max_tokens = 1000

    def decide(self, symbol: str, analyst_reports: str, debate_transcript: str) -> dict:
        context = (
            f"Stock: {symbol}\n\nANALYST REPORTS:\n{analyst_reports}\n\n"
            f"DEBATE TRANSCRIPT:\n{debate_transcript}"
        )
        output = self.run(context)
        from wealth_platform.llm.llm_client import extract_json

        try:
            return extract_json(output.report)
        except ValueError:
            return {
                "rating": "HOLD",
                "confidence": 0,
                "rationale": "Research manager output could not be parsed; defaulting to HOLD.",
                "key_risks": ["unparseable LLM output"],
                "time_horizon": "swing",
            }


def run_debate(bull: BullResearcher, bear: BearResearcher, analyst_reports: str, rounds: int = 1) -> str:
    """Alternate bull and bear for N rounds; return the full transcript."""
    transcript = ""
    bear_argument = ""
    for round_no in range(1, rounds + 1):
        bull_context = f"ANALYST REPORTS:\n{analyst_reports}"
        if bear_argument:
            bull_context += f"\n\nBEAR'S LAST ARGUMENT (rebut this):\n{bear_argument}"
        bull_out = bull.run(bull_context)
        transcript += f"\n--- Round {round_no} | BULL ---\n{bull_out.report}\n"

        bear_context = (
            f"ANALYST REPORTS:\n{analyst_reports}\n\nBULL'S ARGUMENT (rebut this):\n{bull_out.report}"
        )
        bear_out = bear.run(bear_context)
        transcript += f"\n--- Round {round_no} | BEAR ---\n{bear_out.report}\n"
        bear_argument = bear_out.report
    return transcript
