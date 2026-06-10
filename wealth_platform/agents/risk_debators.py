"""Risk management debate: aggressive vs conservative vs neutral on a trade proposal."""

from wealth_platform.agents.base_agent import BaseAgent


class AggressiveDebator(BaseAgent):
    name = "aggressive_debator"
    role = "Aggressive Risk Debator"
    system_prompt = (
        "You are the aggressive risk analyst. Given a trade proposal and market context, argue why "
        "the desk should take the risk — emphasise upside, opportunity cost of inaction, and why "
        "the sizing could even be larger. Critique overly cautious framing. Stay factual."
    )
    max_tokens = 700


class ConservativeDebator(BaseAgent):
    name = "conservative_debator"
    role = "Conservative Risk Debator"
    system_prompt = (
        "You are the conservative risk analyst protecting small retail capital. Given a trade "
        "proposal and market context, argue for capital preservation — emphasise downside scenarios, "
        "transaction-cost drag, correlation with existing positions, and when to skip the trade."
    )
    max_tokens = 700


class NeutralDebator(BaseAgent):
    name = "neutral_debator"
    role = "Neutral Risk Debator"
    system_prompt = (
        "You are the neutral risk analyst. You have heard the aggressive and conservative arguments "
        "on a trade proposal. Weigh both, point out where each exaggerates, and give a balanced "
        "recommendation on whether the proposal's size and stops are appropriate."
    )
    max_tokens = 700


def run_risk_debate(
    aggressive: AggressiveDebator,
    conservative: ConservativeDebator,
    neutral: NeutralDebator,
    proposal: dict,
    portfolio_context: str,
) -> str:
    base = f"TRADE PROPOSAL: {proposal}\n\nPORTFOLIO CONTEXT:\n{portfolio_context}"
    agg = aggressive.run(base)
    con = conservative.run(base + f"\n\nAGGRESSIVE ARGUMENT:\n{agg.report}")
    neu = neutral.run(
        base
        + f"\n\nAGGRESSIVE ARGUMENT:\n{agg.report}\n\nCONSERVATIVE ARGUMENT:\n{con.report}"
    )
    return (
        f"--- AGGRESSIVE ---\n{agg.report}\n\n"
        f"--- CONSERVATIVE ---\n{con.report}\n\n"
        f"--- NEUTRAL ---\n{neu.report}"
    )
