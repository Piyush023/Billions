"""Single-call analyst panel for compact pipeline mode."""

from wealth_platform.agents.base_agent import BaseAgent


class AnalystPanelAgent(BaseAgent):
    name = "analyst_panel"
    role = "Analyst Panel"
    max_tokens = 1400
    system_prompt = (
        "You are a combined equity research panel for NSE positional trading (2-12 week holds). "
        "Given market data, fundamentals, news, and enrichment, produce ONE structured assessment. "
        "Respond ONLY with JSON: "
        '{"technical": {"verdict": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0-100, "summary": "..."}, '
        '"fundamentals": {"verdict": "UNDERVALUED"|"FAIRLY VALUED"|"OVERVALUED", "confidence": 0-100, "summary": "..."}, '
        '"news": {"verdict": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "confidence": 0-100, "summary": "..."}, '
        '"rating": "BUY"|"OVERWEIGHT"|"HOLD"|"UNDERWEIGHT"|"SELL", '
        '"confidence": 0-100, "rationale": "...", "key_risks": ["..."], "time_horizon": "positional"}'
    )

    def analyze(self, symbol: str, data_bundle: str) -> dict:
        try:
            return self.llm.chat_json(
                self.system_prompt + ("\n\n" + self.style_suffix if self.style_suffix else ""),
                f"Stock: {symbol}\n\n{data_bundle}",
                max_tokens=self.max_tokens,
            )
        except Exception:  # noqa: BLE001
            return {
                "rating": "HOLD", "confidence": 0,
                "rationale": "Analyst panel parse failed.",
                "key_risks": ["parse error"],
            }


class CompactDecisionAgent(BaseAgent):
    """Combined trader + PM in one call for compact/minimal modes."""

    name = "compact_decision"
    role = "Trader & Portfolio Manager"
    prefer_provider = "anthropic"
    max_tokens = 900
    system_prompt = (
        "You are trader AND portfolio manager for a small Indian CNC desk. "
        "Given research verdict, analyst context, price, cash, positions, and lessons, "
        "produce a final trade decision. Enforce: min Rs.1000 trade, max 25% capital per stock, "
        "no short selling, positional holds only. "
        'Respond ONLY with JSON: '
        '{"decision": "APPROVE"|"REJECT", "action": "BUY"|"SELL"|"HOLD", "quantity": <int>, '
        '"entry_price": <float>, "stop_loss": <float>, "take_profit": <float>, '
        '"fund_by_selling": "<symbol or null>", "reasoning": "...", "confidence": 0-100}'
    )

    def decide(self, symbol: str, context: str) -> dict:
        try:
            out = self.llm.chat_json(
                self.system_prompt + ("\n\n" + self.style_suffix if self.style_suffix else ""),
                context,
                max_tokens=self.max_tokens,
                prefer=self.prefer_provider,
            )
            out.setdefault("decision", "REJECT")
            out.setdefault("action", "HOLD")
            out.setdefault("quantity", 0)
            return out
        except Exception:  # noqa: BLE001
            return {"decision": "REJECT", "action": "HOLD", "quantity": 0, "reasoning": "parse failed"}
