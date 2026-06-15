"""Cycle-level portfolio planner — one LLM call picks which symbols deserve full analysis."""

import logging
from typing import Callable, List, Optional

from wealth_platform.agents.base_agent import BaseAgent

logger = logging.getLogger("wealth_platform.portfolio_planner")


class PortfolioPlanner(BaseAgent):
    name = "portfolio_planner"
    role = "Portfolio Planner"
    max_tokens = 600
    system_prompt = (
        "You are the desk's portfolio planner for a small Indian equity CNC account. "
        "Given available cash, current holdings, sentinel notes, screener candidates, and "
        "news picks, choose which 1-3 symbols deserve a full multi-agent analysis THIS cycle. "
        "Prefer: filling underweight sectors, strong momentum with catalyst, symbols not in cool-off. "
        "Avoid: re-analyzing held stocks unless rotation is intended, low-cash cycles with no rotation need. "
        'Respond ONLY with JSON: {"symbols": ["SYM1", "SYM2"], "reasoning": "<2 sentences>"}'
    )

    def plan(
        self,
        candidates: List[str],
        portfolio_context: str,
        desk_briefing: str,
        memory_snippet: str,
        max_n: int = 3,
    ) -> List[str]:
        if not candidates:
            return []
        if len(candidates) <= max_n:
            return candidates[:max_n]

        user = (
            f"Pick up to {max_n} symbols from: {candidates}\n\n"
            f"PORTFOLIO:\n{portfolio_context}\n\n"
            f"DESK:\n{desk_briefing or 'none'}\n\n"
            f"LESSONS:\n{memory_snippet[:1500] or 'none'}"
        )
        try:
            result = self.llm.chat_json(
                self.system_prompt + ("\n\n" + self.style_suffix if self.style_suffix else ""),
                user,
                max_tokens=self.max_tokens,
            )
            picked = [s.upper() for s in result.get("symbols", []) if isinstance(s, str)]
            picked = [s for s in picked if s in candidates][:max_n]
            if picked:
                logger.info("Planner selected: %s — %s", picked, result.get("reasoning", ""))
                if self.on_output:
                    from wealth_platform.agents.base_agent import AgentOutput
                    self.on_output(AgentOutput(
                        self.name, self.role,
                        f"Selected: {picked}\n{result.get('reasoning', '')}",
                        "planner", 0,
                    ))
                return picked
        except Exception as exc:  # noqa: BLE001
            logger.warning("Portfolio planner failed (%s); using screener order", exc)
        return candidates[:max_n]
