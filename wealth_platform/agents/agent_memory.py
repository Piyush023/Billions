"""Decision memory: closed-loop trade lifecycle + weekly summarization."""

import os
import re
from datetime import datetime, timedelta

from wealth_platform.paths import AGENT_MEMORY_PATH, AGENT_MEMORY_SUMMARY_PATH, ensure_data_dir

MEMORY_PATH = AGENT_MEMORY_PATH
SUMMARY_PATH = AGENT_MEMORY_SUMMARY_PATH
ensure_data_dir()
MAX_LESSONS_CHARS = 4000


class AgentMemory:
    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self.summary_path = SUMMARY_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, symbol: str, decision: str, rating: str, reasoning: str, outcome_pct: float = None):
        entry = (
            f"\n## {datetime.now():%Y-%m-%d %H:%M} | {symbol}\n"
            f"- Decision: {decision} (research rating: {rating})\n"
            f"- Reasoning: {reasoning}\n"
        )
        if outcome_pct is not None:
            entry += f"- Realized outcome: {outcome_pct:+.2f}%\n"
        self._append(entry)

    def record_trade_lifecycle(
        self,
        symbol: str,
        side: str,
        rating: str,
        thesis: str,
        fill_price: float,
        quantity: int,
        stop_loss: float = 0,
        take_profit: float = 0,
        success: bool = True,
    ):
        status = "FILLED" if success else "FAILED"
        entry = (
            f"\n## {datetime.now():%Y-%m-%d %H:%M} | {symbol} TRADE {status}\n"
            f"- {side} {quantity} @ Rs.{fill_price:.2f} (research: {rating})\n"
            f"- Thesis: {thesis[:300]}\n"
            f"- Stop: Rs.{stop_loss:.2f} | Target: Rs.{take_profit:.2f}\n"
        )
        self._append(entry)

    def record_outcome(self, symbol: str, outcome_pct: float, note: str = "", pnl_rs: float = None):
        entry = (
            f"\n## {datetime.now():%Y-%m-%d %H:%M} | {symbol} OUTCOME\n"
            f"- Realized: {outcome_pct:+.2f}% {note}\n"
        )
        if pnl_rs is not None:
            entry += f"- P&L: Rs.{pnl_rs:+,.2f}\n"
        self._append(entry)

    def symbol_lessons(self, symbol: str, max_chars: int = 1200) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path) as f:
            content = f.read()
        blocks = re.split(r"\n(?=## )", content)
        relevant = [b for b in blocks if symbol.upper() in b.upper()]
        if not relevant:
            return f"No prior lessons for {symbol}."
        # Surface losing outcomes first so agents avoid repeating mistakes
        def sort_key(block: str) -> tuple:
            loss = "OUTCOME" in block and ("-" in block or "Realized: -" in block)
            pnl_hit = "P&L: Rs.-" in block or "Realized: -" in block
            return (0 if (loss or pnl_hit) else 1, block)
        relevant.sort(key=sort_key)
        return f"Past lessons on {symbol} (losses prioritized):\n" + "\n".join(relevant[-4:])[-max_chars:]

    def lessons(self) -> str:
        parts = []
        if os.path.exists(self.summary_path):
            with open(self.summary_path) as f:
                parts.append(f"=== COMPRESSED LESSONS ===\n{f.read()[-2000:]}")
        if os.path.exists(self.path):
            with open(self.path) as f:
                parts.append(f.read()[-MAX_LESSONS_CHARS:])
        return "\n\n".join(parts)

    def summarize_weekly(self, llm) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path) as f:
            content = f.read()
        week_ago = datetime.now() - timedelta(days=7)
        recent = []
        for block in re.split(r"\n(?=## )", content):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", block)
            if m:
                try:
                    if datetime.fromisoformat(m.group(1)) >= week_ago.replace(hour=0, minute=0, second=0):
                        recent.append(block)
                except ValueError:
                    pass
        if not recent:
            return ""
        system = (
            "Compress trading desk memory into max 10 bullet lessons for the portfolio manager. "
            "Focus on what worked, what failed, and patterns to avoid. Plain text bullets only."
        )
        try:
            summary = llm.chat(system, "\n".join(recent[-30:]), max_tokens=500).text
            with open(self.summary_path, "w") as f:
                f.write(f"# Weekly summary {datetime.now():%Y-%m-%d}\n\n{summary}\n")
            return summary
        except Exception:  # noqa: BLE001
            return ""

    def _append(self, entry: str):
        with open(self.path, "a") as f:
            f.write(entry)
