"""Decision memory: logs every cycle outcome so the Portfolio Manager learns.

Stored as markdown in data/agent_memory.md — readable by you, fed back to the
PM agent as 'lessons from past decisions' on each run.
"""

import os
from datetime import datetime

MEMORY_PATH = os.path.join("data", "agent_memory.md")
MAX_LESSONS_CHARS = 4000


class AgentMemory:
    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, symbol: str, decision: str, rating: str, reasoning: str, outcome_pct: float = None):
        entry = (
            f"\n## {datetime.now():%Y-%m-%d %H:%M} | {symbol}\n"
            f"- Decision: {decision} (research rating: {rating})\n"
            f"- Reasoning: {reasoning}\n"
        )
        if outcome_pct is not None:
            entry += f"- Realized outcome: {outcome_pct:+.2f}%\n"
        with open(self.path, "a") as f:
            f.write(entry)

    def record_outcome(self, symbol: str, outcome_pct: float, note: str = ""):
        entry = (
            f"\n## {datetime.now():%Y-%m-%d %H:%M} | {symbol} OUTCOME\n"
            f"- Realized: {outcome_pct:+.2f}% {note}\n"
        )
        with open(self.path, "a") as f:
            f.write(entry)

    def lessons(self) -> str:
        """Most recent slice of memory, newest entries favoured."""
        if not os.path.exists(self.path):
            return ""
        with open(self.path) as f:
            content = f.read()
        return content[-MAX_LESSONS_CHARS:]
