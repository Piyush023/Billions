"""Retrieval layer over the trade/decision history in SQLite.

Before the Trader and Portfolio Manager decide on a stock, this pulls:
  1. That symbol's full decision + trade + outcome history
  2. Overall performance statistics (win rate, avg win/loss, rejection patterns)
and renders them as compact text injected into the agents' context.

This is RAG without embeddings: at this data volume (a handful of rows/day),
exact SQL retrieval per symbol beats vector search — zero cost, perfectly
relevant. If history ever grows to thousands of records, swap `for_symbol`
for an embedding search without touching the agents.
"""

import json
from typing import List

from wealth_platform.storage import Storage


class TradeHistoryRAG:
    def __init__(self, storage: Storage):
        self.storage = storage

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def for_symbol(self, symbol: str, limit: int = 8) -> str:
        """Past decisions and trades for one symbol, newest first."""
        decisions = self.storage._rows(
            "SELECT * FROM decisions WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit)
        )
        trades = self.storage._rows(
            "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit)
        )
        if not decisions and not trades:
            return f"No prior history for {symbol}."

        lines: List[str] = [f"Prior history for {symbol}:"]
        for d in decisions:
            pm = json.loads(d["pm_decision"] or "{}")
            prop = json.loads(d["trade_proposal"] or "{}")
            lines.append(
                f"- {d['created_at'][:10]}: research={d['research_rating']}, "
                f"proposal={prop.get('action', '-')} x{prop.get('quantity', '-')}, "
                f"PM={pm.get('decision', '-')}"
                + (f" — {pm.get('reasoning', '')[:120]}" if pm.get("reasoning") else "")
            )
        for t in trades:
            lines.append(
                f"- {t['created_at'][:10]} TRADE: {t['side']} {t['quantity']} @ {t['price']:.2f} "
                f"[{t['status']}] {t['reasoning'][:100]}"
            )
        return "\n".join(lines)

    def performance_stats(self) -> str:
        """Aggregate outcomes so agents see what has and hasn't worked."""
        trades = self.storage._rows(
            "SELECT * FROM trades WHERE status='filled' ORDER BY id"
        )
        decisions = self.storage._rows("SELECT * FROM decisions ORDER BY id")
        if not trades and not decisions:
            return "No performance history yet — this desk has not completed any trades."

        # Pair buys with sells per symbol (FIFO) to estimate realized outcomes
        open_lots: dict = {}
        realized: List[float] = []
        for t in trades:
            sym = t["symbol"]
            if t["side"] == "BUY":
                open_lots.setdefault(sym, []).append(t)
            elif t["side"] == "SELL" and open_lots.get(sym):
                buy = open_lots[sym].pop(0)
                if buy["price"]:
                    realized.append((t["price"] / buy["price"] - 1) * 100)

        approved = sum(1 for d in decisions if d["approved"])
        rejected = len(decisions) - approved
        lines = [
            f"Decisions to date: {len(decisions)} ({approved} approved, {rejected} rejected/skipped).",
            f"Filled orders: {len(trades)}. Completed round trips: {len(realized)}.",
        ]
        if realized:
            wins = [r for r in realized if r > 0]
            losses = [r for r in realized if r <= 0]
            lines.append(
                f"Win rate: {len(wins)}/{len(realized)} ({100 * len(wins) / len(realized):.0f}%). "
                f"Avg win: {sum(wins) / len(wins):+.1f}%" if wins else "No winning trades yet."
            )
            if losses:
                lines.append(f"Avg loss: {sum(losses) / len(losses):+.1f}%.")
        return "\n".join(lines)

    def build_context(self, symbol: str) -> str:
        """Everything an agent should know from history, ready to inject."""
        return (
            "=== HISTORICAL RECORD (from this desk's own database) ===\n"
            f"{self.performance_stats()}\n\n{self.for_symbol(symbol)}\n"
            "Use this record: avoid repeating past mistakes, respect setups that "
            "previously failed, and weigh confidence accordingly."
        )
