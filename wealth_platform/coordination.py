"""SharedDesk — the coordination layer between the buy-side cycle loop and
the sell-side Portfolio Sentinel.

Concurrency model (deliberately simple to make deadlock impossible):

  1. ONE global trade lock (RLock) serializes every money-moving critical
     section: quote → funds check → PM decision → order placement on the buy
     side; stop enforcement and exits on the sell side. Reads (dashboard,
     monitoring) never take it. A single lock means there is no lock
     ordering, hence no deadlock — only brief waits (an order takes <2s).

  2. SharedDesk holds the cross-agent knowledge, guarded by its own internal
     mutex and persisted to data/desk_state.json:
       - position_theses : why we bought each holding (entry reasoning,
         stop, target). Written by buy side, read by the sentinel so it
         monitors the ORIGINAL thesis, not just price.
       - sentinel_notes  : the sentinel's latest verdict per held symbol.
         Read by the buy-side PM before approving any new trade.
       - recent_exits    : what was sold, when, why, at what P&L. The buy
         side refuses to re-enter a symbol within the cool-off window, so
         the two agents can't ping-pong (sentinel sells, cycle re-buys).

  SharedDesk methods never call broker/LLM/storage code, so holding its
  mutex can never wait on the trade lock — the two locks are independent.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger("wealth_platform.coordination")

STATE_PATH = os.path.join("data", "desk_state.json")

# Single global lock for all order-placing critical sections.
TRADE_LOCK = threading.RLock()


class SharedDesk:
    def __init__(self, path: str = STATE_PATH, exit_cooloff_days: int = 3):
        self._mutex = threading.Lock()
        self.path = path
        self.exit_cooloff_days = exit_cooloff_days
        self.position_theses: Dict[str, dict] = {}
        self.sentinel_notes: Dict[str, dict] = {}
        self.recent_exits: Dict[str, dict] = {}
        self._load()

    # ---- buy side writes ------------------------------------------------

    def record_entry(self, symbol: str, reasoning: str, stop_loss: float, take_profit: float):
        with self._mutex:
            self.position_theses[symbol] = {
                "reasoning": reasoning[:400],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "opened_at": datetime.now().isoformat(),
            }
            self._save()

    # ---- sentinel writes ------------------------------------------------

    def record_sentinel_note(self, symbol: str, action: str, reasoning: str, confidence: int):
        with self._mutex:
            self.sentinel_notes[symbol] = {
                "action": action,
                "reasoning": reasoning[:300],
                "confidence": confidence,
                "at": datetime.now().isoformat(),
            }
            self._save()

    # ---- either side on exit ---------------------------------------------

    def record_exit(self, symbol: str, reason: str, pnl_pct: float):
        with self._mutex:
            self.recent_exits[symbol] = {
                "reason": reason[:300],
                "pnl_pct": pnl_pct,
                "at": datetime.now().isoformat(),
            }
            self.position_theses.pop(symbol, None)
            self.sentinel_notes.pop(symbol, None)
            self._save()

    # ---- reads -----------------------------------------------------------

    def thesis_for(self, symbol: str) -> Optional[dict]:
        with self._mutex:
            return self.position_theses.get(symbol)

    def in_exit_cooloff(self, symbol: str) -> bool:
        """Buy side must not re-enter a symbol shortly after an exit."""
        with self._mutex:
            info = self.recent_exits.get(symbol)
        if not info:
            return False
        exited = datetime.fromisoformat(info["at"])
        return datetime.now() - exited < timedelta(days=self.exit_cooloff_days)

    def pm_briefing(self) -> str:
        """Cross-agent context injected into the buy-side PM's decisions."""
        with self._mutex:
            lines = []
            if self.sentinel_notes:
                lines.append("Sentinel's latest view on current holdings:")
                for s, n in self.sentinel_notes.items():
                    lines.append(f"- {s}: {n['action']} (conf {n['confidence']}) — {n['reasoning']}")
            if self.recent_exits:
                lines.append("Recently exited (do NOT re-enter during cool-off):")
                for s, e in self.recent_exits.items():
                    lines.append(f"- {s}: {e['pnl_pct']:+.1f}% — {e['reason']}")
            return "\n".join(lines) if lines else "No sentinel notes or recent exits."

    # ---- persistence -----------------------------------------------------

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "position_theses": self.position_theses,
                "sentinel_notes": self.sentinel_notes,
                "recent_exits": self.recent_exits,
            }, f, indent=2)
        os.replace(tmp, self.path)  # atomic — no torn writes on crash

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    data = json.load(f)
                self.position_theses = data.get("position_theses", {})
                self.sentinel_notes = data.get("sentinel_notes", {})
                self.recent_exits = data.get("recent_exits", {})
        except Exception:  # noqa: BLE001
            logger.exception("Could not load desk state; starting fresh")
