"""Token/call budget guard — degrades pipeline gracefully on free tiers."""

import logging
from typing import List

logger = logging.getLogger("wealth_platform.token_budget")


class TokenBudget:
    MODE_FULL = "full"
    MODE_STANDARD = "standard"
    MODE_COMPACT = "compact"
    MODE_MINIMAL = "minimal"

    def __init__(self, storage, config: dict):
        self.storage = storage
        self.config = config
        self.daily_limit = config.get("daily_llm_call_budget", 180)
        self._calls_today = 0
        self._refresh()

    def _refresh(self):
        counts = self.storage.provider_counts_today()
        self._calls_today = sum(r["calls"] for r in counts)

    def mode(self) -> str:
        self._refresh()
        configured = self.config.get("pipeline_mode", "auto")
        if configured != "auto":
            return configured
        ratio = self._calls_today / max(self.daily_limit, 1)
        if ratio >= 0.85:
            return self.MODE_MINIMAL
        if ratio >= 0.65:
            return self.MODE_COMPACT
        if ratio >= 0.45:
            return self.MODE_STANDARD
        return self.MODE_FULL

    def calls_remaining(self) -> int:
        self._refresh()
        return max(0, self.daily_limit - self._calls_today)

    def allow_debate(self, needs_debate: bool) -> bool:
        m = self.mode()
        if m in (self.MODE_MINIMAL, self.MODE_COMPACT):
            return False
        if m == self.MODE_STANDARD:
            return needs_debate
        return True

    def allow_risk_debate(self, borderline: bool, position_pct: float) -> bool:
        m = self.mode()
        if m == self.MODE_MINIMAL:
            return False
        if m == self.MODE_COMPACT:
            return borderline and position_pct > 0.12
        if m == self.MODE_STANDARD:
            return borderline or position_pct > 0.15
        # full: only on borderline or large size
        return borderline or position_pct > 0.15

    def analyst_modules(self) -> List[str]:
        m = self.mode()
        if m == self.MODE_MINIMAL:
            return ["technical", "news"]
        if m == self.MODE_COMPACT:
            return ["panel"]  # single combined call
        return ["technical", "fundamentals", "news"]

    def allow_ipo_scan(self) -> bool:
        return self.mode() == self.MODE_FULL and self.config.get("ipo_enabled", True)

    def log_status(self):
        logger.info(
            "Token budget: %d/%d calls today, mode=%s",
            self._calls_today, self.daily_limit, self.mode(),
        )
