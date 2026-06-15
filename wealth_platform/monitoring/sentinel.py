"""Portfolio Sentinel — the parallel guardian agent for OWNED positions.

Runs endlessly in its own thread, independent of the buy-side decision cycle:

    BUY SIDE (cycle loop)               SELL SIDE (this sentinel)
    ─────────────────────               ─────────────────────────
    screener + news discovery           every N minutes, for each HELD stock:
    → 5-stage agent pipeline              → live price vs entry/stop/target
    → PM approves → BUY                   → fresh news on that company
                                          → LLM verdict: HOLD / TIGHTEN_STOP /
                                            EXIT_NOW (+ reasoning)
                                          → acts: tightens stop or sells

Three protection layers on every position:
  1. Hard math stops (orchestrator.manage_exits, every 5 min) — works even
     if every LLM is down
  2. Sentinel-tightened stops (this module, enforced mathematically each pass)
  3. Sentinel LLM exits — news-driven, e.g. "results missed badly, exit now"

Token budget: ONE batched LLM call per pass covering all positions
(~10-40 calls/day at the default 15-minute interval). No positions = no calls.
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Dict, Optional

import requests

from wealth_platform.trading.exit_levels import exit_reason

logger = logging.getLogger("wealth_platform.monitoring.sentinel")

from wealth_platform.paths import SENTINEL_STATE_PATH

STATE_PATH = SENTINEL_STATE_PATH
EXIT_CONFIDENCE_THRESHOLD = 75


class PortfolioSentinel:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.stop_overrides: Dict[str, float] = {}  # symbol -> tightened stop price
        self._news_cache: Dict[str, tuple] = {}  # symbol -> (timestamp, headlines)
        self._load_state()

    # ------------------------------------------------------------------
    # The endless loop (run in a daemon thread)
    # ------------------------------------------------------------------

    def run_forever(self, stop_event: threading.Event, interval_s: int = 900):
        logger.info("Portfolio Sentinel started (interval %ds)", interval_s)
        while not stop_event.is_set():
            try:
                if self._market_open():
                    self.check_positions()
            except Exception:  # noqa: BLE001
                logger.exception("Sentinel pass failed")
            stop_event.wait(interval_s)

    @staticmethod
    def _market_open() -> bool:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 25)

    # ------------------------------------------------------------------
    # One monitoring pass
    # ------------------------------------------------------------------

    def check_positions(self):
        positions = self.orch.broker.get_positions()
        if not positions:
            return

        cfg = self.orch._exit_config()
        # Layer 2: thesis stops + trailing + partial (pure math)
        for symbol, pos in list(positions.items()):
            thesis = self.orch.desk.thesis_for(symbol)
            partial_taken = bool(thesis.get("partial_taken")) if thesis else False
            self.orch.desk.bump_high_water(symbol, pos.last_price)
            sentinel_stop = self.stop_overrides.get(symbol)
            hit = exit_reason(
                symbol, pos.last_price, pos.average_price, self.orch.desk,
                cfg["stop_pct"], cfg["target_pct"],
                sentinel_stop=sentinel_stop,
                partial_taken=partial_taken,
                partial_take_pct=cfg["partial_take_pct"],
                trail_activate_pct=cfg["trail_activate_pct"],
                trail_pct=cfg["trail_pct"],
            )
            if hit:
                self.orch._process_exit(
                    symbol, pos, hit, self.orch.config.get("partial_take_fraction", 0.5),
                )
                positions.pop(symbol, None)
        if not positions:
            return

        # Layer 3: LLM assessment over all remaining positions (one call)
        position_lines, news_blocks = [], []
        for symbol, pos in positions.items():
            change = (pos.last_price / pos.average_price - 1) * 100 if pos.average_price else 0
            position_lines.append(
                f"- {symbol}: qty {pos.quantity}, entry {pos.average_price:.2f}, "
                f"now {pos.last_price:.2f} ({change:+.1f}%), "
                f"hard stop -{self.orch.config.get('stop_loss_pct', 7)}%, "
                f"target +{self.orch.config.get('take_profit_pct', 14)}%"
                + (f", tightened stop {self.stop_overrides[symbol]:.2f}" if symbol in self.stop_overrides else "")
            )
            thesis = self.orch.desk.thesis_for(symbol)
            if thesis:
                position_lines.append(
                    f"    Original entry thesis ({thesis['opened_at'][:10]}): {thesis['reasoning'][:200]}"
                )
            headlines = self._headlines_for(symbol)
            if headlines:
                news_blocks.append(f"{symbol} news:\n" + "\n".join(f"  - {h}" for h in headlines))

        system = (
            "You are the Sentinel, guardian agent for a positional Indian equity desk (2-8 week holds). "
            "You monitor OWNED positions only. For each position, given its P&L state and fresh news, "
            "decide: HOLD (default — do not churn positions on noise), TIGHTEN_STOP (thesis weakening "
            "or meaningful gain to protect; give new_stop price below current price), or EXIT_NOW (only for "
            "materially negative company-specific news or clear thesis break). Exiting costs ~Rs.120 "
            "and positional trades need room to breathe — be decisive but not jumpy. "
            "NEVER recommend EXIT_NOW on a profitable position (+2% or more) unless there is a "
            "severe company-specific catalyst (fraud, results miss, regulatory action). "
            'Respond with JSON: {"assessments": [{"symbol": "...", "action": "HOLD"|"TIGHTEN_STOP"|"EXIT_NOW", '
            '"new_stop": <float or null>, "confidence": 0-100, "reasoning": "<1-2 sentences>"}]}'
        )
        if self.orch.config.get("strategy_profile") == "aggressive":
            system += (
                " DESK PROFILE: AGGRESSIVE — you may flag stagnation (roughly -1% to +1.5% for 5+ days) "
                "for rotation consideration, but do NOT exit positions above +2% just to churn capital."
            )
        user = (
            f"POSITIONS:\n" + "\n".join(position_lines)
            + ("\n\nFRESH NEWS:\n" + "\n\n".join(news_blocks) if news_blocks else "\n\nNo fresh news found.")
        )
        try:
            result = self.orch.llm.chat_json(system, user, max_tokens=1000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentinel LLM call failed (%s) — math stops still active", exc)
            return

        for item in result.get("assessments", []):
            symbol = (item.get("symbol") or "").upper()
            pos = positions.get(symbol)
            if not pos:
                continue
            action = item.get("action", "HOLD")
            confidence = item.get("confidence", 0)
            self.orch._emit("sentinel", {
                "symbol": symbol, "action": action, "confidence": confidence,
                "reasoning": item.get("reasoning", ""), "pnl_pct": round(
                    (pos.last_price / pos.average_price - 1) * 100 if pos.average_price else 0, 2),
            })
            self.orch.storage.log_agent_message(
                None, symbol, "portfolio_sentinel", "Sentinel",
                f"{action} (conf {confidence}): {item.get('reasoning', '')}", "sentinel",
            )
            # Cross-agent sync: buy-side PM sees this note before any new trade
            self.orch.desk.record_sentinel_note(symbol, action, item.get("reasoning", ""), confidence)
            change = (pos.last_price / pos.average_price - 1) * 100 if pos.average_price else 0
            min_loss = self.orch.config.get("sentinel_min_loss_for_exit_pct", -2.0)
            block_profit = self.orch.config.get("sentinel_block_exit_if_profitable", True)
            if action == "EXIT_NOW" and confidence >= EXIT_CONFIDENCE_THRESHOLD:
                if block_profit and change > 0:
                    logger.info("Sentinel blocked profitable exit on %s (+%.1f%%)", symbol, change)
                    continue
                if change > min_loss and confidence < 85:
                    logger.info(
                        "Sentinel blocked soft exit on %s (+%.1f%%, conf %d) — need 85+",
                        symbol, change, confidence,
                    )
                    continue
                self._exit(symbol, pos, f"sentinel exit: {item.get('reasoning', '')[:120]}")
            elif action == "TIGHTEN_STOP" and item.get("new_stop"):
                new_stop = float(item["new_stop"])
                if 0 < new_stop < pos.last_price:  # sanity: stop must be below market
                    self.stop_overrides[symbol] = new_stop
                    self._save_state()
                    logger.info("Sentinel tightened %s stop to %.2f", symbol, new_stop)

    # ------------------------------------------------------------------

    def _exit(self, symbol: str, pos, reason: str):
        # Shared trade lock: a fill here is atomic w.r.t. the buy-side cycle
        # loop's funds re-validation, so the desk can't double-spend cash.
        avg_before = pos.average_price
        with self.orch.trade_lock:
            current = self.orch.broker.get_positions().get(symbol)
            if not current or current.quantity <= 0:
                logger.info("Sentinel exit skipped — %s already closed", symbol)
                return
            result = self.orch.broker.place_order(
                symbol=symbol, quantity=current.quantity, side="SELL", product="CNC"
            )
        change = (pos.last_price / pos.average_price - 1) * 100 if pos.average_price else 0
        fill_price = result.filled_price or pos.last_price
        exit_pnl = (fill_price - avg_before) * pos.quantity if result.success else None
        self.orch.storage.log_trade(
            None, symbol, "SELL", pos.quantity, fill_price,
            self.orch.broker.name, result.order_id or "",
            "filled" if result.success else "failed", reason, pnl=exit_pnl,
        )
        if result.success:
            self.orch.memory.record_outcome(symbol, change, f"({reason})")
            self.orch.desk.record_exit(symbol, reason, change)
            self.stop_overrides.pop(symbol, None)
            self._save_state()
        self.orch._emit("exit", {"symbol": symbol, "reason": reason, "pnl_pct": round(change, 2),
                                  "pnl": round(exit_pnl, 2) if exit_pnl is not None else None})
        if result.success:
            self.orch.notifier.send(
                f"🛡 Sentinel exit: SELL {pos.quantity} {symbol} ({change:+.1f}%)",
                f"Sold {pos.quantity} {symbol} @ Rs.{fill_price:.2f} — {reason} ({self.orch.broker.name} mode)\n"
                f"Realized P&L: Rs.{exit_pnl:+,.2f}",
            )
        else:
            self.orch.notifier.send(
                f"❌ Sentinel exit FAILED: SELL {pos.quantity} {symbol}",
                f"{reason} but order failed ({self.orch.broker.name} mode): {result.message}",
            )

    def _headlines_for(self, symbol: str, max_age_s: int = 900) -> list:
        cached = self._news_cache.get(symbol)
        if cached and time.time() - cached[0] < max_age_s:
            return cached[1]
        headlines = []
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": f"{symbol} NSE stock", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
                timeout=12, headers={"User-Agent": "Mozilla/5.0"},
            )
            found = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", resp.text)
            headlines = [t.strip() for t in found[1:6] if len(t.strip()) > 15]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentinel news fetch failed for %s: %s", symbol, exc)
        self._news_cache[symbol] = (time.time(), headlines)
        return headlines

    def _save_state(self):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump({"stop_overrides": self.stop_overrides}, f, indent=2)

    def _load_state(self):
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    self.stop_overrides = json.load(f).get("stop_overrides", {})
        except Exception:  # noqa: BLE001
            logger.exception("Could not load sentinel state")
