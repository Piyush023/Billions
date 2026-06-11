"""The daily decision cycle: screener -> analysts -> debate -> trader -> risk
debate -> portfolio manager -> execution -> logging -> EOD report.

Free-tier discipline baked in:
  - One full cycle per day (or manual trigger), top-N stocks only
  - All agents on Groq/Gemini free tier; PM optionally on Claude Haiku
  - Intraday monitoring uses pure math (stop-loss/target) — zero LLM calls
"""

import json
import logging
import os
from datetime import datetime
from typing import Callable, List, Optional

from wealth_platform.agents.agent_memory import AgentMemory
from wealth_platform.agents.analysts import (
    FundamentalsAnalyst,
    NewsAnalyst,
    SentimentAnalyst,
    TechnicalAnalyst,
)
from wealth_platform.agents.base_agent import AgentOutput
from wealth_platform.agents.portfolio_manager_agent import PortfolioManagerAgent
from wealth_platform.agents.researchers import BearResearcher, BullResearcher, ResearchManager, run_debate
from wealth_platform.agents.risk_debators import (
    AggressiveDebator,
    ConservativeDebator,
    NeutralDebator,
    run_risk_debate,
)
from wealth_platform.agents.trade_history_rag import TradeHistoryRAG
from wealth_platform.agents.trader_agent import TraderAgent
from wealth_platform.brokers import get_broker
from wealth_platform.investments.ipo_manager import IPOManager
from wealth_platform.investments.mutual_funds import MutualFundManager
from wealth_platform.llm.llm_client import LLMClient
from wealth_platform.notifications import Notifier
from wealth_platform.storage import Storage

logger = logging.getLogger("wealth_platform.orchestrator")

CONFIG_PATH = "wealth_config.json"

AGGRESSIVE_STYLE = (
    "DESK PROFILE: AGGRESSIVE GROWTH. This desk's mandate is maximum capital velocity: "
    "prefer decisive action over caution, prioritize high-momentum setups with near-term "
    "catalysts, accept elevated volatility, and favor quick 4-8% swing gains with tight "
    "stops over slow positional grinds. Idle cash is a cost — capital should always be "
    "working in the best available opportunity. You may still reject genuinely bad setups, "
    "but when evidence is mixed, lean toward action with controlled size rather than HOLD. "
    "Realism constraint: target setups with honest 3-10% upside over days to weeks; do NOT "
    "fabricate conviction or inflate confidence numbers to force trades — bad trades at "
    "Rs.60/round-trip cost compound against the goal."
)


class WealthOrchestrator:
    def __init__(self, config_path: str = CONFIG_PATH, on_event: Optional[Callable[[dict], None]] = None):
        self.config = self._load_config(config_path)
        self.on_event = on_event  # streams events to dashboard websocket
        self.llm = LLMClient()
        self.storage = Storage()
        self.memory = AgentMemory()
        self.broker = get_broker(
            self.config.get("broker", "paper"),
            starting_cash=self.config.get("capital", 15000),
        )
        self.mf_manager = MutualFundManager(self.llm)
        self.ipo_manager = IPOManager(self.llm)
        self.notifier = Notifier()
        self.history_rag = TradeHistoryRAG(self.storage)
        from wealth_platform.coordination import TRADE_LOCK, SharedDesk
        from wealth_platform.news_discovery import NewsStockDiscovery
        from wealth_platform.screener import BuiltInScreener

        self.screener = BuiltInScreener()
        self.news_discovery = NewsStockDiscovery(self.llm)
        self.trade_lock = TRADE_LOCK
        self.desk = SharedDesk(exit_cooloff_days=self.config.get("exit_cooloff_days", 3))
        self.style_suffix = AGGRESSIVE_STYLE if self.config.get("strategy_profile") == "aggressive" else ""
        self._analyzed_on: Optional[str] = None  # date string
        self._analyzed_today: set = set()
        self._current_cycle_id: Optional[int] = None
        self._current_symbol: str = ""

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {"broker": "paper", "capital": 15000, "max_stocks_per_cycle": 3, "debate_rounds": 1}

    # ------------------------------------------------------------------
    # Event streaming
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict):
        event = {"type": event_type, "time": datetime.now().isoformat(), **payload}
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001
                logger.exception("Event emit failed")

    def _on_agent_output(self, output: AgentOutput):
        self.storage.log_agent_message(
            self._current_cycle_id, self._current_symbol, output.agent_name,
            output.role, output.report, output.provider,
        )
        self._emit("agent_report", {
            "symbol": self._current_symbol,
            "agent": output.agent_name,
            "role": output.role,
            "report": output.report,
            "provider": output.provider,
        })

    # ------------------------------------------------------------------
    # Stock selection
    # ------------------------------------------------------------------

    def select_symbols(self) -> List[str]:
        """Pick the next batch from the screener ranking, rotating through the
        day: stocks already analyzed today are skipped so consecutive cycles
        cover different parts of the market instead of repeating the same 3."""
        from datetime import date as _date

        today = str(_date.today())
        if self._analyzed_on != today:
            self._analyzed_on = today
            self._analyzed_today = set()

        n = self.config.get("max_stocks_per_cycle", 3)

        # News-driven discovery: stocks making headlines (including outside
        # the NIFTY-100 universe) get priority slots in the batch.
        batch: List[str] = []
        if self.config.get("news_discovery", True):
            try:
                news_slots = self.config.get("news_slots", 1)
                for item in self.news_discovery.discover():
                    if item["symbol"] not in self._analyzed_today and len(batch) < news_slots:
                        batch.append(item["symbol"])
                        self._emit("news_pick", {"symbol": item["symbol"], "reason": item.get("reason", "")})
            except Exception as exc:  # noqa: BLE001
                logger.warning("News discovery failed (%s)", exc)

        ranked: List[str] = []
        try:
            ranked = self.screener.ranked_symbols(top_n=40)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Built-in screener failed (%s)", exc)
        if not ranked:
            ranked = self.config.get("watchlist", ["RELIANCE", "HDFCBANK", "TCS"])

        # Exclude symbols the desk exited recently (cool-off) so the buy side
        # can't re-enter what the sentinel just sold.
        ranked = [s for s in ranked if not self.desk.in_exit_cooloff(s)]
        batch = [s for s in batch if not self.desk.in_exit_cooloff(s)]
        fresh = [s for s in ranked if s not in self._analyzed_today and s not in batch]
        if not fresh and not batch:  # whole ranking covered today — start over
            self._analyzed_today = set()
            fresh = ranked
        batch.extend(fresh[: n - len(batch)])
        self._analyzed_today.update(batch)
        logger.info("Cycle batch: %s (analyzed today: %d)", batch, len(self._analyzed_today))
        return batch

    # ------------------------------------------------------------------
    # The full decision cycle for one stock
    # ------------------------------------------------------------------

    def analyze_stock(self, symbol: str) -> dict:
        self._current_symbol = symbol
        cb = self._on_agent_output
        self._emit("stage", {"symbol": symbol, "stage": "analysts"})

        technical = TechnicalAnalyst(self.llm, cb, self.style_suffix).analyze(symbol)
        fundamentals = FundamentalsAnalyst(self.llm, cb, self.style_suffix).analyze(symbol)
        news = NewsAnalyst(self.llm, cb, self.style_suffix).analyze(symbol)
        sentiment = SentimentAnalyst(self.llm, cb, self.style_suffix).analyze(symbol, news.report, technical.report)

        # Cap each report in the combined context: it gets re-sent to the
        # debate (twice per round), research manager, trader, and PM — on
        # free tiers with daily TOKEN caps, uncapped reports burn the whole
        # day's budget in one or two cycles.
        cap = self.config.get("report_context_chars", 1800)
        analyst_reports = (
            f"=== TECHNICAL ===\n{technical.report[:cap]}\n\n=== FUNDAMENTALS ===\n{fundamentals.report[:cap]}\n\n"
            f"=== NEWS ===\n{news.report[:cap]}\n\n=== SENTIMENT ===\n{sentiment.report[:cap]}"
        )

        self._emit("stage", {"symbol": symbol, "stage": "debate"})
        transcript = run_debate(
            BullResearcher(self.llm, cb, self.style_suffix), BearResearcher(self.llm, cb, self.style_suffix),
            analyst_reports, rounds=self.config.get("debate_rounds", 1),
        )
        research_decision = ResearchManager(self.llm, cb, self.style_suffix).decide(symbol, analyst_reports, transcript)
        self._emit("research_verdict", {"symbol": symbol, "verdict": research_decision})

        gate = self.config.get("research_confidence_gate", 40)
        if research_decision.get("rating") in ("HOLD",) or research_decision.get("confidence", 0) < gate:
            self.storage.log_decision(self._current_cycle_id, symbol, research_decision.get("rating", "HOLD"),
                                      {}, {"decision": "SKIPPED", "reasoning": "Research verdict below action threshold"}, False)
            return {"symbol": symbol, "action": "none", "research": research_decision}

        self._emit("stage", {"symbol": symbol, "stage": "trader"})
        quote = self.broker.get_quote(symbol)
        funds = self.broker.get_funds()
        positions = {s: p.quantity for s, p in self.broker.get_positions().items()}

        history_context = self.history_rag.build_context(symbol)
        proposal = TraderAgent(self.llm, cb, self.style_suffix).propose(
            symbol, research_decision,
            analyst_reports[:3000] + "\n\n" + history_context,
            quote.last_price, funds.available_cash, positions,
        )
        self._emit("trade_proposal", {"symbol": symbol, "proposal": proposal})

        if proposal.get("action") == "HOLD" or not proposal.get("quantity"):
            self.storage.log_decision(self._current_cycle_id, symbol, research_decision.get("rating"),
                                      proposal, {"decision": "NO_TRADE", "reasoning": proposal.get("reasoning", "")}, False)
            return {"symbol": symbol, "action": "none", "research": research_decision, "proposal": proposal}

        # Hard guard: CNC accounts cannot short. A SELL on a stock we don't
        # hold is a no-trade regardless of what the agents concluded.
        if proposal.get("action") == "SELL" and positions.get(symbol, 0) < proposal.get("quantity", 0):
            reason = "SELL proposed without sufficient holding (no short selling in CNC) — converted to no-trade"
            self.storage.log_decision(self._current_cycle_id, symbol, research_decision.get("rating"),
                                      proposal, {"decision": "NO_TRADE", "reasoning": reason}, False)
            self._emit("trade_proposal", {"symbol": symbol, "proposal": {**proposal, "action": "HOLD", "reasoning": reason}})
            return {"symbol": symbol, "action": "none", "research": research_decision, "proposal": proposal}

        self._emit("stage", {"symbol": symbol, "stage": "risk_debate"})
        portfolio_context = (
            f"Total capital: Rs.{self.config.get('capital', 15000)}\n"
            f"Available cash: Rs.{funds.available_cash:.2f}\n"
            f"Open positions: {positions or 'none'}\n"
            f"Broker: {self.broker.name}"
        )
        risk_transcript = run_risk_debate(
            AggressiveDebator(self.llm, cb, self.style_suffix), ConservativeDebator(self.llm, cb, self.style_suffix),
            NeutralDebator(self.llm, cb, self.style_suffix), proposal, portfolio_context,
        )

        self._emit("stage", {"symbol": symbol, "stage": "portfolio_manager"})
        pm_decision = PortfolioManagerAgent(self.llm, cb, self.style_suffix).decide(
            symbol, research_decision, proposal, risk_transcript,
            portfolio_context + "\n\nDESK BRIEFING (from the sentinel agent watching holdings):\n"
            + self.desk.pm_briefing(),
            self.memory.lessons() + "\n\n" + history_context,
        )
        approved = pm_decision.get("decision") == "APPROVE"
        self._emit("pm_decision", {"symbol": symbol, "decision": pm_decision, "approved": approved})
        self.storage.log_decision(self._current_cycle_id, symbol, research_decision.get("rating"),
                                  proposal, pm_decision, approved)
        self.memory.record(symbol, pm_decision.get("decision", "REJECT"),
                           research_decision.get("rating", ""), pm_decision.get("reasoning", ""))

        if not approved:
            return {"symbol": symbol, "action": "rejected", "research": research_decision,
                    "proposal": proposal, "pm": pm_decision}

        quantity = int(pm_decision.get("adjusted_quantity") or proposal["quantity"])
        # Critical section: the sentinel may have traded while the agents were
        # debating (minutes). Re-validate funds/positions and fill atomically
        # under the shared trade lock so the two loops can't double-spend.
        with self.trade_lock:
            # PM-approved capital rotation: sell a weaker holding to fund this buy.
            rotate_out = pm_decision.get("fund_by_selling")
            if rotate_out and proposal["action"] == "BUY":
                held = self.broker.get_positions().get(rotate_out)
                if held and held.quantity > 0:
                    sell_result = self.broker.place_order(
                        symbol=rotate_out, quantity=held.quantity, side="SELL", product="CNC")
                    change = (held.last_price / held.average_price - 1) * 100 if held.average_price else 0
                    self.storage.log_trade(
                        self._current_cycle_id, rotate_out, "SELL", held.quantity,
                        sell_result.filled_price or held.last_price, self.broker.name,
                        sell_result.order_id or "", "filled" if sell_result.success else "failed",
                        f"PM rotation: freeing capital for {symbol}")
                    if sell_result.success:
                        self.memory.record_outcome(rotate_out, change, f"(rotated into {symbol})")
                        self.desk.record_exit(rotate_out, f"rotated into {symbol}", change)
                        self._emit("exit", {"symbol": rotate_out,
                                            "reason": f"PM rotation → funding {symbol}",
                                            "pnl_pct": round(change, 2)})
            funds_now = self.broker.get_funds()
            if proposal["action"] == "BUY":
                est_cost = quantity * quote.last_price * 1.005  # + costs headroom
                if est_cost > funds_now.available_cash:
                    reason = (f"stale-funds guard: needs Rs.{est_cost:.0f} but only "
                              f"Rs.{funds_now.available_cash:.0f} available now (sentinel may have traded)")
                    self.storage.log_decision(self._current_cycle_id, symbol, research_decision.get("rating"),
                                              proposal, {"decision": "ABORTED", "reasoning": reason}, False)
                    self._emit("trade_executed", {"symbol": symbol, "side": proposal["action"],
                                                  "quantity": quantity, "success": False, "message": reason})
                    return {"symbol": symbol, "action": "aborted", "research": research_decision,
                            "proposal": proposal, "pm": pm_decision}
            result = self.broker.place_order(symbol=symbol, quantity=quantity, side=proposal["action"], product="CNC")
        self.storage.log_trade(
            self._current_cycle_id, symbol, proposal["action"], quantity,
            result.filled_price or quote.last_price, self.broker.name,
            result.order_id or "", "filled" if result.success else "failed",
            proposal.get("reasoning", ""),
        )
        if result.success and proposal["action"] == "BUY":
            self.desk.record_entry(symbol, proposal.get("reasoning", ""),
                                   proposal.get("stop_loss", 0), proposal.get("take_profit", 0))
        self._emit("trade_executed", {
            "symbol": symbol, "side": proposal["action"], "quantity": quantity,
            "success": result.success, "message": result.message,
        })
        if result.success:
            self.notifier.send(
                f"✅ Trade: {proposal['action']} {quantity} {symbol}",
                f"Filled @ Rs.{result.filled_price or quote.last_price:.2f} ({self.broker.name} mode)\n\n"
                f"Reasoning: {proposal.get('reasoning', '')}\n"
                f"Stop-loss: {proposal.get('stop_loss', 'n/a')} | Target: {proposal.get('take_profit', 'n/a')}",
            )
        return {"symbol": symbol, "action": "executed" if result.success else "failed",
                "research": research_decision, "proposal": proposal, "pm": pm_decision}

    # ------------------------------------------------------------------
    # Exit management (math only — zero LLM cost, callable every 5 min)
    # ------------------------------------------------------------------

    def manage_exits(self):
        stop_pct = self.config.get("stop_loss_pct", 7.0)
        target_pct = self.config.get("take_profit_pct", 14.0)
        for symbol, pos in list(self.broker.get_positions().items()):
            change = (pos.last_price / pos.average_price - 1) * 100 if pos.average_price else 0
            reason = None
            if change <= -stop_pct:
                reason = f"stop-loss hit ({change:.1f}%)"
            elif change >= target_pct:
                reason = f"target hit ({change:.1f}%)"
            if reason:
                with self.trade_lock:
                    result = self.broker.place_order(symbol=symbol, quantity=pos.quantity, side="SELL", product="CNC")
                self.storage.log_trade(None, symbol, "SELL", pos.quantity, result.filled_price or pos.last_price,
                                       self.broker.name, result.order_id or "",
                                       "filled" if result.success else "failed", reason)
                self.memory.record_outcome(symbol, change, f"({reason})")
                if result.success:
                    self.desk.record_exit(symbol, reason, change)
                self._emit("exit", {"symbol": symbol, "reason": reason, "pnl_pct": round(change, 2)})
                self.notifier.send(
                    f"🚪 Exit: {symbol} ({change:+.1f}%)",
                    f"Sold {pos.quantity} {symbol} — {reason} ({self.broker.name} mode)",
                )

    # ------------------------------------------------------------------
    # Full daily cycle + EOD report
    # ------------------------------------------------------------------

    def run_daily_cycle(self) -> dict:
        symbols = self.select_symbols()
        self._current_cycle_id = self.storage.start_cycle(symbols)
        self._emit("cycle_started", {"cycle_id": self._current_cycle_id, "symbols": symbols})

        results = []
        for symbol in symbols:
            try:
                results.append(self.analyze_stock(symbol))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Cycle failed for %s", symbol)
                results.append({"symbol": symbol, "action": "error", "error": str(exc)})

        ipo_analyses = []
        if self.config.get("ipo_enabled", True):
            try:
                ipo_analyses = self.ipo_manager.daily_ipo_scan(self.broker.get_funds().available_cash)
                for analysis in ipo_analyses:
                    self._emit("ipo_analysis", analysis)
            except Exception:  # noqa: BLE001
                logger.exception("IPO scan failed")

        self._snapshot()
        summary = json.dumps({"stocks": [{"symbol": r["symbol"], "action": r["action"]} for r in results],
                              "ipos_analyzed": len(ipo_analyses)})
        self.storage.finish_cycle(self._current_cycle_id, summary)
        self._emit("cycle_finished", {"cycle_id": self._current_cycle_id, "summary": summary})
        return {"cycle_id": self._current_cycle_id, "results": results, "ipos": ipo_analyses}

    def _snapshot(self):
        funds = self.broker.get_funds()
        positions = {s: {"qty": p.quantity, "avg": p.average_price, "ltp": p.last_price, "pnl": round(p.pnl, 2)}
                     for s, p in self.broker.get_positions().items()}
        equity_value = funds.available_cash + sum(p["ltp"] * p["qty"] for p in positions.values())
        mf_value = 0.0
        try:
            mf_value = self.mf_manager.portfolio_snapshot()["total_value"]
        except Exception:  # noqa: BLE001
            pass
        self.storage.snapshot_portfolio(equity_value + mf_value, funds.available_cash, positions, mf_value)

    def generate_eod_report(self) -> str:
        """One LLM call summarising the day. Sent to Telegram + saved to DB."""
        trades = self.storage.recent_trades(20)
        decisions = self.storage.recent_decisions(20)
        history = self.storage.portfolio_history(2)
        today = str(datetime.now().date())
        todays_trades = [t for t in trades if t["created_at"].startswith(today)]
        todays_decisions = [d for d in decisions if d["created_at"].startswith(today)]

        system = (
            "You are a portfolio reporting assistant. Write a concise end-of-day report for a retail "
            "investor in plain language: portfolio value and day change, trades executed and why, "
            "decisions skipped/rejected and why, and anything to watch tomorrow. Keep it under 300 words."
        )
        user = (
            f"Date: {today}\nPortfolio snapshots (latest first): {history}\n\n"
            f"Today's trades: {todays_trades or 'none'}\n\nToday's decisions: {todays_decisions or 'none'}"
        )
        try:
            report = self.llm.chat(system, user, max_tokens=800).text
        except Exception as exc:  # noqa: BLE001
            report = f"EOD report generation failed: {exc}"
        self.storage.save_eod_report(report)
        self._emit("eod_report", {"report": report})
        self.notifier.send(f"📊 EOD Report — {today}", report)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    orchestrator = WealthOrchestrator()
    outcome = orchestrator.run_daily_cycle()
    print(json.dumps(outcome, indent=2, default=str))
    print("\n" + orchestrator.generate_eod_report())
