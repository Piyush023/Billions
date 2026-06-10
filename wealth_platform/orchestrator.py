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
        """Use the existing dynamic screener when available; fall back to config list."""
        try:
            try:
                from legacy_bot.dynamic_stock_screener import DynamicStockScreener
            except ImportError:
                from dynamic_stock_screener import DynamicStockScreener

            screener = DynamicStockScreener()
            results = screener.quick_screen()
            if results:
                top = [r["symbol"].replace(".NS", "") for r in results[: self.config.get("max_stocks_per_cycle", 3)]]
                logger.info("Screener selected: %s", top)
                return top
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screener unavailable (%s); using config watchlist", exc)
        return self.config.get("watchlist", ["RELIANCE", "HDFCBANK", "TCS"])[: self.config.get("max_stocks_per_cycle", 3)]

    # ------------------------------------------------------------------
    # The full decision cycle for one stock
    # ------------------------------------------------------------------

    def analyze_stock(self, symbol: str) -> dict:
        self._current_symbol = symbol
        cb = self._on_agent_output
        self._emit("stage", {"symbol": symbol, "stage": "analysts"})

        technical = TechnicalAnalyst(self.llm, cb).analyze(symbol)
        fundamentals = FundamentalsAnalyst(self.llm, cb).analyze(symbol)
        news = NewsAnalyst(self.llm, cb).analyze(symbol)
        sentiment = SentimentAnalyst(self.llm, cb).analyze(symbol, news.report, technical.report)

        analyst_reports = (
            f"=== TECHNICAL ===\n{technical.report}\n\n=== FUNDAMENTALS ===\n{fundamentals.report}\n\n"
            f"=== NEWS ===\n{news.report}\n\n=== SENTIMENT ===\n{sentiment.report}"
        )

        self._emit("stage", {"symbol": symbol, "stage": "debate"})
        transcript = run_debate(
            BullResearcher(self.llm, cb), BearResearcher(self.llm, cb),
            analyst_reports, rounds=self.config.get("debate_rounds", 1),
        )
        research_decision = ResearchManager(self.llm, cb).decide(symbol, analyst_reports, transcript)
        self._emit("research_verdict", {"symbol": symbol, "verdict": research_decision})

        if research_decision.get("rating") in ("HOLD",) or research_decision.get("confidence", 0) < 40:
            self.storage.log_decision(self._current_cycle_id, symbol, research_decision.get("rating", "HOLD"),
                                      {}, {"decision": "SKIPPED", "reasoning": "Research verdict below action threshold"}, False)
            return {"symbol": symbol, "action": "none", "research": research_decision}

        self._emit("stage", {"symbol": symbol, "stage": "trader"})
        quote = self.broker.get_quote(symbol)
        funds = self.broker.get_funds()
        positions = {s: p.quantity for s, p in self.broker.get_positions().items()}

        history_context = self.history_rag.build_context(symbol)
        proposal = TraderAgent(self.llm, cb).propose(
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
            AggressiveDebator(self.llm, cb), ConservativeDebator(self.llm, cb),
            NeutralDebator(self.llm, cb), proposal, portfolio_context,
        )

        self._emit("stage", {"symbol": symbol, "stage": "portfolio_manager"})
        pm_decision = PortfolioManagerAgent(self.llm, cb).decide(
            symbol, research_decision, proposal, risk_transcript, portfolio_context,
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
        result = self.broker.place_order(symbol=symbol, quantity=quantity, side=proposal["action"], product="CNC")
        self.storage.log_trade(
            self._current_cycle_id, symbol, proposal["action"], quantity,
            result.filled_price or quote.last_price, self.broker.name,
            result.order_id or "", "filled" if result.success else "failed",
            proposal.get("reasoning", ""),
        )
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
                result = self.broker.place_order(symbol=symbol, quantity=pos.quantity, side="SELL", product="CNC")
                self.storage.log_trade(None, symbol, "SELL", pos.quantity, result.filled_price or pos.last_price,
                                       self.broker.name, result.order_id or "",
                                       "filled" if result.success else "failed", reason)
                self.memory.record_outcome(symbol, change, f"({reason})")
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
