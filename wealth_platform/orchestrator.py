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
from wealth_platform.pipeline.analyst_panel import AnalystPanelAgent, CompactDecisionAgent
from wealth_platform.pipeline.analyst_utils import (
    compute_ml_signal,
    consensus_vote,
    derive_sentiment_block,
    parse_verdict,
    research_gate_allows,
)
from wealth_platform.brokers import get_broker
from wealth_platform.market.enrichment import enrich
from wealth_platform.trading.coordination import TRADE_LOCK, SharedDesk
from wealth_platform.trading.entry_filters import nifty_trend_ok, stock_entry_ok
from wealth_platform.trading.exit_levels import ExitSignal, exit_reason
from wealth_platform.trading.risk_guard import RiskGuard
from wealth_platform.trading.screener import BuiltInScreener
from wealth_platform.trading.token_budget import TokenBudget
from wealth_platform.pipeline.portfolio_planner import PortfolioPlanner
from wealth_platform.market.discovery import NewsStockDiscovery
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
    "Rs.120/round-trip cost compound against the goal."
)

BALANCED_STYLE = (
    "DESK PROFILE: BALANCED POSITIONAL. Hold quality NIFTY-100 setups for 2-8 weeks. "
    "Target +10-15% with trailing protection after +4%. Require at least 2:1 reward/risk "
    "and net edge after ~Rs.120 round-trip fees. Prefer pullbacks above SMA50 in an uptrending "
    "market; skip extended blow-off tops. Default to HOLD when analysts disagree or confidence "
    "is borderline. Only BUY when the multi-week thesis is clear."
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
        self.risk_guard = RiskGuard(self.config, self.broker)
        self.token_budget = TokenBudget(self.storage, self.config)
        self.screener = BuiltInScreener()
        self.news_discovery = NewsStockDiscovery(self.llm)
        self.trade_lock = TRADE_LOCK
        self.desk = SharedDesk(exit_cooloff_days=self.config.get("exit_cooloff_days", 3))
        self.style_suffix = (
            BALANCED_STYLE if self.config.get("strategy_profile") == "balanced"
            else AGGRESSIVE_STYLE if self.config.get("strategy_profile") == "aggressive"
            else ""
        )
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
    # Math pre-gate (zero LLM)
    # ------------------------------------------------------------------

    def _pre_gate(self, symbol: str) -> Optional[str]:
        if self.desk.in_exit_cooloff(symbol):
            return "exit cool-off active"
        positions = self.broker.get_positions()
        held = positions.get(symbol)
        funds = self.broker.get_funds()
        min_score = self.config.get("screener_min_score", 0)
        score = self.screener.score_for(symbol)
        if score is not None and score < min_score:
            return f"screener score {score} below floor {min_score}"
        if score is None and not held and self.config.get("require_screener_score", True):
            watchlist = set(self.config.get("watchlist", []))
            if symbol not in watchlist:
                return "symbol outside screened universe (news/unlisted pick blocked)"
        max_positions = self.config.get("max_open_positions", 3)
        if not held and len(positions) >= max_positions and funds.available_cash < 1000:
            return f"max positions ({max_positions}) and insufficient cash"
        if not held:
            if self.config.get("require_index_trend", True):
                ok, msg = nifty_trend_ok()
                if not ok:
                    return msg
            ok, msg = stock_entry_ok(symbol, self.config)
            if not ok:
                return msg
        return None

    def _gate_kwargs(self) -> dict:
        return {
            "require_consensus": self.config.get("require_consensus_for_buy", True),
        }

    def _research_gate(self, rating, confidence, held_qty, consensus=None) -> tuple:
        gate = self.history_rag.calibrated_confidence_gate(
            self.config.get("research_confidence_gate", 58),
        )
        return research_gate_allows(
            rating, confidence, gate, held_qty,
            consensus=consensus, **self._gate_kwargs(),
        )

    def _gather_analyst_data(self, symbol: str) -> str:
        """Raw data bundle for compact analyst panel."""
        tech = TechnicalAnalyst(self.llm, None)._gather(symbol)  # noqa: SLF001
        fund = FundamentalsAnalyst(self.llm, None)._gather(symbol)  # noqa: SLF001
        news_raw = NewsAnalyst(self.llm, None)._gather(symbol)  # noqa: SLF001
        ml = compute_ml_signal(symbol)
        ml_line = f"\nQuant signal: {ml}" if ml else ""
        return (
            f"{enrich(symbol)}\n\n=== TECHNICAL DATA ===\n{tech}\n\n"
            f"=== FUNDAMENTALS ===\n{fund}\n\n=== NEWS ===\n{news_raw}{ml_line}"
        )

    def _execute_trade(
        self,
        symbol: str,
        proposal: dict,
        pm_decision: dict,
        research_decision: dict,
        quote_price: float,
    ) -> dict:
        """Shared execution path with price refresh, risk guard, and notifications."""
        if proposal.get("action") == "BUY":
            proposal = self.risk_guard.apply_stop_target(symbol, proposal, quote_price)

        approved, quantity, guard_reason = self.risk_guard.validate_and_resize(
            symbol, proposal, pm_decision, quote_price,
        )
        if not approved:
            self.storage.log_decision(
                self._current_cycle_id, symbol, research_decision.get("rating"),
                proposal, {"decision": "REJECTED", "reasoning": guard_reason}, False,
            )
            return {"symbol": symbol, "action": "rejected", "research": research_decision,
                    "proposal": proposal, "pm": pm_decision, "reason": guard_reason}

        avg_before = 0.0
        if proposal["action"] == "SELL":
            pos = self.broker.get_positions().get(symbol)
            avg_before = pos.average_price if pos else 0.0

        with self.trade_lock:
            fresh = self.broker.get_quote(symbol)
            ok, drift_msg = self.risk_guard.price_drift_ok(
                quote_price, fresh.last_price, self.config.get("max_price_drift_pct", 1.5),
            )
            if not ok:
                self.storage.log_decision(
                    self._current_cycle_id, symbol, research_decision.get("rating"),
                    proposal, {"decision": "ABORTED", "reasoning": drift_msg}, False,
                )
                return {"symbol": symbol, "action": "aborted", "research": research_decision,
                        "proposal": proposal, "pm": pm_decision}

            rotate_out = pm_decision.get("fund_by_selling")
            if rotate_out and proposal["action"] == "BUY":
                rot_ok, rot_msg = self.risk_guard.rotation_allowed(rotate_out)
                if not rot_ok:
                    self.storage.log_decision(
                        self._current_cycle_id, symbol, research_decision.get("rating"),
                        proposal, {"decision": "ABORTED", "reasoning": rot_msg}, False,
                    )
                    return {"symbol": symbol, "action": "aborted", "research": research_decision,
                            "proposal": proposal, "pm": pm_decision, "reason": rot_msg}
                held = self.broker.get_positions().get(rotate_out)
                if held and held.quantity > 0:
                    sell_result = self.broker.place_order(
                        symbol=rotate_out, quantity=held.quantity, side="SELL", product="CNC")
                    change = (held.last_price / held.average_price - 1) * 100 if held.average_price else 0
                    rotate_fill = sell_result.filled_price or held.last_price
                    rotate_pnl = (rotate_fill - held.average_price) * held.quantity if sell_result.success else None
                    self.storage.log_trade(
                        self._current_cycle_id, rotate_out, "SELL", held.quantity,
                        rotate_fill, self.broker.name,
                        sell_result.order_id or "", "filled" if sell_result.success else "failed",
                        f"PM rotation: freeing capital for {symbol}", pnl=rotate_pnl)
                    if sell_result.success:
                        self.memory.record_outcome(rotate_out, change, f"(rotated into {symbol})", rotate_pnl)
                        self.desk.record_exit(rotate_out, f"rotated into {symbol}", change)
                        self._emit("exit", {"symbol": rotate_out,
                                            "reason": f"PM rotation → funding {symbol}",
                                            "pnl_pct": round(change, 2),
                                            "pnl": round(rotate_pnl, 2) if rotate_pnl is not None else None})
                        self.notifier.send(
                            f"🔄 Rotation SELL: {held.quantity} {rotate_out} ({change:+.1f}%)",
                            f"Sold @ Rs.{rotate_fill:.2f} to fund {symbol}\n"
                            f"Realized P&L: Rs.{rotate_pnl:+,.2f}" if rotate_pnl else "",
                        )

            funds_now = self.broker.get_funds()
            if proposal["action"] == "BUY":
                est_cost = quantity * fresh.last_price * 1.005
                if est_cost > funds_now.available_cash:
                    reason = f"stale-funds: need Rs.{est_cost:.0f}, have Rs.{funds_now.available_cash:.0f}"
                    self.storage.log_decision(
                        self._current_cycle_id, symbol, research_decision.get("rating"),
                        proposal, {"decision": "ABORTED", "reasoning": reason}, False,
                    )
                    return {"symbol": symbol, "action": "aborted", "research": research_decision,
                            "proposal": proposal, "pm": pm_decision}

            result = self.broker.place_order(
                symbol=symbol, quantity=quantity, side=proposal["action"], product="CNC",
            )

        fill_price = result.filled_price or fresh.last_price
        trade_pnl = None
        if proposal["action"] == "SELL" and result.success and avg_before:
            trade_pnl = (fill_price - avg_before) * quantity

        self.storage.log_trade(
            self._current_cycle_id, symbol, proposal["action"], quantity,
            fill_price, self.broker.name,
            result.order_id or "", "filled" if result.success else "failed",
            proposal.get("reasoning", ""), pnl=trade_pnl,
        )

        if result.success:
            if proposal["action"] == "BUY":
                self.desk.record_entry(
                    symbol, proposal.get("reasoning", ""),
                    float(proposal.get("stop_loss") or 0),
                    float(proposal.get("take_profit") or 0),
                    fill_price,
                )
            self.memory.record_trade_lifecycle(
                symbol, proposal["action"], research_decision.get("rating", ""),
                proposal.get("reasoning", ""), fill_price, quantity,
                float(proposal.get("stop_loss") or 0),
                float(proposal.get("take_profit") or 0),
                success=True,
            )
            if proposal["action"] == "SELL" and trade_pnl is not None and avg_before:
                pct = (fill_price / avg_before - 1) * 100
                self.memory.record_outcome(symbol, pct, "(trade exit)", trade_pnl)
        else:
            self.memory.record_trade_lifecycle(
                symbol, proposal["action"], research_decision.get("rating", ""),
                proposal.get("reasoning", ""), fill_price, quantity, success=False,
            )

        self._emit("trade_executed", {
            "symbol": symbol, "side": proposal["action"], "quantity": quantity,
            "success": result.success, "message": result.message,
            "pnl": round(trade_pnl, 2) if trade_pnl is not None else None,
        })
        if result.success:
            pnl_line = f"Realized P&L: Rs.{trade_pnl:+,.2f}\n" if trade_pnl is not None else ""
            self.notifier.send(
                f"✅ Trade: {proposal['action']} {quantity} {symbol} @ Rs.{fill_price:.2f}",
                f"Filled @ Rs.{fill_price:.2f} ({self.broker.name} mode)\n{pnl_line}\n"
                f"Reasoning: {proposal.get('reasoning', '')}\n"
                f"Stop: {proposal.get('stop_loss', 'n/a')} | Target: {proposal.get('take_profit', 'n/a')}",
            )
        else:
            self.notifier.send(
                f"❌ Trade FAILED: {proposal['action']} {quantity} {symbol}",
                f"{result.message}\n\nReasoning: {proposal.get('reasoning', '')}",
            )
        return {"symbol": symbol, "action": "executed" if result.success else "failed",
                "research": research_decision, "proposal": proposal, "pm": pm_decision}

    # ------------------------------------------------------------------
    # The full decision cycle for one stock
    # ------------------------------------------------------------------

    def analyze_stock(self, symbol: str) -> dict:
        self._current_symbol = symbol
        cb = self._on_agent_output
        self.token_budget.log_status()
        mode = self.token_budget.mode()

        skip = self._pre_gate(symbol)
        if skip:
            self.storage.log_decision(
                self._current_cycle_id, symbol, "SKIP", {}, {"decision": "SKIPPED", "reasoning": skip}, False,
            )
            self._emit("stage", {"symbol": symbol, "stage": "pre_gate_skip", "reason": skip})
            return {"symbol": symbol, "action": "none", "reason": skip}

        symbol_lessons = self.memory.symbol_lessons(symbol)
        history_context = self.history_rag.build_analyst_context(symbol, symbol_lessons)
        ml_signal = compute_ml_signal(symbol)
        enrichment = enrich(symbol)
        cap = self.config.get("report_context_chars", 1800)

        # ---- COMPACT / MINIMAL: 2 LLM calls (panel + decision) ----
        if mode in (TokenBudget.MODE_COMPACT, TokenBudget.MODE_MINIMAL):
            self._emit("stage", {"symbol": symbol, "stage": "analyst_panel"})
            bundle = self._gather_analyst_data(symbol) + f"\n\n{history_context[:2000]}"
            panel = AnalystPanelAgent(self.llm, cb, self.style_suffix).analyze(symbol, bundle)
            research_decision = {
                "rating": panel.get("rating", "HOLD"),
                "confidence": panel.get("confidence", 0),
                "rationale": panel.get("rationale", ""),
                "key_risks": panel.get("key_risks", []),
            }
            self._emit("research_verdict", {"symbol": symbol, "verdict": research_decision})

            gate = self.history_rag.calibrated_confidence_gate(
                self.config.get("research_confidence_gate", 58),
            )
            positions = {s: p.quantity for s, p in self.broker.get_positions().items()}
            ok, gate_reason = research_gate_allows(
                research_decision["rating"], research_decision["confidence"], gate, positions.get(symbol, 0),
                **self._gate_kwargs(),
            )
            if not ok:
                self.storage.log_decision(
                    self._current_cycle_id, symbol, research_decision.get("rating", "HOLD"),
                    {}, {"decision": "SKIPPED", "reasoning": gate_reason}, False,
                )
                return {"symbol": symbol, "action": "none", "research": research_decision}

            quote = self.broker.get_quote(symbol)
            funds = self.broker.get_funds()
            ctx = (
                f"Stock: {symbol}\nPrice: Rs.{quote.last_price:.2f}\nCash: Rs.{funds.available_cash:.2f}\n"
                f"Positions: {positions}\n\nRESEARCH: {research_decision}\n\nPANEL: {panel}\n\n{history_context[:1500]}"
            )
            self._emit("stage", {"symbol": symbol, "stage": "compact_decision"})
            combined = CompactDecisionAgent(self.llm, cb, self.style_suffix).decide(symbol, ctx)
            proposal = {
                "action": combined.get("action", "HOLD"),
                "quantity": combined.get("quantity", 0),
                "entry_price": combined.get("entry_price", quote.last_price),
                "stop_loss": combined.get("stop_loss"),
                "take_profit": combined.get("take_profit"),
                "reasoning": combined.get("reasoning", ""),
            }
            pm_decision = {
                "decision": combined.get("decision", "REJECT"),
                "adjusted_quantity": combined.get("quantity"),
                "fund_by_selling": combined.get("fund_by_selling"),
                "reasoning": combined.get("reasoning", ""),
            }
            self._emit("trade_proposal", {"symbol": symbol, "proposal": proposal})
            self._emit("pm_decision", {"symbol": symbol, "decision": pm_decision, "approved": pm_decision["decision"] == "APPROVE"})
            if pm_decision["decision"] != "APPROVE" or proposal.get("action") == "HOLD":
                self.storage.log_decision(
                    self._current_cycle_id, symbol, research_decision.get("rating"),
                    proposal, pm_decision, False,
                )
                return {"symbol": symbol, "action": "rejected", "research": research_decision, "proposal": proposal, "pm": pm_decision}
            self.storage.log_decision(
                self._current_cycle_id, symbol, research_decision.get("rating"),
                proposal, pm_decision, True,
            )
            return self._execute_trade(symbol, proposal, pm_decision, research_decision, quote.last_price)

        # ---- STANDARD / FULL pipeline ----
        self._emit("stage", {"symbol": symbol, "stage": "analysts"})
        modules = self.token_budget.analyst_modules()
        technical = fundamentals = news = None
        if "technical" in modules:
            technical = TechnicalAnalyst(self.llm, cb, self.style_suffix).analyze(symbol, ml_signal)
        if "fundamentals" in modules:
            fundamentals = FundamentalsAnalyst(self.llm, cb, self.style_suffix).analyze(symbol)
        if "news" in modules:
            news = NewsAnalyst(self.llm, cb, self.style_suffix).analyze(symbol)

        tech_v = parse_verdict(technical.report if technical else "")
        fund_v = parse_verdict(fundamentals.report if fundamentals else "")
        news_v = parse_verdict(news.report if news else "")
        # Rule-based sentiment (no LLM)
        try:
            import pandas as pd
            import yfinance as yf
            df = yf.download(
                f"{symbol}.NS" if not symbol.endswith(".NS") else symbol,
                period="1mo", progress=False, auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            ch1w = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-6]) - 1) * 100 if len(df) > 6 else 0
            vol_r = float(df["Volume"].iloc[-5:].mean()) / max(float(df["Volume"].rolling(20).mean().iloc[-1]), 1)
        except Exception:  # noqa: BLE001
            ch1w, vol_r = 0, 1
        sentiment_text = derive_sentiment_block([], ch1w, vol_r)
        sent_v = parse_verdict(sentiment_text)

        vote = consensus_vote(tech_v, fund_v, news_v, sent_v)
        analyst_reports = (
            f"{enrichment}\n\n=== TECHNICAL ===\n{(technical.report if technical else '')[:cap]}\n\n"
            f"=== FUNDAMENTALS ===\n{(fundamentals.report if fundamentals else '')[:cap]}\n\n"
            f"=== NEWS ===\n{(news.report if news else '')[:cap]}\n\n"
            f"=== SENTIMENT (rule-based) ===\n{sentiment_text}\n\n"
            f"=== PANEL VOTE ===\n{vote}\n\n{history_context[:1500]}"
        )

        if vote["agreement"] == "split" and self.config.get("require_consensus_for_buy", True):
            research_decision = {
                "rating": "HOLD",
                "confidence": vote["confidence"],
                "rationale": "Analyst panel split — default HOLD until clear consensus.",
                "key_risks": [],
            }
            self._emit("research_verdict", {"symbol": symbol, "verdict": research_decision})
            self.storage.log_decision(
                self._current_cycle_id, symbol, "HOLD",
                {}, {"decision": "SKIPPED", "reasoning": "split analyst consensus"}, False,
            )
            return {"symbol": symbol, "action": "none", "research": research_decision}

        transcript = ""
        if self.token_budget.allow_debate(vote["needs_debate"]):
            self._emit("stage", {"symbol": symbol, "stage": "debate"})
            transcript = run_debate(
                BullResearcher(self.llm, cb, self.style_suffix),
                BearResearcher(self.llm, cb, self.style_suffix),
                analyst_reports, rounds=self.config.get("debate_rounds", 1),
            )
        elif vote["agreement"] in ("strong_bull", "strong_bear", "lean_bull", "lean_bear"):
            research_decision = {
                "rating": vote["rating"],
                "confidence": vote["confidence"],
                "rationale": f"Analyst consensus ({vote['agreement']}) — debate skipped.",
                "key_risks": [],
            }
        else:
            research_decision = None

        if research_decision is None:
            self._emit("stage", {"symbol": symbol, "stage": "research"})
            research_decision = ResearchManager(self.llm, cb, self.style_suffix).decide(
                symbol, analyst_reports, transcript,
            )
        self._emit("research_verdict", {"symbol": symbol, "verdict": research_decision})

        gate = self.history_rag.calibrated_confidence_gate(self.config.get("research_confidence_gate", 58))
        positions = {s: p.quantity for s, p in self.broker.get_positions().items()}
        ok, gate_reason = research_gate_allows(
            research_decision["rating"], research_decision["confidence"], gate, positions.get(symbol, 0),
            consensus=vote.get("agreement"), **self._gate_kwargs(),
        )
        if not ok:
            self.storage.log_decision(
                self._current_cycle_id, symbol, research_decision.get("rating", "HOLD"),
                {}, {"decision": "SKIPPED", "reasoning": gate_reason}, False,
            )
            return {"symbol": symbol, "action": "none", "research": research_decision}

        self._emit("stage", {"symbol": symbol, "stage": "trader"})
        quote = self.broker.get_quote(symbol)
        funds = self.broker.get_funds()
        proposal = TraderAgent(self.llm, cb, self.style_suffix).propose(
            symbol, research_decision,
            analyst_reports[:3000],
            quote.last_price, funds.available_cash, positions,
        )
        self._emit("trade_proposal", {"symbol": symbol, "proposal": proposal})

        if proposal.get("action") == "HOLD" or not proposal.get("quantity"):
            self.storage.log_decision(
                self._current_cycle_id, symbol, research_decision.get("rating"),
                proposal, {"decision": "NO_TRADE", "reasoning": proposal.get("reasoning", "")}, False,
            )
            return {"symbol": symbol, "action": "none", "research": research_decision, "proposal": proposal}

        proposal = self.risk_guard.apply_stop_target(symbol, proposal, quote.last_price)

        if proposal.get("action") == "SELL" and positions.get(symbol, 0) < proposal.get("quantity", 0):
            reason = "SELL without sufficient holding (CNC)"
            self.storage.log_decision(
                self._current_cycle_id, symbol, research_decision.get("rating"),
                proposal, {"decision": "NO_TRADE", "reasoning": reason}, False,
            )
            return {"symbol": symbol, "action": "none", "research": research_decision, "proposal": proposal}

        notional_pct = (proposal.get("quantity", 0) * quote.last_price) / max(self.config.get("capital", 15000), 1)
        borderline = 35 <= research_decision.get("confidence", 0) <= 55
        risk_transcript = "Risk debate skipped — clear conviction or small size."
        if self.token_budget.allow_risk_debate(borderline, notional_pct):
            self._emit("stage", {"symbol": symbol, "stage": "risk_debate"})
            portfolio_context = self.risk_guard.portfolio_context_text(self.desk.pm_briefing())
            risk_transcript = run_risk_debate(
                AggressiveDebator(self.llm, cb, self.style_suffix),
                ConservativeDebator(self.llm, cb, self.style_suffix),
                NeutralDebator(self.llm, cb, self.style_suffix),
                proposal, portfolio_context,
            )

        self._emit("stage", {"symbol": symbol, "stage": "portfolio_manager"})
        pm_decision = PortfolioManagerAgent(self.llm, cb, self.style_suffix).decide(
            symbol, research_decision, proposal, risk_transcript,
            self.risk_guard.portfolio_context_text(self.desk.pm_briefing()),
            self.memory.lessons() + "\n\n" + history_context,
        )
        approved = pm_decision.get("decision") == "APPROVE"
        self._emit("pm_decision", {"symbol": symbol, "decision": pm_decision, "approved": approved})
        self.storage.log_decision(
            self._current_cycle_id, symbol, research_decision.get("rating"),
            proposal, pm_decision, approved,
        )
        if not approved:
            return {"symbol": symbol, "action": "rejected", "research": research_decision,
                    "proposal": proposal, "pm": pm_decision}

        return self._execute_trade(symbol, proposal, pm_decision, research_decision, quote.last_price)

    # ------------------------------------------------------------------
    # Exit management (math only — zero LLM cost, callable every 5 min)
    # ------------------------------------------------------------------

    def _exit_config(self) -> dict:
        trail = self.config.get("trailing_stop", {})
        return {
            "stop_pct": self.config.get("stop_loss_pct", 6.0),
            "target_pct": self.config.get("take_profit_pct", 12.0),
            "partial_take_pct": self.config.get("partial_take_profit_pct", 5.0),
            "trail_activate_pct": trail.get("activate_pct", 4.0),
            "trail_pct": trail.get("trail_pct", 2.5),
        }

    def _process_exit(self, symbol: str, pos, signal: ExitSignal, partial_fraction: float = 0.5):
        sell_qty = pos.quantity
        if signal.source == "partial":
            sell_qty = max(1, int(pos.quantity * partial_fraction))
            if sell_qty >= pos.quantity and pos.quantity > 1:
                sell_qty = pos.quantity // 2 or 1

        change = (pos.last_price / pos.average_price - 1) * 100 if pos.average_price else 0
        avg_before = pos.average_price
        with self.trade_lock:
            result = self.broker.place_order(symbol=symbol, quantity=sell_qty, side="SELL", product="CNC")
        fill_price = result.filled_price or pos.last_price
        exit_pnl = (fill_price - avg_before) * sell_qty if result.success else None
        self.storage.log_trade(
            None, symbol, "SELL", sell_qty, fill_price,
            self.broker.name, result.order_id or "",
            "filled" if result.success else "failed", signal.reason,
            pnl=exit_pnl,
        )
        if result.success:
            self.memory.record_outcome(symbol, change, f"({signal.reason})", exit_pnl)
            if signal.source == "partial":
                self.desk.mark_partial_taken(symbol)
            else:
                self.desk.record_exit(symbol, signal.reason, change)
        self._emit("exit", {"symbol": symbol, "reason": signal.reason, "pnl_pct": round(change, 2),
                            "pnl": round(exit_pnl, 2) if exit_pnl is not None else None,
                            "quantity": sell_qty})
        if result.success:
            self.notifier.send(
                f"🚪 Exit: SELL {sell_qty} {symbol} ({change:+.1f}%)",
                f"Sold {sell_qty} {symbol} @ Rs.{fill_price:.2f} — {signal.reason} ({self.broker.name} mode)\n"
                f"Realized P&L: Rs.{exit_pnl:+,.2f}",
            )
        else:
            self.notifier.send(
                f"❌ Exit FAILED: SELL {sell_qty} {symbol}",
                f"{signal.reason} but order failed ({self.broker.name} mode): {result.message}",
            )

    def manage_exits(self):
        cfg = self._exit_config()
        partial_fraction = self.config.get("partial_take_fraction", 0.5)
        for symbol, pos in list(self.broker.get_positions().items()):
            thesis = self.desk.thesis_for(symbol)
            partial_taken = bool(thesis.get("partial_taken")) if thesis else False
            self.desk.bump_high_water(symbol, pos.last_price)
            hit = exit_reason(
                symbol, pos.last_price, pos.average_price, self.desk,
                cfg["stop_pct"], cfg["target_pct"],
                partial_taken=partial_taken,
                partial_take_pct=cfg["partial_take_pct"],
                trail_activate_pct=cfg["trail_activate_pct"],
                trail_pct=cfg["trail_pct"],
            )
            if not hit:
                continue
            self._process_exit(symbol, pos, hit, partial_fraction)

    # ------------------------------------------------------------------
    # Full daily cycle + EOD report
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bot lifecycle notifications (scheduled once/day — not per cycle)
    # ------------------------------------------------------------------

    def _portfolio_summary(self) -> str:
        try:
            funds = self.broker.get_funds()
            positions = self.broker.get_positions()
            lines = [f"Broker: {self.broker.name}", f"Cash: Rs.{funds.available_cash:,.2f}"]
            if positions:
                lines.append("Open positions:")
                for s, p in positions.items():
                    lines.append(
                        f"  {s}: {p.quantity} @ Rs.{p.average_price:.2f} "
                        f"(LTP Rs.{p.last_price:.2f}, P&L Rs.{p.pnl:+,.2f})"
                    )
            else:
                lines.append("Open positions: none")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"Portfolio snapshot unavailable: {exc}"

    def notify_bot_start(self):
        self._emit("bot_start", {})
        self.notifier.send(
            "🤖 Bot Started — trading day beginning",
            f"The wealth platform bot is starting its trading day "
            f"({datetime.now().strftime('%d %b %Y, %H:%M')} IST).\n\n{self._portfolio_summary()}",
        )

    def notify_bot_stop(self):
        self._emit("bot_stop", {})
        self.notifier.send(
            "🛑 Bot Stopped — trading day over",
            f"Market closed; the bot has finished trading for today "
            f"({datetime.now().strftime('%d %b %Y, %H:%M')} IST).\n\n{self._portfolio_summary()}",
        )

    def run_daily_cycle(self) -> dict:
        self.risk_guard.broker = self.broker  # refresh after reconnect
        candidates = self.select_symbols()
        if self.config.get("portfolio_planner", True) and len(candidates) > self.config.get("max_stocks_per_cycle", 3):
            planner = PortfolioPlanner(self.llm, self._on_agent_output, self.style_suffix)
            candidates = planner.plan(
                candidates,
                self.risk_guard.portfolio_context_text(),
                self.desk.pm_briefing(),
                self.memory.lessons(),
                max_n=self.config.get("max_stocks_per_cycle", 3),
            )
        symbols = candidates
        self._current_cycle_id = self.storage.start_cycle(symbols)
        self._emit("cycle_started", {"cycle_id": self._current_cycle_id, "symbols": symbols, "pipeline_mode": self.token_budget.mode()})

        results = []
        for symbol in symbols:
            try:
                results.append(self.analyze_stock(symbol))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Cycle failed for %s", symbol)
                results.append({"symbol": symbol, "action": "error", "error": str(exc)})

        ipo_analyses = []
        if self.token_budget.allow_ipo_scan():
            try:
                ipo_analyses = self.ipo_manager.daily_ipo_scan(self.broker.get_funds().available_cash)
                for analysis in ipo_analyses:
                    self._emit("ipo_analysis", analysis)
            except Exception:  # noqa: BLE001
                logger.exception("IPO scan failed")

        self._snapshot()
        report = self._build_cycle_report(results, ipo_analyses)
        self.storage.finish_cycle(self._current_cycle_id, report)
        self._emit("cycle_report", {"cycle_id": self._current_cycle_id, "report": report})
        self._emit("cycle_finished", {"cycle_id": self._current_cycle_id})
        return {"cycle_id": self._current_cycle_id, "results": results, "ipos": ipo_analyses}

    def _build_cycle_report(self, results: List[dict], ipo_analyses: List[dict]) -> str:
        """Human-readable per-cycle summary — assembled deterministically (no LLM cost)."""
        lines = []
        executed = [r for r in results if r.get("action") == "executed"]
        for r in results:
            symbol = r.get("symbol", "?")
            research = r.get("research", {}) or {}
            proposal = r.get("proposal", {}) or {}
            pm = r.get("pm", {}) or {}
            rating = research.get("rating", "-")
            conf = research.get("confidence", "-")
            action = r.get("action", "?")
            outcome = {
                "executed": f"**TRADED** — {proposal.get('action', '')} {proposal.get('quantity', '')} filled",
                "rejected": f"PM rejected — {pm.get('reasoning', '')[:140]}",
                "aborted": "aborted at execution (stale funds guard)",
                "failed": "order failed at broker",
                "none": "no trade — " + (proposal.get("reasoning") or research.get("rationale") or "below conviction gate")[:140],
                "error": f"errored: {r.get('error', '')[:100]}",
            }.get(action, action)
            lines.append(f"- **{symbol}**: research {rating} ({conf}%) → {outcome}")
        funds = self.broker.get_funds()
        positions = self.broker.get_positions()
        lines.append("")
        lines.append(f"**Outcome:** {len(executed)} trade(s) executed, "
                     f"{len(results) - len(executed)} passed. "
                     f"Cash now Rs.{funds.available_cash:,.0f}, open positions: "
                     f"{', '.join(positions.keys()) or 'none'}.")
        if ipo_analyses:
            for a in ipo_analyses:
                lines.append(f"- IPO {a.get('ipo', {}).get('company', '?')}: "
                             f"{a.get('recommendation', '?')} ({a.get('confidence', 0)}%)")
        return "\n".join(lines)

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
