"""FastAPI platform server: REST API + WebSocket live feed + dashboard.

Run:  uvicorn wealth_platform.server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000

Free hosting: deploy this on Oracle Cloud Free Tier (or run locally);
the dashboard is a single static HTML file served by this same process.

Scheduler (IST):
  08:45  Mon-Fri  bot start email + pre-market cycle
  09:15-14:45  continuous back-to-back cycles during market hours
  every 5 min during market hours: math-only exit management (no LLM)
  15:30  Mon-Fri  bot stop email with portfolio summary
  15:35  Mon-Fri  EOD report -> DB + email/Telegram
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from wealth_platform.orchestrator import WealthOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("wealth_platform.server")

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


class EventBus:
    """Fans orchestrator events out to all connected websocket clients."""

    def __init__(self):
        self.clients: List[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop = None
        self.recent_events: List[dict] = []

    def publish(self, event: dict):
        self.recent_events.append(event)
        self.recent_events = self.recent_events[-200:]
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(event, default=str))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)


bus = EventBus()
orchestrator = WealthOrchestrator(on_event=bus.publish)
cycle_lock = threading.Lock()


def run_cycle_blocking():
    if not cycle_lock.acquire(blocking=False):
        logger.warning("Cycle already running; skipping trigger")
        return
    try:
        orchestrator.run_daily_cycle()
    finally:
        cycle_lock.release()


# IMPORTANT: timezone must be set on each CronTrigger explicitly. A standalone
# CronTrigger ignores the scheduler-level timezone and falls back to the
# machine's local zone (UTC on cloud servers) — which silently shifted every
# job by 5.5 hours on the first deployment.
IST = "Asia/Kolkata"
scheduler = BackgroundScheduler(timezone=IST)
scheduler.add_job(
    orchestrator.notify_bot_start,
    CronTrigger(day_of_week="mon-fri", hour=8, minute=45, timezone=IST),
    id="bot_start_notice",
)
scheduler.add_job(
    run_cycle_blocking,
    CronTrigger(day_of_week="mon-fri", hour=8, minute=46, timezone=IST),
    id="premarket_cycle",
)
# Continuous mode: cycles run back-to-back during market hours — as soon as
# one completes, the next starts (60s breather between them to be polite to
# data APIs). Stock rotation in select_symbols() makes each cycle cover
# different stocks, so a full day sweeps most of the NIFTY-100 universe.
_loop_stop = threading.Event()


def continuous_cycle_loop():
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    cooldown = orchestrator.config.get("cycle_cooldown_seconds", 60)
    while not _loop_stop.is_set():
        now = _dt.now(ZoneInfo(IST))
        in_window = (
            now.weekday() < 5
            and (now.hour, now.minute) >= (9, 15)
            and (now.hour, now.minute) <= (14, 45)  # last start leaves room to finish pre-close
        )
        if in_window:
            run_cycle_blocking()
            _loop_stop.wait(cooldown)
        else:
            _loop_stop.wait(60)
scheduler.add_job(
    orchestrator.manage_exits,
    CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone=IST),
    id="exit_management",
)
scheduler.add_job(
    orchestrator.notify_bot_stop,
    CronTrigger(day_of_week="mon-fri", hour=15, minute=30, timezone=IST),
    id="bot_stop_notice",
)
scheduler.add_job(
    orchestrator.generate_eod_report,
    CronTrigger(day_of_week="mon-fri", hour=15, minute=35, timezone=IST),
    id="eod_report",
)
scheduler.add_job(
    lambda: orchestrator.memory.summarize_weekly(orchestrator.llm),
    CronTrigger(day_of_week="sun", hour=9, minute=0, timezone=IST),
    id="weekly_memory_summary",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.loop = asyncio.get_running_loop()
    scheduler.start()
    for job in scheduler.get_jobs():
        logger.info("Scheduled job %s — next run: %s", job.id, job.next_run_time)
    if orchestrator.config.get("continuous_cycles", True):
        threading.Thread(target=continuous_cycle_loop, daemon=True, name="cycle-loop").start()
        logger.info("Continuous cycle loop started (back-to-back 09:15-14:45 IST, %ss cooldown)",
                    orchestrator.config.get("cycle_cooldown_seconds", 60))
    if orchestrator.config.get("portfolio_sentinel", True):
        from wealth_platform.portfolio_monitor import PortfolioSentinel

        sentinel = PortfolioSentinel(orchestrator)
        interval = orchestrator.config.get("sentinel_interval_minutes", 15) * 60
        threading.Thread(
            target=sentinel.run_forever, args=(_loop_stop, interval),
            daemon=True, name="portfolio-sentinel",
        ).start()
    yield
    _loop_stop.set()
    scheduler.shutdown(wait=False)


app = FastAPI(title="AI Wealth Platform", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_PATH.read_text()


@app.get("/api/portfolio")
def portfolio():
    funds = orchestrator.broker.get_funds()
    positions = {
        s: {"quantity": p.quantity, "avg_price": p.average_price, "last_price": p.last_price,
            "pnl": round(p.pnl, 2), "value": round(p.value, 2)}
        for s, p in orchestrator.broker.get_positions().items()
    }
    equity_value = funds.available_cash + sum(p["value"] for p in positions.values())
    mf = {}
    try:
        mf = orchestrator.mf_manager.portfolio_snapshot()
    except Exception:  # noqa: BLE001
        mf = {"holdings": [], "total_value": 0}
    starting_capital = getattr(orchestrator.broker, "starting_cash", orchestrator.config.get("capital", 15000))
    total_value = equity_value + mf.get("total_value", 0)
    return {
        "broker": orchestrator.broker.name,
        "cash": funds.available_cash,
        "positions": positions,
        "stocks_value": round(sum(p["value"] for p in positions.values()), 2),
        "equity_value": round(equity_value, 2),
        "mutual_funds": mf,
        "total_value": round(total_value, 2),
        "starting_capital": starting_capital,
        "pnl": round(total_value - starting_capital, 2),
        "pnl_pct": round((total_value / starting_capital - 1) * 100, 2) if starting_capital else 0,
    }


# Free-tier limits (approximate, as published by each provider mid-2026)
PROVIDER_LIMITS = {
    "groq": {"rpm": 30, "tpm": "12K", "daily": "100K tokens/day"},
    "cerebras": {"rpm": 30, "tpm": "60K", "daily": "1M tokens/day"},
    "gemini": {"rpm": 10, "tpm": "250K", "daily": "250 req/day"},
    "openrouter": {"rpm": 20, "tpm": "-", "daily": "~50 req/day (free models)"},
    "ollama": {"rpm": "-", "tpm": "-", "daily": "unlimited (local)"},
    "anthropic": {"rpm": 50, "tpm": "50K", "daily": "pay-per-use"},
}


@app.get("/api/llm-status")
def llm_status():
    try:
        order = orchestrator.llm._provider_order(None)
    except Exception:  # noqa: BLE001
        order = []
    counts = {r["provider"]: r["calls"] for r in orchestrator.storage.provider_counts_today()}
    providers = [
        {
            "provider": p,
            "model": orchestrator.llm._model_for(p),
            "calls_today": counts.get(p, 0),
            "limits": PROVIDER_LIMITS.get(p, {}),
            "priority": i + 1,
        }
        for i, p in enumerate(order)
    ]
    return {
        "pipeline_mode": orchestrator.token_budget.mode(),
        "calls_remaining": orchestrator.token_budget.calls_remaining(),
        "calibrated_gate": orchestrator.history_rag.calibrated_confidence_gate(
            orchestrator.config.get("research_confidence_gate", 40),
        ),
        "providers": providers,
    }


@app.get("/api/history")
def history():
    return orchestrator.storage.portfolio_history(365)


@app.get("/api/trades")
def trades(limit: int = 1000):
    return orchestrator.storage.recent_trades(limit)


@app.get("/api/decisions")
def decisions():
    return orchestrator.storage.recent_decisions(50)


@app.get("/api/cycles")
def cycles():
    return orchestrator.storage.recent_cycles(10)


@app.get("/api/cycles/{cycle_id}/messages")
def cycle_messages(cycle_id: int):
    return orchestrator.storage.cycle_messages(cycle_id)


@app.get("/api/eod")
def eod():
    return orchestrator.storage.latest_eod_report() or {"report": "No EOD report yet."}


@app.post("/api/run-cycle")
def trigger_cycle():
    if cycle_lock.locked():
        return JSONResponse({"status": "already_running"}, status_code=409)
    threading.Thread(target=run_cycle_blocking, daemon=True).start()
    return {"status": "started"}


@app.post("/api/run-eod")
def trigger_eod():
    threading.Thread(target=orchestrator.generate_eod_report, daemon=True).start()
    return {"status": "started"}


@app.get("/api/mf/recommend")
def mf_recommend(amount: float = 2000, risk: str = "moderate"):
    return orchestrator.mf_manager.recommend(amount, risk)


@app.get("/api/ipos")
def ipo_scan():
    cash = orchestrator.broker.get_funds().available_cash
    return orchestrator.ipo_manager.daily_ipo_scan(cash)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    bus.clients.append(ws)
    for event in bus.recent_events[-50:]:
        await ws.send_text(json.dumps(event, default=str))
    try:
        while True:
            await ws.receive_text()  # keepalive; client doesn't send commands
    except WebSocketDisconnect:
        if ws in bus.clients:
            bus.clients.remove(ws)
