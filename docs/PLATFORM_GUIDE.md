# AI Wealth Platform — Complete Flow Guide

This document explains every flow in the `wealth_platform/` codebase: what runs,
when, why, and what it costs. The design rule throughout is **free-tier first**:
heavy LLM reasoning happens once per day; everything intraday is pure math.

---

## 1. Architecture Overview

```
                        ┌──────────────────────────────┐
                        │   Dashboard (dashboard.html)  │
                        │  live feed · portfolio · EOD  │
                        └───────┬──────────▲───────────┘
                          REST  │          │ WebSocket
                        ┌───────▼──────────┴───────────┐
                        │     server.py (FastAPI)       │
                        │  scheduler · API · event bus  │
                        └───────────────┬──────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │  orchestrator.py              │
                        │  the daily decision cycle     │
                        └──┬─────────┬─────────┬───────┘
                           │         │         │
              ┌────────────▼──┐ ┌────▼────┐ ┌──▼─────────────┐
              │ agents/        │ │ brokers/ │ │ investments/   │
              │ 12 LLM agents  │ │ paper    │ │ mutual_funds   │
              │ (TradingAgents │ │ zerodha  │ │ ipo_manager    │
              │  style)        │ │ groww    │ │                │
              └───────┬───────┘ └─────────┘ └────────────────┘
                      │
              ┌───────▼───────┐         ┌────────────────┐
              │ llm/llm_client │         │ storage.py      │
              │ Groq→Gemini→   │         │ SQLite: cycles, │
              │ Ollama→Haiku   │         │ trades, reports │
              └───────────────┘         └────────────────┘
```

---

## 2. The LLM Layer (`wealth_platform/llm/llm_client.py`)

Every agent talks to `LLMClient.chat()`, which tries providers in priority
order and falls through automatically on rate limits or errors:

| Priority | Provider | Model | Cost | Limits |
|---|---|---|---|---|
| 1 | Groq | llama-3.3-70b-versatile | Free | 14,400 req/day |
| 2 | Google Gemini | gemini-2.0-flash | Free | 1,500 req/day |
| 3 | Ollama (local) | llama3.1:8b (configurable) | Free | unlimited, needs 8GB+ RAM |
| 4 | Anthropic | claude-haiku-4-5 | ~Rs.2/day | only if key set; only PM agent prefers it |

A full daily cycle for 3 stocks makes **~36 LLM calls** — comfortably inside
Groq's free tier alone. If Groq rate-limits mid-cycle, the call silently moves
to Gemini, then Ollama. **A single provider outage never stops the cycle.**

`chat_json()` wraps `chat()` for agents that must return structured JSON
(Research Manager, Trader, Portfolio Manager); `extract_json()` tolerates
markdown fences and surrounding prose.

---

## 3. The Agent Pipeline (`wealth_platform/agents/`)

Modeled on TauricResearch/TradingAgents, implemented without LangGraph
(plain sequential Python — fewer dependencies, easier to debug).

### Stage 1 — Analyst Team (4 agents, `analysts.py`)

| Agent | Data it gathers (free) | Verdict format |
|---|---|---|
| `TechnicalAnalyst` | 6 months OHLCV from yfinance → RSI, MACD, SMAs, Bollinger, volume | BULLISH / BEARISH / NEUTRAL + confidence |
| `FundamentalsAnalyst` | yfinance company info: P/E, P/B, ROE, debt, growth, margins | UNDERVALUED / FAIR / OVERVALUED + confidence |
| `NewsAnalyst` | yfinance company news + Google News RSS (Indian market) | POSITIVE / NEGATIVE / NEUTRAL + confidence |
| `SentimentAnalyst` | Reasons over news report + volume/price action (no paid social API) | BULLISH / BEARISH / MIXED + confidence |

Each gathers raw data in Python, then asks the LLM to *interpret* it — the
LLM never invents numbers, it reasons over computed facts.

### Stage 2 — Researcher Debate (`researchers.py`)

`run_debate()` alternates `BullResearcher` and `BearResearcher` for
`debate_rounds` rounds (default 1 to save tokens). Each side must directly
rebut the other's strongest points. The full transcript goes to the
`ResearchManager`, which returns structured JSON:

```json
{"rating": "BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL",
 "confidence": 0-100, "rationale": "...", "key_risks": [...],
 "time_horizon": "intraday|swing|positional"}
```

**Gate:** if rating is HOLD or confidence < 40, the pipeline stops here for
that stock — no trader/risk/PM tokens are spent.

### Stage 3 — Trader (`trader_agent.py`)

Receives the research verdict + live quote + available cash + open positions.
Its system prompt encodes small-capital realities: ~Rs.60 round-trip costs,
max 25% per stock, min Rs.1,000 trade. Returns:

```json
{"action": "BUY|SELL|HOLD", "quantity": N, "entry_price": X,
 "stop_loss": X, "take_profit": X, "trade_type": "...", "reasoning": "..."}
```

**Gate:** action HOLD or quantity 0 ends the pipeline for that stock.

### Stage 4 — Risk Debate (`risk_debators.py`)

Three perspectives run sequentially on the trade proposal:
`AggressiveDebator` (take the risk) → `ConservativeDebator` (preserve capital,
sees the aggressive argument) → `NeutralDebator` (sees both, arbitrates).

### Stage 5 — Portfolio Manager (`portfolio_manager_agent.py`)

The final gatekeeper. Receives *everything*: research verdict, proposal, risk
debate, portfolio state, and **lessons from past decisions** (see Memory).
Hard rules in its prompt: never exceed available cash, never >25% in one
stock, never risk >2.5% of capital on one stop. Returns APPROVE/REJECT with
an optional `adjusted_quantity`. This is the only agent that prefers Claude
Haiku when `ANTHROPIC_API_KEY` is set; otherwise it uses the free tier.

**Only an APPROVE reaches the broker.** Unparseable output = automatic REJECT.

### Memory (`agent_memory.py`)

Every PM decision and every realized exit is appended to
`data/agent_memory.md`. The most recent ~4,000 characters are fed back into
the PM's context on every future decision — the cheapest possible form of
"learning from history", and you can read/edit the file yourself.

---

## 4. Broker Layer (`wealth_platform/brokers/`)

All brokers implement `BaseBroker`: `connect / get_quote / place_order /
get_positions / get_holdings / get_funds`. The orchestrator never knows which
one is active. Selected by `"broker"` in `wealth_config.json`.

| Broker | File | API cost | Notes |
|---|---|---|---|
| **paper** (default) | `paper_broker.py` | Free | Simulates fills with realistic Indian costs (Rs.20 brokerage + STT + charges). State persists to `data/paper_portfolio.json` across restarts. |
| **groww** | `groww_broker.py` | **Free API** | Official `growwapi` SDK. Auth = API key + TOTP secret (env vars). Recommended live broker for the free-tier setup. Also exposes holdings (MF-friendly). |
| **zerodha** | `zerodha_broker.py` | Rs.500/month | Kept for compatibility with your existing setup. Token expires daily — run `python zerodha_auth.py`. |

**Safety:** if a live broker fails to connect, `get_broker()` falls back to
paper automatically — the platform degrades to simulation, never crashes into
accidental live orders.

---

## 5. Mutual Funds (`investments/mutual_funds.py`)

- **Data:** free AMFI NAV feed via `mftool` — no key, no cost.
- **Holdings:** you record units once (`data/mf_holdings.json` or
  `MutualFundManager.add_holding()`); the platform then values them live and
  shows them in the dashboard's total.
- **Advisory:** `recommend(monthly_amount, risk_profile)` has the LLM allocate
  a SIP across a curated universe (index core + flexi/mid/small per risk +
  liquid fund), returning structured allocations with reasoning.
- **Honest limitation:** no broker offers a free public API to *place* MF
  orders (Groww's trade API is equities/F&O only). So the platform researches,
  recommends, monitors, and values — you tap "invest" in the Groww app
  (~30 seconds). Everything else is automated.

## 6. IPOs (`investments/ipo_manager.py`)

- **Data:** NSE's public current/upcoming-issues endpoints (free; the client
  primes cookies and degrades gracefully if NSE blocks).
- **Analysis:** each open/upcoming IPO goes through one LLM call that knows
  SME-vs-mainboard risk, lot affordability vs your capital, and the 30%-of-
  capital cap. Output: APPLY / AVOID / RESEARCH_MORE + confidence + risks.
- **Honest limitation:** IPO application legally requires your UPI mandate
  approval (ASBA) — it cannot and should not be automated. The platform's job
  is to make sure you never miss an issue and always have an analyzed
  recommendation before close. Runs inside the daily cycle (capped at 5 LLM
  calls) and on demand via `GET /api/ipos`.

---

## 7. The Daily Cycle (`orchestrator.py`)

```
08:45 IST (Mon-Fri, scheduler) or POST /api/run-cycle
│
├─ select_symbols()       existing dynamic_stock_screener if importable,
│                         else wealth_config.json watchlist (top 3)
├─ for each symbol:
│    analysts → debate → research gate → trader → gate →
│    risk debate → PM → APPROVE? → broker.place_order()
│    (every agent output: saved to SQLite + streamed to dashboard live)
├─ daily_ipo_scan()       analyze open/upcoming IPOs
├─ _snapshot()            portfolio snapshot (equity + MF) → SQLite
└─ cycle summary → SQLite

09:00–15:30 IST, every 5 min (scheduler)
└─ manage_exits()         pure math: stop_loss_pct (-7%) / take_profit_pct
                          (+14%) checks on every position. ZERO LLM calls.
                          Exits are executed, logged, and written to memory
                          so the PM learns from realized outcomes.

15:35 IST (Mon-Fri, scheduler) or POST /api/run-eod
└─ generate_eod_report()  ONE LLM call summarising the day → SQLite +
                          dashboard + Telegram (if configured)
```

Token budget for a normal day: ~36 free Groq calls + 1 optional Haiku call
+ up to 5 IPO calls. **Effective cost: Rs.0–2/day.**

---

## 8. Storage (`storage.py`)

SQLite at `data/platform.db` — free, zero-config, survives restarts. Tables:

| Table | Contents |
|---|---|
| `cycles` | every decision cycle: start/end, symbols, summary |
| `agent_messages` | every agent report, per cycle/symbol — the full audit trail |
| `decisions` | research rating, trade proposal, PM verdict, approved flag |
| `trades` | every order: side, qty, price, broker, order id, status, reasoning |
| `portfolio_snapshots` | total value, cash, positions, MF value over time |
| `eod_reports` | one report per trading day |

The dashboard reads exclusively from these tables + live broker state, so
every number on screen is reconstructable and auditable.

---

## 9. Running the Platform — Complete Instructions

### 9.1 One-time setup

```bash
cd <project-root>

# 1. Create a virtual environment (required on macOS — system Python is locked)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements_platform.txt

# 3. Create .env with your keys (see docs/DEPENDENCIES_NEEDED.md)
#    Minimum required: GROQ_API_KEY
#    The platform auto-loads .env on startup — no manual exports needed.
```

### 9.2 Start the platform

```bash
# Option A — platform only
source .venv/bin/activate
uvicorn wealth_platform.server:app --host 0.0.0.0 --port 8000

# Option B — platform + your nodemailer service together
MAILER_DIR=~/path/to/your/mailer ./start_platform.sh
```

Then open **http://localhost:8000**. On startup the scheduler arms three jobs
(IST): full decision cycle **08:45 Mon–Fri**, math-only exit checks **every
5 min 09:00–15:30**, EOD report **15:35**.

### 9.3 Trigger things manually

| What | How |
|---|---|
| Run a decision cycle now | Click **▶ Run Cycle** on the dashboard, or `curl -X POST localhost:8000/api/run-cycle` |
| Generate EOD report now | Click **EOD Report**, or `curl -X POST localhost:8000/api/run-eod` |
| MF SIP recommendation | `curl "localhost:8000/api/mf/recommend?amount=2000&risk=moderate"` |
| Scan current IPOs | `curl localhost:8000/api/ipos` |
| Run a cycle without the server | `python -m wealth_platform.orchestrator` |

A cycle takes ~2–5 minutes for 3 stocks (mostly LLM latency). The dashboard's
pipeline view and live feed show every agent's report as it lands. Only one
cycle can run at a time (a second trigger returns HTTP 409).

### 9.4 Inspect state

- **Dashboard**: positions, trades, decisions, EOD — http://localhost:8000
- **Database**: open `data/platform.db` in DBeaver / DB Browser for SQLite
- **Paper portfolio**: `data/paper_portfolio.json` (cash, positions, fills)
- **PM's memory**: `data/agent_memory.md` (human-readable lessons file)
- **Server log**: wherever you redirected uvicorn's output

### 9.5 Switch brokers (paper → live)

1. Prove profitability in paper mode first (20+ trades, 55%+ win rate).
2. Put `GROWW_API_KEY` / `GROWW_API_SECRET` in `.env` (already done if you
   followed setup).
3. Change `"broker": "paper"` → `"broker": "groww"` in `wealth_config.json`.
4. Restart the server. If Groww auth fails, the platform **falls back to
   paper automatically** and logs a warning — check the log after restart.

### 9.6 Stop / restart

- Foreground: `Ctrl-C` (start_platform.sh also stops the mailer).
- Background: `pkill -f "uvicorn wealth_platform"`.
- State (SQLite, paper portfolio, memory) survives restarts; the scheduler
  re-arms automatically on startup.

### 9.7 Server reference (`server.py`, `dashboard.html`)

**REST endpoints**

| Endpoint | Purpose |
|---|---|
| `GET /` | the dashboard |
| `GET /api/portfolio` | cash, positions, MF value, totals |
| `GET /api/history` | portfolio value snapshots (charting) |
| `GET /api/trades` · `/api/decisions` · `/api/cycles` | logs |
| `GET /api/cycles/{id}/messages` | full agent transcript of a cycle |
| `GET /api/eod` | latest end-of-day report |
| `POST /api/run-cycle` | trigger a decision cycle now (409 if running) |
| `POST /api/run-eod` | trigger EOD report now |
| `GET /api/mf/recommend?amount=2000&risk=moderate` | MF SIP advice |
| `GET /api/ipos` | scan + analyze current IPOs |
| `WS /ws` | live event stream (replays last 50 events on connect) |

**Dashboard panels:** stat cards (total/cash/equity/MF + trigger buttons),
animated 5-stage decision pipeline, live agent feed (every report as it
happens, color-coded), positions, recent trades, PM decisions, and the EOD
report. Pure HTML/JS served by FastAPI — nothing to build, free to host.

**Free hosting:** Oracle Cloud Free Tier ARM VM (permanently free) running
`uvicorn` behind its public IP, or just your own machine. SQLite means no
database service needed.

---

## 10. Safety Model (defense in depth)

1. **Paper by default** — live trading requires editing `wealth_config.json`.
2. **Two LLM gates** — Research Manager confidence gate, then PM
   approve/reject with hard capital rules in the prompt.
3. **Parse-fail = reject** — any unparseable LLM output results in no trade.
4. **Math-only exits** — stops/targets never depend on an LLM being up.
5. **Broker fallback** — live connection failure degrades to paper, loudly.
6. **Cycle lock** — concurrent cycle triggers are refused (409).
7. **Full audit trail** — every report, decision, and order is in SQLite.

## 11. File Map

```
wealth_platform/
├── llm/llm_client.py            free-tier LLM router with fallback
├── agents/
│   ├── base_agent.py            shared agent runner + event streaming
│   ├── analysts.py              Technical/Fundamentals/News/Sentiment
│   ├── researchers.py           Bull, Bear, debate loop, Research Manager
│   ├── trader_agent.py          trade proposal JSON
│   ├── risk_debators.py         Aggressive/Conservative/Neutral debate
│   ├── portfolio_manager_agent.py  final APPROVE/REJECT gate
│   └── agent_memory.py          lessons file fed back to the PM
├── brokers/
│   ├── base_broker.py           common interface
│   ├── paper_broker.py          default; realistic costs; persisted state
│   ├── groww_broker.py          free live API (recommended)
│   └── zerodha_broker.py        legacy paid option
├── investments/
│   ├── mutual_funds.py          AMFI NAVs, valuation, LLM SIP advisor
│   └── ipo_manager.py           NSE IPO feed + LLM apply/avoid analysis
├── storage.py                   SQLite audit trail
├── orchestrator.py              the daily cycle (also runnable directly)
├── server.py                    FastAPI + scheduler + WebSocket
└── dashboard.html               the platform UI
wealth_config.json               your settings (no secrets!)
requirements_platform.txt        dependencies
docs/DEPENDENCIES_NEEDED.md      what YOU must provide (keys, accounts)
```

Existing modules (`dynamic_stock_screener.py`, `ai_trading_engine.py`, etc.)
remain untouched and are used opportunistically: the screener picks the daily
universe when importable, and ML signals can be passed into
`TechnicalAnalyst.analyze(symbol, ml_signal=...)`.
