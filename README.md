# 🏦 Billions — AI Wealth Platform

An autonomous, free-tier AI wealth manager for Indian markets: a 12-agent LLM
decision pipeline (analysts → bull/bear debate → trader → risk debate →
portfolio manager), multi-broker execution (paper / Groww / Zerodha), mutual
fund advisory, IPO analysis, and a live web dashboard — at a running cost of
**₹0–80/month**.

## Repository structure

```
Billions/
├── wealth_platform/          ⭐ THE ACTIVE PROJECT — everything runs from here
│   ├── llm/                  Free-tier LLM router (Groq → Gemini → Ollama → Haiku)
│   ├── agents/               12 LLM agents + memory + trade-history RAG
│   ├── brokers/              paper (default) / Groww / Zerodha adapters
│   ├── investments/          Mutual funds (AMFI) + IPO analysis (NSE)
│   ├── orchestrator.py       The daily decision cycle
│   ├── server.py             FastAPI + scheduler + WebSocket
│   ├── storage.py            SQLite audit trail
│   └── dashboard.html        The web UI
│
├── legacy_bot/               🗄 OLD quantitative bot (pre-LLM era). Not part of
│                             the platform; kept because the platform optionally
│                             reuses its stock screener. Don't run it directly.
│
├── docs/
│   ├── PLATFORM_GUIDE.md     How every flow works + complete running instructions
│   ├── DEPLOYMENT.md         Oracle Cloud Free Tier deployment guide
│   └── DEPENDENCIES_NEEDED.md  Keys/accounts you must provide
│
├── deploy/                   systemd unit files for the server
├── data/                     Runtime state: SQLite DB, paper portfolio, agent
│                             memory (created automatically; never committed)
├── .env.example              Template for your secrets — copy to .env
├── wealth_config.json        Platform settings (broker, capital, mode — no secrets)
├── requirements_platform.txt Python dependencies
└── start_platform.sh         Starts platform (+ optional nodemailer) together
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements_platform.txt
cp .env.example .env && nano .env        # add GROQ_API_KEY at minimum
.venv/bin/uvicorn wealth_platform.server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000 → click "▶ Run Cycle"
```

Starts in **paper trading mode** — no real money moves until you change
`"broker"` in `wealth_config.json`. Full instructions: [docs/PLATFORM_GUIDE.md](docs/PLATFORM_GUIDE.md).

## The decision cycle (daily, 08:45 IST, automatic)

```
Screener picks 3 stocks
  → 4 analyst agents (technical, fundamentals, news, sentiment)
  → bull vs bear researcher debate → research manager verdict
  → trader proposes position (positional, 2-12 week holds, CNC only)
  → aggressive/conservative/neutral risk debate
  → portfolio manager approves/rejects (with RAG over past trade records)
  → approved orders → broker · everything logged to SQLite + live dashboard
  → 15:35 IST: LLM end-of-day report → email
```

## Safety model

- Paper mode by default; live trading is an explicit config change
- Two LLM gates + hard-coded capital rules + no-short guard in code
- Unparseable LLM output = automatic rejection
- Stops/targets are pure math — they work even if every LLM is down
- Full audit trail of every agent report, decision, and order
