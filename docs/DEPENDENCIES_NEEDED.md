# What You Need to Provide

Everything the platform needs from you, in priority order. Items marked
**REQUIRED** are needed for the first run; everything else can wait.

All secrets go in a `.env` file (or exported env vars) — **never** in
`wealth_config.json` or any committed file.

---

## 1. LLM Provider Key — REQUIRED (pick at least one, both recommended)

| Provider | Where to get it | Env var | Cost |
|---|---|---|---|
| **Groq** (primary) | console.groq.com → API Keys | `GROQ_API_KEY` | Free, 14,400 req/day |
| **Gemini** (fallback) | aistudio.google.com → Get API key | `GEMINI_API_KEY` | Free, 1,500 req/day |
| Ollama (offline fallback) | Install from ollama.com, then `ollama pull llama3.1:8b` | `OLLAMA_MODEL` (optional) | Free, needs 8GB+ RAM |
| Anthropic (optional, PM only) | console.anthropic.com | `ANTHROPIC_API_KEY` | ~Rs.50-80/month |

Sign-up time: ~2 minutes each. No credit card needed for Groq or Gemini.

## 2. Python Environment — REQUIRED

```bash
python3 --version          # need 3.10+
pip install -r requirements_platform.txt
```

## 3. Notifications — your nodemailer service (primary) or Telegram (fallback)

The platform emails you trade executions, exits, and the EOD report via your
existing Node.js nodemailer service. Configure in `.env`:

```
EMAIL_SERVICE_URL=http://localhost:3001/send   # your nodemailer endpoint
EMAIL_TO=piyushkhurana23@gmail.com
# EMAIL_SERVICE_API_KEY=                       # optional, sent as Bearer token
```

The platform POSTs JSON: `{"to", "subject", "text", "html"}`. If your
service expects different field names or a different route, adjust either the
service or `wealth_platform/notifications.py` (one small payload dict).

Run both together: `MAILER_DIR=~/path/to/mailer ./start_platform.sh`

Telegram still works as a fallback if you ever set `TELEGRAM_BOT_TOKEN` +
`TELEGRAM_CHAT_ID`. ⚠️ Your old bot token is committed in `config.json` in
plaintext — **revoke it in BotFather** even if you never use Telegram again.
Same for the Zerodha keys in that file.

## 4. Broker (only when you go live — NOT needed for paper trading)

**Groww (recommended — free API):**
1. Groww app/web → profile → **Trading APIs** → generate API key + TOTP secret
2. ```
   GROWW_API_KEY=...
   GROWW_API_SECRET=...   # the TOTP secret
   ```
3. Set `"broker": "groww"` in `wealth_config.json`

**Zerodha (only if you prefer it — Rs.500/month API fee):**
- `ZERODHA_API_KEY`, `ZERODHA_SECRET_KEY`, then run `python zerodha_auth.py`
  daily (token expires every day).

## 5. Mutual Fund Holdings (optional, one-time)

If you already hold funds, record them once so the dashboard values them:

```python
from wealth_platform.llm.llm_client import LLMClient
from wealth_platform.investments.mutual_funds import MutualFundManager
mf = MutualFundManager(LLMClient())
mf.add_holding(scheme_code="120716", units=45.231, avg_nav=110.50)
```

(Scheme code is on your fund statement, or search amfiindia.com.)

## 6. Your `.env` file (template)

```bash
# --- LLM (at least one required) ---
GROQ_API_KEY=
GEMINI_API_KEY=
# ANTHROPIC_API_KEY=          # optional, PM quality boost ~Rs.60/mo

# --- Notifications (recommended) ---
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# --- Broker (only for live mode) ---
# GROWW_API_KEY=
# GROWW_API_SECRET=
```

## 7. First Run Checklist

```bash
# 1. Install deps
pip install -r requirements_platform.txt

# 2. Set keys (at minimum GROQ_API_KEY)
export GROQ_API_KEY=gsk_...

# 3. Start the platform
uvicorn wealth_platform.server:app --port 8000

# 4. Open http://localhost:8000 and click "▶ Run Cycle"
#    Watch the agents debate live. Everything is paper-traded.
```

## 8. Monthly Cost Summary

| Item | Cost |
|---|---|
| LLM (Groq + Gemini free tiers) | Rs.0 |
| LLM (optional Claude Haiku for PM) | Rs.50–80 |
| Hosting (your machine or Oracle Free Tier) | Rs.0 |
| Database (SQLite) | Rs.0 |
| Groww trading API | Rs.0 |
| Market data (yfinance + AMFI + NSE public) | Rs.0 |
| **Total** | **Rs.0–80/month** |
