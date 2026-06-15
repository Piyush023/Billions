# Database setup — PostgreSQL (Supabase / Neon)

The platform uses **SQLite by default** (`data/platform.db`). Set `DATABASE_URL` to switch to **PostgreSQL** with a web UI you can browse from anywhere.

---

## Recommended free hosts

| Provider | Free tier | Web GUI | Best for |
|----------|-----------|---------|----------|
| **[Supabase](https://supabase.com)** | 500 MB DB, 2 projects | **Table Editor** in dashboard | Easiest GUI — browse rows like a spreadsheet |
| **[Neon](https://neon.tech)** | 512 MB, 1 project | SQL Editor in dashboard | Serverless Postgres, good performance |

Both give you a connection string like:
```
postgresql://postgres:YOUR_PASSWORD@db.xxxxx.supabase.co:5432/postgres
```

---

## Step 1 — Create Supabase project (5 min)

1. Sign up at [supabase.com](https://supabase.com)
2. **New project** → pick region close to your Oracle VM (e.g. Mumbai / Singapore)
3. Save the **database password** (shown once)
4. Go to **Project Settings → Database → Connection string → URI**
5. Copy the URI and replace `[YOUR-PASSWORD]` with your password

Example:
```
postgresql://postgres.xxxxx:MySecretPass@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
```

Use **Session mode** (port 5432) or **Transaction pooler** (port 6543) — either works for this app.

---

## Step 2 — Configure the platform

On your **Oracle VM** (and locally if you test there):

```bash
nano ~/Billions/.env
```

Add:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@db.xxxxx.supabase.co:5432/postgres
```

Install the Postgres driver and restart:

```bash
cd ~/Billions
.venv/bin/pip install 'psycopg[binary]>=3.1'
sudo systemctl restart wealth-platform
```

Verify:
```bash
curl -s localhost:8000/api/data-status | python3 -m json.tool
```

You should see `"backend": "postgres"` and table counts.

---

## Step 3 — Migrate existing SQLite data (optional)

If you already have trades on the server in `data/platform.db`:

```bash
# On the server — copy SQLite from server, or run there directly
export DATABASE_URL='postgresql://...'
.venv/bin/python scripts/migrate_sqlite_to_postgres.py --sqlite data/platform.db
sudo systemctl restart wealth-platform
```

---

## Step 4 — Browse data in Supabase GUI

1. Open your project at [supabase.com/dashboard](https://supabase.com/dashboard)
2. **Table Editor** (left sidebar)
3. Click tables: `trades`, `cycles`, `decisions`, `agent_messages`, etc.

No SSH or file copy needed — all data is visible in the browser.

**SQL Editor** (for custom queries):
```sql
SELECT * FROM trades ORDER BY id DESC LIMIT 20;
SELECT * FROM cycles ORDER BY id DESC LIMIT 10;
```

---

## Step 5 — Connect from your Mac (TablePlus / DBeaver)

Use the same `DATABASE_URL` credentials:

| Field | Value |
|-------|--------|
| Host | `db.xxxxx.supabase.co` |
| Port | `5432` |
| User | `postgres` |
| Password | your project password |
| Database | `postgres` |
| SSL | **Required** |

Supabase: **Project Settings → Database → Connection string → Direct connection**

---

## What stays in SQLite / JSON files

Even with Postgres, these remain **local files** on the VM:

| File | Purpose |
|------|---------|
| `data/paper_portfolio.json` | Paper broker cash & positions |
| `data/desk_state.json` | Entry thesis, sentinel notes |
| `data/agent_memory.md` | PM memory |

Only the **audit trail** (cycles, trades, decisions, agent reports, EOD) moves to Postgres.

---

## Troubleshooting

**Connection refused / timeout**
- Supabase **Database Settings → Network** — allow your Oracle VM public IP (or use “allow all” temporarily for testing)

**`backend` still `sqlite`**
- `DATABASE_URL` not in `.env` or service not restarted after edit

**Empty tables after migrate**
- Run a cycle: `curl -X POST localhost:8000/api/run-cycle`
- Re-check `/api/data-status`

**SSL required**
- Supabase requires SSL; `psycopg` enables it automatically from the URI

---

## Neon alternative (brief)

1. [neon.tech](https://neon.tech) → Create project
2. Copy connection string from dashboard
3. Same steps: `DATABASE_URL` in `.env`, `pip install psycopg[binary]`, restart
4. Use **Tables** or **SQL Editor** in Neon console to browse data
