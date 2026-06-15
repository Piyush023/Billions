"""Persistence for cycles, agent messages, trades, and snapshots.

Backends (set via env):
  - SQLite (default): ``data/platform.db`` — zero setup, local file
  - PostgreSQL: ``DATABASE_URL=postgresql://...`` — Supabase, Neon, etc.

When DATABASE_URL is set, all reads/writes go to Postgres. SQLite file is ignored.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List, Optional, Tuple

from wealth_platform.db_url import DatabaseUrlError, mask_database_url, resolve_database_url
from wealth_platform.paths import DB_PATH, ensure_data_dir

ensure_data_dir()

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT DEFAULT 'running',
    symbols TEXT,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER,
    symbol TEXT,
    agent_name TEXT,
    role TEXT,
    report TEXT,
    provider TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER,
    symbol TEXT,
    research_rating TEXT,
    trade_proposal TEXT,
    pm_decision TEXT,
    approved INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER,
    symbol TEXT,
    side TEXT,
    quantity INTEGER,
    price REAL,
    broker TEXT,
    order_id TEXT,
    status TEXT,
    reasoning TEXT,
    pnl REAL,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_value REAL,
    cash REAL,
    positions TEXT,
    mf_value REAL DEFAULT 0,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS eod_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date TEXT UNIQUE,
    report TEXT,
    created_at TEXT
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id SERIAL PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT DEFAULT 'running',
    symbols TEXT,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS agent_messages (
    id SERIAL PRIMARY KEY,
    cycle_id INTEGER,
    symbol TEXT,
    agent_name TEXT,
    role TEXT,
    report TEXT,
    provider TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    cycle_id INTEGER,
    symbol TEXT,
    research_rating TEXT,
    trade_proposal TEXT,
    pm_decision TEXT,
    approved INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    cycle_id INTEGER,
    symbol TEXT,
    side TEXT,
    quantity INTEGER,
    price DOUBLE PRECISION,
    broker TEXT,
    order_id TEXT,
    status TEXT,
    reasoning TEXT,
    pnl DOUBLE PRECISION,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    total_value DOUBLE PRECISION,
    cash DOUBLE PRECISION,
    positions TEXT,
    mf_value DOUBLE PRECISION DEFAULT 0,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS eod_reports (
    id SERIAL PRIMARY KEY,
    report_date TEXT UNIQUE,
    report TEXT,
    created_at TEXT
);
"""


def _mask_database_url(url: str) -> str:
    return mask_database_url(url)


class Storage:
    def __init__(self, db_path: str = DB_PATH, database_url: Optional[str] = None):
        try:
            self.database_url = resolve_database_url(database_url)
        except DatabaseUrlError:
            raise
        self.backend = "postgres" if self.database_url else "sqlite"
        self.db_path = db_path
        if self.backend == "sqlite":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._conn() as conn:
            self._init_schema(conn)
            self._migrate(conn)

    @property
    def backend_label(self) -> str:
        if self.backend == "postgres":
            return f"postgres ({_mask_database_url(self.database_url)})"
        return f"sqlite ({os.path.abspath(self.db_path)})"

    def _init_schema(self, conn):
        if self.backend == "sqlite":
            conn.executescript(SQLITE_SCHEMA)
        else:
            for stmt in POSTGRES_SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

    def _migrate(self, conn):
        if self.backend == "sqlite":
            cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
            if "pnl" not in cols:
                conn.execute("ALTER TABLE trades ADD COLUMN pnl REAL")
        else:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='trades' AND column_name='pnl'"
            ).fetchall()
            if not rows:
                conn.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS pnl DOUBLE PRECISION")

    def _adapt(self, sql: str) -> str:
        if self.backend == "sqlite":
            return sql
        return sql.replace("?", "%s")

    @contextmanager
    def _conn(self):
        if self.backend == "postgres":
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self.database_url, row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self.db_path, timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=15000")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _execute(self, conn, sql: str, params: Tuple = ()) -> Any:
        cur = conn.execute(self._adapt(sql), params)
        return cur

    def _insert_returning_id(self, conn, sql: str, params: Tuple) -> int:
        if self.backend == "postgres":
            cur = self._execute(conn, sql + " RETURNING id", params)
            row = cur.fetchone()
            return int(row["id"])
        cur = self._execute(conn, sql, params)
        return int(cur.lastrowid)

    # -- cycles ---------------------------------------------------------

    def start_cycle(self, symbols: List[str]) -> int:
        with self._conn() as conn:
            return self._insert_returning_id(
                conn,
                "INSERT INTO cycles (started_at, symbols) VALUES (?, ?)",
                (datetime.now().isoformat(), json.dumps(symbols)),
            )

    def finish_cycle(self, cycle_id: int, summary: str):
        with self._conn() as conn:
            self._execute(
                conn,
                "UPDATE cycles SET finished_at=?, status='done', summary=? WHERE id=?",
                (datetime.now().isoformat(), summary, cycle_id),
            )

    # -- agent messages --------------------------------------------------

    def log_agent_message(self, cycle_id: int, symbol: str, agent_name: str, role: str, report: str, provider: str):
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO agent_messages (cycle_id, symbol, agent_name, role, report, provider, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cycle_id, symbol, agent_name, role, report, provider, datetime.now().isoformat()),
            )

    # -- decisions & trades -----------------------------------------------

    def log_decision(self, cycle_id: int, symbol: str, rating: str, proposal: dict, pm: dict, approved: bool):
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO decisions (cycle_id, symbol, research_rating, trade_proposal, pm_decision, approved, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cycle_id, symbol, rating, json.dumps(proposal), json.dumps(pm), int(approved), datetime.now().isoformat()),
            )

    def log_trade(
        self, cycle_id: Optional[int], symbol: str, side: str, quantity: int, price: float,
        broker: str, order_id: str, status: str, reasoning: str = "",
        pnl: Optional[float] = None,
    ):
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO trades (cycle_id, symbol, side, quantity, price, broker, order_id, status, reasoning, pnl, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cycle_id, symbol, side, quantity, price, broker, order_id, status, reasoning,
                    round(pnl, 2) if pnl is not None else None, datetime.now().isoformat(),
                ),
            )

    def snapshot_portfolio(self, total_value: float, cash: float, positions: dict, mf_value: float = 0):
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO portfolio_snapshots (total_value, cash, positions, mf_value, created_at) VALUES (?, ?, ?, ?, ?)",
                (total_value, cash, json.dumps(positions), mf_value, datetime.now().isoformat()),
            )

    def save_eod_report(self, report: str):
        with self._conn() as conn:
            if self.backend == "postgres":
                self._execute(
                    conn,
                    "INSERT INTO eod_reports (report_date, report, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT (report_date) DO UPDATE SET report = EXCLUDED.report, created_at = EXCLUDED.created_at",
                    (str(datetime.now().date()), report, datetime.now().isoformat()),
                )
            else:
                self._execute(
                    conn,
                    "INSERT OR REPLACE INTO eod_reports (report_date, report, created_at) VALUES (?, ?, ?)",
                    (str(datetime.now().date()), report, datetime.now().isoformat()),
                )

    # -- queries for the dashboard ---------------------------------------

    def _rows(self, query: str, params=()) -> List[dict]:
        with self._conn() as conn:
            cur = self._execute(conn, query, params if isinstance(params, tuple) else tuple(params))
            rows = cur.fetchall()
            if self.backend == "sqlite":
                return [dict(r) for r in rows]
            return list(rows)

    def recent_cycles(self, limit=10):
        return self._rows("SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,))

    def cycle_messages(self, cycle_id: int):
        return self._rows("SELECT * FROM agent_messages WHERE cycle_id=? ORDER BY id", (cycle_id,))

    def recent_trades(self, limit=50):
        trades = self._rows("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
        missing = [t["id"] for t in trades if t["side"] == "SELL" and t["status"] == "filled" and t["pnl"] is None]
        if missing:
            computed = self._fifo_pnl()
            for t in trades:
                if t["id"] in missing and t["id"] in computed:
                    t["pnl"] = computed[t["id"]]
        return trades

    def _fifo_pnl(self) -> dict:
        rows = self._rows("SELECT id, symbol, side, quantity, price FROM trades WHERE status='filled' ORDER BY id")
        lots: dict = {}
        result = {}
        for r in rows:
            sym = r["symbol"]
            if r["side"] == "BUY":
                lots.setdefault(sym, []).append([r["quantity"], r["price"] or 0.0])
                continue
            qty, pnl = r["quantity"], 0.0
            queue = lots.get(sym, [])
            while qty > 0 and queue:
                lot = queue[0]
                take = min(qty, lot[0])
                pnl += ((r["price"] or 0.0) - lot[1]) * take
                lot[0] -= take
                qty -= take
                if lot[0] == 0:
                    queue.pop(0)
            result[r["id"]] = round(pnl, 2)
        return result

    def recent_decisions(self, limit=50):
        return self._rows("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

    def portfolio_history(self, limit=365):
        return self._rows("SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT ?", (limit,))

    def latest_eod_report(self):
        rows = self._rows("SELECT * FROM eod_reports ORDER BY id DESC LIMIT 1")
        return rows[0] if rows else None

    def provider_counts_today(self):
        today = str(datetime.now().date())
        if self.backend == "postgres":
            return self._rows(
                "SELECT provider, COUNT(*) AS calls FROM agent_messages "
                "WHERE created_at::date = ?::date GROUP BY provider",
                (today,),
            )
        return self._rows(
            "SELECT provider, COUNT(*) AS calls FROM agent_messages "
            "WHERE created_at LIKE ? GROUP BY provider",
            (today + "%",),
        )
