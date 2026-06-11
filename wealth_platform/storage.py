"""SQLite persistence for every decision cycle, agent message, trade, and
portfolio snapshot. Free, zero-config, and the dashboard reads straight from it.
Swap the path to a Postgres/Supabase URL later if you outgrow it."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

DB_PATH = os.path.join("data", "platform.db")

SCHEMA = """
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


class Storage:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        # WAL allows concurrent readers while one thread writes; the busy
        # timeout makes a second writer wait instead of raising
        # "database is locked" when the cycle loop and sentinel log at once.
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- cycles ---------------------------------------------------------

    def start_cycle(self, symbols: List[str]) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO cycles (started_at, symbols) VALUES (?, ?)",
                (datetime.now().isoformat(), json.dumps(symbols)),
            )
            return cur.lastrowid

    def finish_cycle(self, cycle_id: int, summary: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE cycles SET finished_at=?, status='done', summary=? WHERE id=?",
                (datetime.now().isoformat(), summary, cycle_id),
            )

    # -- agent messages --------------------------------------------------

    def log_agent_message(self, cycle_id: int, symbol: str, agent_name: str, role: str, report: str, provider: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_messages (cycle_id, symbol, agent_name, role, report, provider, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cycle_id, symbol, agent_name, role, report, provider, datetime.now().isoformat()),
            )

    # -- decisions & trades -----------------------------------------------

    def log_decision(self, cycle_id: int, symbol: str, rating: str, proposal: dict, pm: dict, approved: bool):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO decisions (cycle_id, symbol, research_rating, trade_proposal, pm_decision, approved, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cycle_id, symbol, rating, json.dumps(proposal), json.dumps(pm), int(approved), datetime.now().isoformat()),
            )

    def log_trade(self, cycle_id: Optional[int], symbol: str, side: str, quantity: int, price: float,
                  broker: str, order_id: str, status: str, reasoning: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO trades (cycle_id, symbol, side, quantity, price, broker, order_id, status, reasoning, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cycle_id, symbol, side, quantity, price, broker, order_id, status, reasoning, datetime.now().isoformat()),
            )

    def snapshot_portfolio(self, total_value: float, cash: float, positions: dict, mf_value: float = 0):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots (total_value, cash, positions, mf_value, created_at) VALUES (?, ?, ?, ?, ?)",
                (total_value, cash, json.dumps(positions), mf_value, datetime.now().isoformat()),
            )

    def save_eod_report(self, report: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eod_reports (report_date, report, created_at) VALUES (?, ?, ?)",
                (str(datetime.now().date()), report, datetime.now().isoformat()),
            )

    # -- queries for the dashboard ---------------------------------------

    def _rows(self, query: str, params=()) -> List[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def recent_cycles(self, limit=10):
        return self._rows("SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,))

    def cycle_messages(self, cycle_id: int):
        return self._rows("SELECT * FROM agent_messages WHERE cycle_id=? ORDER BY id", (cycle_id,))

    def recent_trades(self, limit=50):
        return self._rows("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))

    def recent_decisions(self, limit=50):
        return self._rows("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

    def portfolio_history(self, limit=365):
        return self._rows("SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT ?", (limit,))

    def latest_eod_report(self):
        rows = self._rows("SELECT * FROM eod_reports ORDER BY id DESC LIMIT 1")
        return rows[0] if rows else None
