#!/usr/bin/env python3
"""Copy existing SQLite data into PostgreSQL (one-time migration).

Usage:
  export DATABASE_URL='postgresql://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres'
  python scripts/migrate_sqlite_to_postgres.py
  python scripts/migrate_sqlite_to_postgres.py --sqlite data/platform.db
"""

import argparse
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TABLES = [
    "cycles",
    "agent_messages",
    "decisions",
    "trades",
    "portfolio_snapshots",
    "eod_reports",
]


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite platform.db → PostgreSQL")
    parser.add_argument("--sqlite", default=os.path.join(ROOT, "data", "platform.db"))
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if not database_url:
        print("ERROR: set DATABASE_URL to your Postgres connection string")
        sys.exit(1)
    if not os.path.exists(args.sqlite):
        print(f"ERROR: SQLite file not found: {args.sqlite}")
        sys.exit(1)

    import psycopg
    from psycopg.rows import dict_row

    from wealth_platform.storage import Storage

    src = sqlite3.connect(args.sqlite)
    src.row_factory = sqlite3.Row
    dst_storage = Storage(database_url=database_url)
    assert dst_storage.backend == "postgres"

    with psycopg.connect(database_url, row_factory=dict_row) as pg:
        for table in TABLES:
            rows = src.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            if not rows:
                print(f"  {table}: 0 rows (skip)")
                continue
            cols = rows[0].keys()
            col_list = ", ".join(cols)
            placeholders = ", ".join(["%s"] * len(cols))
            # Preserve IDs so foreign keys (cycle_id) stay valid
            sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT (id) DO NOTHING"
            if table == "eod_reports":
                sql = (
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    "ON CONFLICT (report_date) DO NOTHING"
                )
            inserted = 0
            with pg.cursor() as cur:
                for row in rows:
                    cur.execute(sql, tuple(row[c] for c in cols))
                    inserted += cur.rowcount
                # Reset serial sequence to max(id)
                if "id" in cols:
                    cur.execute(f"SELECT COALESCE(MAX(id), 0) AS m FROM {table}")
                    max_id = cur.fetchone()["m"]
                    if max_id:
                        cur.execute(
                            "SELECT setval(pg_get_serial_sequence(%s, 'id'), %s, true)",
                            (table, max_id),
                        )
            pg.commit()
            print(f"  {table}: {len(rows)} source rows, {inserted} inserted")

    print("\nDone. Set DATABASE_URL in .env and restart the platform.")


if __name__ == "__main__":
    main()
