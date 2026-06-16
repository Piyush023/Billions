#!/usr/bin/env python3
"""Copy existing SQLite data into PostgreSQL (one-time migration).

Usage (single URI — password must be URL-encoded if it has @#:/ etc.):
  export DATABASE_URL='postgresql://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres'
  python scripts/migrate_sqlite_to_postgres.py

Usage (separate vars — password can be raw, recommended for Supabase):
  export POSTGRES_HOST=db.xxxxx.supabase.co
  export POSTGRES_PASSWORD='your password with special chars!'
  python scripts/migrate_sqlite_to_postgres.py

Test connection only:
  python scripts/migrate_sqlite_to_postgres.py --check
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
    parser.add_argument("--check", action="store_true", help="Test Postgres connection only")
    args = parser.parse_args()

    from wealth_platform.db_url import (
        DatabaseUrlError,
        diagnose_config,
        mask_database_url,
        open_postgres_connection,
        resolve_database_url,
    )

    print("Config check:")
    for k, v in diagnose_config().items():
        print(f"  {k}: {v}")

    try:
        database_url = resolve_database_url()
    except DatabaseUrlError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    if not database_url:
        print("\nERROR: Postgres not configured.")
        print("Set POSTGRES_HOST + POSTGRES_PASSWORD in .env (see docs/DATABASE.md)")
        print("Comment out any broken DATABASE_URL line in .env")
        sys.exit(1)

    print(f"\nTarget: {mask_database_url(database_url)}")

    from psycopg.rows import dict_row

    try:
        with open_postgres_connection(database_url, row_factory=dict_row) as pg:
            row = pg.execute("SELECT 1 AS ok").fetchone()
            print(f"Connection OK (postgres responded: {row})")
            from wealth_platform.db_url import _clean_host, resolve_ipv4
            host = _clean_host(os.environ.get("POSTGRES_HOST", ""))
            if host:
                print(f"  IPv4 hostaddr used: {resolve_ipv4(host)}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not connect to Postgres: {exc}")
        sys.exit(1)

    if args.check:
        return

    if not os.path.exists(args.sqlite):
        print(f"ERROR: SQLite file not found: {args.sqlite}")
        sys.exit(1)

    from wealth_platform.storage import Storage

    src = sqlite3.connect(args.sqlite)
    src.row_factory = sqlite3.Row
    dst_storage = Storage(database_url=database_url)
    assert dst_storage.backend == "postgres"

    with open_postgres_connection(database_url, row_factory=dict_row) as pg:
        for table in TABLES:
            rows = src.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            if not rows:
                print(f"  {table}: 0 rows (skip)")
                continue
            cols = rows[0].keys()
            col_list = ", ".join(cols)
            placeholders = ", ".join(["%s"] * len(cols))
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

    print("\nDone. Add the same Postgres vars to .env and restart the platform.")


if __name__ == "__main__":
    main()
