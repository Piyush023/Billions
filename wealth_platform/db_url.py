"""Parse and validate PostgreSQL connection settings from environment."""

import os
import re
from typing import Optional
from urllib.parse import quote, urlparse


class DatabaseUrlError(ValueError):
    """Raised when DATABASE_URL or Postgres env vars are invalid."""


def resolve_database_url(explicit: Optional[str] = None) -> Optional[str]:
    """Return a validated postgresql:// URL or None for SQLite mode.

    Supports:
      - DATABASE_URL / POSTGRES_URL (single URI)
      - POSTGRES_HOST + POSTGRES_PASSWORD (+ optional USER, PORT, DB)
        — use this when the password contains @ # : / etc.
    """
    raw = (explicit or os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or "").strip()
    if raw:
        return _normalize_uri(raw)

    host = (os.environ.get("POSTGRES_HOST") or os.environ.get("SUPABASE_DB_HOST") or "").strip()
    password = os.environ.get("POSTGRES_PASSWORD") or os.environ.get("SUPABASE_DB_PASSWORD") or ""
    if host and password:
        user = (os.environ.get("POSTGRES_USER") or "postgres").strip()
        port = (os.environ.get("POSTGRES_PORT") or "5432").strip()
        db = (os.environ.get("POSTGRES_DB") or "postgres").strip()
        return (
            f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
            f"@{host}:{port}/{quote(db, safe='')}"
        )
    return None


def _normalize_uri(raw: str) -> str:
    url = raw.strip().strip('"').strip("'")
    if not url:
        raise DatabaseUrlError("DATABASE_URL is empty")

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    if not url.startswith("postgresql://"):
        raise DatabaseUrlError(
            "DATABASE_URL must start with postgresql:// (or postgres://). "
            f"Got: {url[:40]}..."
        )

    if re.search(r"\[.*?\]", url):
        raise DatabaseUrlError(
            "DATABASE_URL still contains placeholder brackets like [YOUR-PASSWORD]. "
            "Replace them with your real Supabase/Neon password."
        )

    parsed = urlparse(url)
    if not parsed.hostname:
        raise DatabaseUrlError(
            "DATABASE_URL could not be parsed — usually the password contains special "
            "characters (@, #, :, /, %) that break the URI.\n\n"
            "Fix option A — URL-encode the password in DATABASE_URL:\n"
            "  @ → %40   # → %23   : → %3A   / → %2F   % → %25\n\n"
            "Fix option B — use separate vars in .env (recommended):\n"
            "  POSTGRES_HOST=db.xxxxx.supabase.co\n"
            "  POSTGRES_PASSWORD=your-raw-password-here\n"
            "  POSTGRES_USER=postgres\n"
            "  POSTGRES_PORT=5432\n"
            "  POSTGRES_DB=postgres\n"
            "  (remove or comment out DATABASE_URL)\n"
        )

    if len(parsed.hostname) > 253:
        raise DatabaseUrlError(f"DATABASE_URL hostname looks invalid (too long): {parsed.hostname[:60]}...")

    return url


def mask_database_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.hostname or "?"
        port = p.port or 5432
        user = p.username or "?"
        db = (p.path or "/").lstrip("/") or "postgres"
        return f"postgresql://{user}:***@{host}:{port}/{db}"
    except Exception:  # noqa: BLE001
        return "postgresql://***"
