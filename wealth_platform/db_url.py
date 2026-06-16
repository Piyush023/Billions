"""Parse and validate PostgreSQL connection settings from environment."""

import os
import re
import socket
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse


class DatabaseUrlError(ValueError):
    """Raised when DATABASE_URL or Postgres env vars are invalid."""


def resolve_database_url(explicit: Optional[str] = None) -> Optional[str]:
    """Return a validated postgresql:// URL or None for SQLite mode.

    Priority:
      1. explicit argument
      2. POSTGRES_HOST + POSTGRES_PASSWORD (recommended — avoids URI encoding issues)
      3. DATABASE_URL / POSTGRES_URL
    """
    if explicit and explicit.strip():
        return _finalize_url(_normalize_uri(explicit.strip()))

    component_url = _url_from_components()
    if component_url:
        return _finalize_url(component_url)

    raw = (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or "").strip()
    if raw:
        return _finalize_url(_normalize_uri(raw))

    return None


def _url_from_components() -> Optional[str]:
    host_raw = (os.environ.get("POSTGRES_HOST") or os.environ.get("SUPABASE_DB_HOST") or "").strip()
    password = os.environ.get("POSTGRES_PASSWORD") or os.environ.get("SUPABASE_DB_PASSWORD") or ""
    if not host_raw or not password:
        return None

    # User pasted a full URI into POSTGRES_HOST — parse it instead
    if host_raw.startswith("postgres://") or host_raw.startswith("postgresql://"):
        return _normalize_uri(host_raw)

    host = _clean_host(host_raw)
    if not host or host in (".", "..", "..."):
        raise DatabaseUrlError(
            f"POSTGRES_HOST is invalid: {host_raw!r}\n"
            "Use only the hostname, e.g. db.abcdefghijklmnop.supabase.co\n"
            "(Supabase → Project Settings → Database → Host)"
        )

    user = (os.environ.get("POSTGRES_USER") or "postgres").strip()
    port = (os.environ.get("POSTGRES_PORT") or "5432").strip()
    db = (os.environ.get("POSTGRES_DB") or "postgres").strip().lstrip("/")

    if not user:
        raise DatabaseUrlError("POSTGRES_USER is empty — set it to postgres")

    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(db, safe='')}"
    )


def _clean_host(host: str) -> str:
    host = host.strip().strip('"').strip("'")
    host = host.replace("https://", "").replace("http://", "")
    # db.xxx.supabase.co:5432 → db.xxx.supabase.co
    if host.count(":") == 1 and not host.startswith("["):
        host = host.split(":", 1)[0]
    host = host.split("/")[0].strip()
    return host


def _normalize_uri(raw: str) -> str:
    url = raw.strip().strip('"').strip("'")
    if not url:
        raise DatabaseUrlError("DATABASE_URL is empty")

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    if not url.startswith("postgresql://"):
        raise DatabaseUrlError(
            "DATABASE_URL must start with postgresql:// (or postgres://). "
            f"Got: {url[:60]}..."
        )

    if re.search(r"\[.*?\]", url):
        raise DatabaseUrlError(
            "DATABASE_URL still contains placeholder brackets like [YOUR-PASSWORD]. "
            "Copy the real password from Supabase → Project Settings → Database."
        )

    parsed = urlparse(url)
    if not parsed.hostname:
        raise DatabaseUrlError(_bad_uri_help(url))

    host = parsed.hostname.strip()
    if not host or host in (".", "..", "...") or len(host) > 253:
        raise DatabaseUrlError(_bad_uri_help(url))

    if not parsed.username:
        raise DatabaseUrlError(
            "DATABASE_URL is missing the username (should be postgres or postgres.xxxx for Supabase).\n"
            + _bad_uri_help(url)
        )

    return url


def _bad_uri_help(url: str) -> str:
    return (
        "DATABASE_URL could not be parsed — usually the password contains @ # : / % "
        "that break the URI, or DATABASE_URL is still a template.\n\n"
        "Recommended fix — in .env comment out DATABASE_URL and use:\n"
        "  POSTGRES_HOST=db.xxxxx.supabase.co\n"
        "  POSTGRES_PASSWORD=your-real-password\n"
        "  POSTGRES_USER=postgres\n"
        "  POSTGRES_PORT=5432\n"
        "  POSTGRES_DB=postgres\n\n"
        f"Current value starts with: {url[:50]}..."
    )


def _finalize_url(url: str) -> str:
    """Add sslmode=require for known cloud hosts if not already set."""
    if "sslmode=" in url:
        return url
    lower = url.lower()
    if any(x in lower for x in ("supabase", "neon.tech", "pooler.supabase")):
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}sslmode=require"
    return url


def mask_database_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.hostname or "(missing host)"
        port = p.port or 5432
        user = p.username or "(missing user)"
        db = (p.path or "/").lstrip("/") or "postgres"
        return f"postgresql://{user}:***@{host}:{port}/{db}"
    except Exception:  # noqa: BLE001
        return "postgresql://(unparseable)"


def diagnose_config() -> Dict[str, Any]:
    """Non-secret summary for --check / debugging."""
    return {
        "DATABASE_URL_set": bool((os.environ.get("DATABASE_URL") or "").strip()),
        "POSTGRES_URL_set": bool((os.environ.get("POSTGRES_URL") or "").strip()),
        "POSTGRES_HOST": _clean_host(os.environ.get("POSTGRES_HOST") or "") or None,
        "POSTGRES_USER": (os.environ.get("POSTGRES_USER") or "").strip() or "postgres (default)",
        "POSTGRES_PORT": (os.environ.get("POSTGRES_PORT") or "").strip() or "5432 (default)",
        "POSTGRES_DB": (os.environ.get("POSTGRES_DB") or "").strip() or "postgres (default)",
        "POSTGRES_PASSWORD_set": bool(os.environ.get("POSTGRES_PASSWORD") or os.environ.get("SUPABASE_DB_PASSWORD")),
        "resolved": mask_database_url(resolve_database_url()) if _can_resolve() else None,
    }


def _can_resolve() -> bool:
    try:
        return resolve_database_url() is not None
    except DatabaseUrlError:
        return False


def _is_ip_literal(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except OSError:
        return False


def resolve_ipv4(host: str) -> str:
    """Resolve hostname to IPv4 — Oracle Cloud VMs often have no working IPv6 route."""
    override = (os.environ.get("POSTGRES_HOSTADDR") or "").strip()
    if override:
        return override
    if _is_ip_literal(host):
        return host
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DatabaseUrlError(
            f"Could not resolve IPv4 address for {host!r}: {exc}\n"
            "Set POSTGRES_HOSTADDR to the IPv4 from: dig +short A db.xxxx.supabase.co"
        ) from exc
    if not infos:
        raise DatabaseUrlError(f"No IPv4 (A record) found for {host!r}")
    return infos[0][4][0]


def open_postgres_connection(database_url: str, row_factory=None):
    """Open psycopg connection, forcing IPv4 (hostaddr) to avoid IPv6 unreachable errors."""
    import psycopg
    from psycopg import conninfo

    kwargs = conninfo.conninfo_to_dict(database_url)
    host = kwargs.get("host")
    if host and not _is_ip_literal(host):
        kwargs["hostaddr"] = resolve_ipv4(host)
    kwargs.setdefault("connect_timeout", 15)
    if row_factory is not None:
        return psycopg.connect(**kwargs, row_factory=row_factory)
    return psycopg.connect(**kwargs)
