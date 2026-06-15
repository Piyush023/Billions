"""Canonical runtime paths — always anchored to the repo root, not process cwd."""

from pathlib import Path

# wealth_platform/paths.py → repo root is one level up from the package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DB_PATH = str(DATA_DIR / "platform.db")
PAPER_PORTFOLIO_PATH = str(DATA_DIR / "paper_portfolio.json")
DESK_STATE_PATH = str(DATA_DIR / "desk_state.json")
SENTINEL_STATE_PATH = str(DATA_DIR / "sentinel_state.json")
AGENT_MEMORY_PATH = str(DATA_DIR / "agent_memory.md")
AGENT_MEMORY_SUMMARY_PATH = str(DATA_DIR / "agent_memory_summary.md")
MF_HOLDINGS_PATH = str(DATA_DIR / "mf_holdings.json")


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
