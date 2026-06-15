"""Trading engine: screener, entry/exit rules, risk guard, desk coordination."""

from wealth_platform.trading.coordination import TRADE_LOCK, SharedDesk
from wealth_platform.trading.entry_filters import (
    atr_stop_price,
    nifty_trend_ok,
    normalize_stops,
    stock_entry_ok,
)
from wealth_platform.trading.exit_levels import ExitSignal, exit_reason
from wealth_platform.trading.risk_guard import RiskGuard
from wealth_platform.trading.screener import BuiltInScreener
from wealth_platform.trading.token_budget import TokenBudget

__all__ = [
    "TRADE_LOCK",
    "SharedDesk",
    "atr_stop_price",
    "nifty_trend_ok",
    "normalize_stops",
    "stock_entry_ok",
    "ExitSignal",
    "exit_reason",
    "RiskGuard",
    "BuiltInScreener",
    "TokenBudget",
]
