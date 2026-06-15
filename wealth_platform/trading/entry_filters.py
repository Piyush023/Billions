"""Code-enforced entry filters — trend, pullback, ATR stops. Zero LLM cost."""

import logging
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger("wealth_platform.trading.entry_filters")


def _sym(symbol: str) -> str:
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def _download(sym: str, period: str = "6mo"):
    df = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def nifty_trend_ok() -> Tuple[bool, str]:
    """Index must be above SMA50 for new long entries."""
    try:
        df = _download("^NSEI")
        if len(df) < 55:
            return True, "nifty data thin — gate skipped"
        close = df["Close"]
        price = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        if price >= sma50:
            return True, f"NIFTY Rs.{price:.0f} above SMA50 Rs.{sma50:.0f}"
        return False, f"NIFTY Rs.{price:.0f} below SMA50 Rs.{sma50:.0f} — risk-off for new buys"
    except Exception as exc:  # noqa: BLE001
        logger.warning("NIFTY trend check failed: %s", exc)
        return True, "nifty check failed — gate skipped"


def stock_entry_ok(symbol: str, config: dict) -> Tuple[bool, str]:
    """Price above SMA50 and not buying an extended blow-off top."""
    max_high = config.get("max_pct_of_52w_high", 0.97)
    require_sma = config.get("require_sma50", True)
    try:
        df = _download(_sym(symbol))
        if len(df) < 55:
            return True, "stock data thin — gate skipped"
        close = df["Close"]
        price = float(close.iloc[-1])
        hi_52 = float(close.max())
        pct_of_high = price / hi_52 if hi_52 else 1.0
        if pct_of_high > max_high:
            return False, f"{symbol} at {pct_of_high:.0%} of 52w high (max {max_high:.0%}) — extended"
        if require_sma:
            sma50 = float(close.rolling(50).mean().iloc[-1])
            if price < sma50:
                return False, f"{symbol} Rs.{price:.0f} below SMA50 Rs.{sma50:.0f}"
        return True, f"{symbol} trend OK ({pct_of_high:.0%} of 52w high)"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stock entry check failed for %s: %s", symbol, exc)
        return True, "stock check failed — gate skipped"


def atr_stop_price(symbol: str, entry_price: float, config: dict) -> Optional[float]:
    """ATR-based stop clamped to min/max stop % from config."""
    if not config.get("use_atr_stops", True):
        return None
    try:
        df = _download(_sym(symbol), period="3mo")
        if len(df) < 20:
            return None
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        mult = config.get("atr_stop_multiplier", 2.0)
        stop = entry_price - atr * mult
        min_pct = config.get("min_stop_loss_pct", 5.0) / 100
        max_pct = config.get("max_stop_loss_pct", 8.0) / 100
        stop = max(stop, entry_price * (1 - max_pct))
        stop = min(stop, entry_price * (1 - min_pct))
        return round(stop, 2)
    except Exception:  # noqa: BLE001
        return None


def normalize_stops(entry: float, stop: float, target: float, config: dict) -> Tuple[float, float]:
    """Ensure stop/target align with desk config and ATR when available."""
    stop_pct = config.get("stop_loss_pct", 6.0) / 100
    target_pct = config.get("take_profit_pct", 12.0) / 100
    if stop <= 0:
        stop = entry * (1 - stop_pct)
    if target <= 0:
        target = entry * (1 + target_pct)
    if stop >= entry:
        stop = entry * (1 - stop_pct)
    if target <= entry:
        target = entry * (1 + target_pct)
    return round(stop, 2), round(target, 2)
