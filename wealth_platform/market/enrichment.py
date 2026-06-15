"""Indian-market context enrichment before analysts run — rule-based, zero LLM."""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger("wealth_platform.market.enrichment")


def _sym(symbol: str) -> str:
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def enrich(symbol: str) -> str:
    """Build a compact factual block: vs NIFTY, recent move, volume, 52w range."""
    lines = [f"Market enrichment for {symbol} ({datetime.now():%Y-%m-%d}):"]
    try:
        stock = yf.download(_sym(symbol), period="6mo", interval="1d", progress=False, auto_adjust=True)
        nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False, auto_adjust=True)
        if stock.empty:
            return lines[0] + " price data unavailable."

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
        if not nifty.empty and isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)

        close = stock["Close"]
        price = float(close.iloc[-1])
        ch_5d = (price / float(close.iloc[-6]) - 1) * 100 if len(close) > 6 else 0
        ch_1m = (price / float(close.iloc[-22]) - 1) * 100 if len(close) > 22 else 0
        hi_52 = float(close.max())
        lo_52 = float(close.min())
        vol = stock["Volume"]
        vol_ratio = float(vol.iloc[-5:].mean()) / max(float(vol.rolling(20).mean().iloc[-1]), 1)

        lines.append(f"  Last: Rs.{price:.2f} | 5d: {ch_5d:+.1f}% | 1m: {ch_1m:+.1f}%")
        lines.append(f"  52w range: Rs.{lo_52:.0f}–Rs.{hi_52:.0f} ({100 * price / hi_52:.0f}% of high)")
        lines.append(f"  Volume vs 20d avg: {vol_ratio:.2f}x")

        if not nifty.empty and len(nifty) > 22:
            n_close = nifty["Close"]
            nifty_1m = (float(n_close.iloc[-1]) / float(n_close.iloc[-22]) - 1) * 100
            rel = ch_1m - nifty_1m
            lines.append(f"  vs NIFTY 1m: stock {ch_1m:+.1f}% vs index {nifty_1m:+.1f}% (relative {rel:+.1f}%)")

        if ch_5d > 8:
            lines.append("  Note: sharp 5-day rally — watch for mean-reversion / ASM risk on small caps.")
        if vol_ratio > 2.5:
            lines.append("  Note: volume spike — event or momentum day likely.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Enrichment failed for %s: %s", symbol, exc)
        lines.append(f"  enrichment error: {exc}")
    return "\n".join(lines)
