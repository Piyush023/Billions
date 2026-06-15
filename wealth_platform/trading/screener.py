"""Built-in lightweight stock screener — no TA-Lib, no extra dependencies.

Scans a ~95-stock NIFTY-100 universe with one batched yfinance download and
scores each stock on momentum, trend, volume activity, and distance from its
52-week high. Pure math, zero LLM cost. Results are cached for the day so
hourly cycles don't re-download the universe.

This replaces the legacy_bot screener on servers where TA-Lib isn't installed
(the orchestrator tries this first now).
"""

import logging
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger("wealth_platform.trading.screener")

# Liquid NIFTY-100 constituents (NSE symbols, sans .NS suffix)
NIFTY_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "TCS", "BHARTIARTL", "ICICIBANK", "SBIN", "INFY",
    "BAJFINANCE", "HINDUNILVR", "ITC", "LT", "HCLTECH", "KOTAKBANK", "SUNPHARMA",
    "MARUTI", "M&M", "AXISBANK", "ULTRACEMCO", "NTPC", "BAJAJFINSV", "ONGC",
    "TITAN", "ADANIENT", "ADANIPORTS", "WIPRO", "POWERGRID", "JSWSTEEL",
    "TATAMOTORS", "ASIANPAINT", "NESTLEIND", "COALINDIA", "BAJAJ-AUTO",
    "TATASTEEL", "GRASIM", "HINDALCO", "SBILIFE", "EICHERMOT", "TECHM",
    "HDFCLIFE", "BRITANNIA", "DIVISLAB", "CIPLA", "DRREDDY", "APOLLOHOSP",
    "TATACONSUM", "INDUSINDBK", "HEROMOTOCO", "VEDL", "PIDILITIND", "SIEMENS",
    "DLF", "AMBUJACEM", "GODREJCP", "DABUR", "HAVELLS", "ICICIPRULI", "ABB",
    "BOSCHLTD", "SHREECEM", "TORNTPHARM", "BANKBARODA", "PNB", "CANBK",
    "IOC", "BPCL", "GAIL", "TATAPOWER", "ADANIGREEN", "ADANIPOWER", "LICI",
    "ZOMATO", "PAYTM", "NYKAA", "IRCTC", "INDIGO", "TRENT", "BEL", "HAL",
    "ZYDUSLIFE", "LUPIN", "AUROPHARMA", "MOTHERSON", "TVSMOTOR", "ASHOKLEY",
    "CHOLAFIN", "MUTHOOTFIN", "SRF", "UPL", "JINDALSTEL", "SAIL", "NMDC",
    "BHEL", "CUMMINSIND", "POLYCAB", "ASTRAL", "PAGEIND",
]


class BuiltInScreener:
    """Momentum/volume screener over the NIFTY-100 universe, cached per day."""

    def __init__(self, universe: Optional[List[str]] = None):
        self.universe = universe or NIFTY_UNIVERSE
        self._cache_date: Optional[date] = None
        self._cache: List[Tuple[str, float]] = []

    def ranked_symbols(self, top_n: int = 20) -> List[str]:
        if self._cache_date == date.today() and self._cache:
            return [s for s, _ in self._cache[:top_n]]

        scores = self._score_universe()
        if not scores:
            logger.warning("Screener produced no scores; returning empty list")
            return []
        self._cache = scores
        self._cache_date = date.today()
        top = [s for s, _ in scores[:top_n]]
        logger.info("Screener top %d: %s", min(top_n, 10), top[:10])
        return top

    def score_for(self, symbol: str) -> Optional[float]:
        """Return cached screener score for pre-gate checks."""
        if self._cache_date != date.today() or not self._cache:
            self.ranked_symbols(top_n=len(self.universe))
        for sym, score in self._cache:
            if sym == symbol:
                return score
        return None

    def _score_universe(self) -> List[Tuple[str, float]]:
        tickers = [f"{s}.NS" for s in self.universe]
        try:
            data = yf.download(
                tickers, period="1y", interval="1d", progress=False,
                auto_adjust=True, group_by="ticker", threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Screener batch download failed: %s", exc)
            return []

        scores: List[Tuple[str, float]] = []
        for symbol in self.universe:
            try:
                df = data[f"{symbol}.NS"].dropna()
                if len(df) < 60:
                    continue
                close, volume = df["Close"], df["Volume"]
                price = float(close.iloc[-1])
                if price < 20:  # skip penny-priced
                    continue

                r_1m = price / float(close.iloc[-22]) - 1
                r_3m = price / float(close.iloc[-63]) - 1
                sma50 = float(close.rolling(50).mean().iloc[-1])
                above_sma50 = 1.0 if price > sma50 else 0.0
                vol_ratio = float(volume.iloc[-5:].mean()) / max(float(volume.rolling(60).mean().iloc[-1]), 1.0)
                pct_of_52w_high = price / float(close.max())
                # Prefer pullbacks in uptrends (75–92% of 52w high) over blow-off tops
                if pct_of_52w_high > 0.97:
                    high_penalty = -0.15
                elif pct_of_52w_high < 0.75:
                    high_penalty = -0.05
                else:
                    high_penalty = 0.10 * (1 - abs(pct_of_52w_high - 0.85))

                score = (
                    40 * r_3m
                    + 25 * r_1m
                    + 15 * above_sma50
                    + 10 * min(vol_ratio, 2.0) / 2.0
                    + 10 * high_penalty
                )
                scores.append((symbol, round(float(score), 3)))
            except Exception:  # noqa: BLE001
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
