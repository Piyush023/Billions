"""Analyst team: Technical, Fundamentals, News, Sentiment.

Each analyst gathers its own data (yfinance / NSE / RSS — all free), then
asks the LLM to interpret it and write a short professional report.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from wealth_platform.agents.base_agent import BaseAgent

logger = logging.getLogger("wealth_platform.agents.analysts")


def _yf_symbol(symbol: str) -> str:
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


class TechnicalAnalyst(BaseAgent):
    name = "technical_analyst"
    role = "Technical Analyst"
    system_prompt = (
        "You are a senior technical analyst at an Indian equity trading desk. "
        "You receive computed indicator values and recent price action for one NSE stock. "
        "Interpret trend, momentum, support/resistance, and volume behaviour. "
        "End your report with a one-line verdict: BULLISH, BEARISH, or NEUTRAL, with confidence (0-100)."
    )

    def analyze(self, symbol: str, ml_signal: Optional[dict] = None):
        data = self._gather(symbol)
        context = f"Stock: {symbol} (NSE)\nDate: {datetime.now():%Y-%m-%d}\n\n{data}"
        if ml_signal:
            context += (
                f"\n\nQuant ML model signal (ensemble of RandomForest/XGBoost/LightGBM): "
                f"{ml_signal.get('signal', 'N/A')} with confidence {ml_signal.get('confidence', 'N/A')}. "
                "Weigh this as one independent input, not ground truth."
            )
        return self.run(context)

    @staticmethod
    def _gather(symbol: str) -> str:
        df = yf.download(_yf_symbol(symbol), period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return "No price data available."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        std20 = close.rolling(20).std()
        last = -1
        lines = [
            f"Last close: {close.iloc[last]:.2f}",
            f"1-week change: {(close.iloc[last] / close.iloc[-6] - 1) * 100:.2f}%",
            f"1-month change: {(close.iloc[last] / close.iloc[-22] - 1) * 100:.2f}%",
            f"RSI(14): {rsi.iloc[last]:.1f}",
            f"MACD: {macd.iloc[last]:.2f} vs signal {signal.iloc[last]:.2f}",
            f"SMA20: {sma20.iloc[last]:.2f} | SMA50: {sma50.iloc[last]:.2f} | SMA200: {sma200.iloc[last]:.2f}",
            f"Bollinger: upper {(sma20.iloc[last] + 2 * std20.iloc[last]):.2f}, "
            f"lower {(sma20.iloc[last] - 2 * std20.iloc[last]):.2f}",
            f"Avg volume (20d): {df['Volume'].rolling(20).mean().iloc[last]:,.0f}",
            f"Last volume: {df['Volume'].iloc[last]:,.0f}",
            f"52w high: {close.max():.2f} | 52w low: {close.min():.2f}",
        ]
        return "\n".join(lines)


class FundamentalsAnalyst(BaseAgent):
    name = "fundamentals_analyst"
    role = "Fundamentals Analyst"
    system_prompt = (
        "You are a fundamentals research analyst covering Indian equities. "
        "You receive financial metrics for one company. Evaluate valuation, profitability, "
        "leverage, and growth. Flag red flags explicitly. "
        "End with a one-line verdict: UNDERVALUED, FAIRLY VALUED, or OVERVALUED, with confidence (0-100)."
    )

    def analyze(self, symbol: str):
        data = self._gather(symbol)
        return self.run(f"Stock: {symbol} (NSE)\n\n{data}")

    @staticmethod
    def _gather(symbol: str) -> str:
        try:
            ticker = yf.Ticker(_yf_symbol(symbol))
            info = ticker.info or {}
        except Exception as exc:  # noqa: BLE001
            return f"Fundamental data unavailable: {exc}"
        keys = {
            "marketCap": "Market cap",
            "trailingPE": "Trailing P/E",
            "forwardPE": "Forward P/E",
            "priceToBook": "P/B",
            "returnOnEquity": "ROE",
            "debtToEquity": "Debt/Equity",
            "profitMargins": "Profit margin",
            "revenueGrowth": "Revenue growth (yoy)",
            "earningsGrowth": "Earnings growth (yoy)",
            "freeCashflow": "Free cash flow",
            "totalCash": "Total cash",
            "totalDebt": "Total debt",
            "dividendYield": "Dividend yield",
            "heldPercentInstitutions": "Institutional holding",
            "sector": "Sector",
            "industry": "Industry",
        }
        lines = [f"{label}: {info[k]}" for k, label in keys.items() if info.get(k) is not None]
        return "\n".join(lines) if lines else "No fundamental metrics returned by data source."


class NewsAnalyst(BaseAgent):
    name = "news_analyst"
    role = "News Analyst"
    system_prompt = (
        "You are a macro and news analyst for Indian markets. You receive recent headlines for "
        "a company plus broad market headlines. Assess what is materially relevant for the stock "
        "over the next days to weeks. Ignore noise. "
        "End with a one-line verdict: POSITIVE, NEGATIVE, or NEUTRAL news environment, with confidence (0-100)."
    )

    def analyze(self, symbol: str):
        headlines = self._gather(symbol)
        return self.run(f"Stock: {symbol} (NSE)\nDate: {datetime.now():%Y-%m-%d}\n\n{headlines}")

    @staticmethod
    def _gather(symbol: str) -> str:
        sections = []
        try:
            news = yf.Ticker(_yf_symbol(symbol)).news or []
            company_lines = []
            for item in news[:10]:
                content = item.get("content", item)
                title = content.get("title", "")
                when = content.get("pubDate", "")
                if title:
                    company_lines.append(f"- {title} ({when})")
            if company_lines:
                sections.append("Company headlines:\n" + "\n".join(company_lines))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Company news fetch failed for %s: %s", symbol, exc)
        try:
            feed = requests.get(
                "https://news.google.com/rss/search",
                params={"q": "nifty sensex indian stock market", "hl": "en-IN", "gl": "IN"},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            import re

            titles = re.findall(r"<title>(.*?)</title>", feed.text)[1:9]
            if titles:
                sections.append("Market headlines:\n" + "\n".join(f"- {t}" for t in titles))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Market news fetch failed: %s", exc)
        return "\n\n".join(sections) if sections else "No recent news retrieved."


class SentimentAnalyst(BaseAgent):
    name = "sentiment_analyst"
    role = "Sentiment Analyst"
    system_prompt = (
        "You are a market sentiment analyst. You receive recent headlines and price/volume context "
        "for one NSE stock. Infer the prevailing crowd mood: are retail and institutions accumulating, "
        "distributing, fearful, or euphoric? Note that you have no direct social media feed, so reason "
        "from news tone, volume behaviour, and price action; say so when uncertain. "
        "End with a one-line verdict: sentiment BULLISH, BEARISH, or MIXED, with confidence (0-100)."
    )

    def analyze(self, symbol: str, news_report: str, technical_report: str):
        context = (
            f"Stock: {symbol} (NSE)\n\n"
            f"News analyst's report:\n{news_report}\n\n"
            f"Technical analyst's report (use volume/price behaviour as sentiment proxy):\n{technical_report}"
        )
        return self.run(context)
