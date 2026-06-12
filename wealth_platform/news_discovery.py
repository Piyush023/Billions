"""News-driven stock discovery.

Scans the latest Indian market headlines (free RSS feeds), has the LLM extract
which NSE-listed companies are in the news and why, validates the symbols
against real market data, and feeds them into cycle selection — so the bot
also looks at stocks OUTSIDE the static NIFTY-100 universe when they're
making headlines (results, order wins, upgrades, regulatory news).

Cost: one LLM call per refresh, cached for 30 minutes.
"""

import logging
import re
import time
from typing import List

import requests

from wealth_platform.llm.llm_client import LLMClient

logger = logging.getLogger("wealth_platform.news_discovery")

RSS_FEEDS = [
    # Google News searches, India edition
    "https://news.google.com/rss/search?q=NSE+stock+surge+OR+rally+OR+results&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=indian+stocks+buy+order+win+OR+upgrade&hl=en-IN&gl=IN&ceid=IN:en",
    # Economic Times markets RSS
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
]

CACHE_TTL_S = 1800  # 30 minutes


class NewsStockDiscovery:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._cache: List[dict] = []
        self._cache_at: float = 0.0

    def discover(self, max_stocks: int = 5) -> List[dict]:
        """Returns [{symbol, company, reason}] for stocks currently in the news."""
        if time.time() - self._cache_at < CACHE_TTL_S and self._cache:
            return self._cache[:max_stocks]

        headlines = self._fetch_headlines()
        if not headlines:
            return []

        candidates = self._extract_symbols(headlines)
        validated = []
        for item in candidates:
            symbol = (item.get("symbol") or "").upper().replace(".NS", "").strip()
            if not symbol or not re.fullmatch(r"[A-Z0-9&-]{2,20}", symbol):
                continue
            if self._symbol_exists(symbol):
                item["symbol"] = symbol
                validated.append(item)
            if len(validated) >= max_stocks:
                break

        self._cache = validated
        self._cache_at = time.time()
        logger.info("News discovery: %s", [v["symbol"] for v in validated])
        return validated

    def _fetch_headlines(self) -> List[str]:
        titles: List[str] = []
        for feed in RSS_FEEDS:
            try:
                resp = requests.get(feed, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                found = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", resp.text)
                titles.extend(t.strip() for t in found[1:20] if len(t.strip()) > 15)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Feed fetch failed (%s): %s", feed[:50], exc)
        # de-dupe, cap for token budget
        seen, unique = set(), []
        for t in titles:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:40]

    def _extract_symbols(self, headlines: List[str]) -> List[dict]:
        system = (
            "You are a market news parser for Indian equities. From the headlines, identify "
            "individual NSE-LISTED companies with materially positive or negative company-specific "
            "news (results, large orders, upgrades/downgrades, regulatory actions, M&A). "
            "Skip index/macro/sector-wide headlines and foreign stocks. "
            "Respond with JSON: {\"stocks\": [{\"symbol\": \"<NSE trading symbol>\", "
            "\"company\": \"...\", \"reason\": \"<one line: what the news is>\"}]} "
            "Maximum 8 stocks, most newsworthy first. Use correct official NSE symbols."
        )
        try:
            result = self.llm.chat_json(system, "Headlines:\n" + "\n".join(f"- {h}" for h in headlines))
            return result.get("stocks", [])
        except Exception as exc:  # noqa: BLE001
            logger.error("News symbol extraction failed: %s", exc)
            return []

    @staticmethod
    def _symbol_exists(symbol: str) -> bool:
        """Validate against real market data — the LLM sometimes invents tickers."""
        try:
            import yfinance as yf

            df = yf.Ticker(f"{symbol}.NS").history(period="5d")
            return not df.empty
        except Exception:  # noqa: BLE001
            return False
