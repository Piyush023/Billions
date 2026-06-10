"""IPO module: fetches current/upcoming NSE IPOs and runs an LLM apply/avoid analysis.

Data source: NSE India public endpoints (free, no key — needs browser-like
headers and a cookie-priming request). IPO APPLICATION cannot be automated
(UPI mandate + ASBA require human approval by regulation), so the platform's
job is detection + analysis + a clear APPLY/AVOID recommendation with
reasoning, delivered to the dashboard and Telegram before the issue closes.
"""

import logging
from typing import List

import requests

from wealth_platform.llm.llm_client import LLMClient, extract_json

logger = logging.getLogger("wealth_platform.ipo")

NSE_BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/all-upcoming-issues-ipo",
}


class IPOManager:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _prime_cookies(self):
        try:
            self.session.get(NSE_BASE, timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSE cookie priming failed: %s", exc)

    def fetch_ipos(self) -> List[dict]:
        """Current + upcoming IPOs from NSE. Returns [] gracefully if NSE blocks."""
        self._prime_cookies()
        ipos = []
        for endpoint in ("/api/ipo-current-issue", "/api/all-upcoming-issues?category=ipo"):
            try:
                resp = self.session.get(NSE_BASE + endpoint, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                items = data if isinstance(data, list) else data.get("data", [])
                for item in items:
                    ipos.append(
                        {
                            "company": item.get("companyName") or item.get("company"),
                            "symbol": item.get("symbol"),
                            "issue_start": item.get("issueStartDate"),
                            "issue_end": item.get("issueEndDate"),
                            "price_band": item.get("priceBand") or f"{item.get('issuePrice', '')}",
                            "issue_size": item.get("issueSize"),
                            "lot_size": item.get("lotSize"),
                            "series": item.get("series", "EQ"),
                            "status": item.get("status", "upcoming"),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NSE IPO fetch failed (%s): %s", endpoint, exc)
        return ipos

    def analyze_ipo(self, ipo: dict, available_capital: float) -> dict:
        """LLM evaluates one IPO and returns an APPLY/AVOID recommendation."""
        system = (
            "You are an IPO analyst for an Indian retail investor with small capital. "
            "Evaluate the IPO using the given facts. Consider: SME vs mainboard (series SM/ST = SME, "
            "which is far riskier and has larger lots), issue size, price band sanity, and whether one "
            "lot fits the investor's capital (never recommend more than 30% of capital into one IPO). "
            "You may not have financials — say so and weight your confidence down accordingly. "
            "Respond with JSON: "
            '{"recommendation": "APPLY"|"AVOID"|"RESEARCH_MORE", "confidence": 0-100, '
            '"lots_suggested": <int>, "reasoning": "<3-5 sentences>", "key_risks": ["..."]}'
        )
        user = f"Available capital: Rs.{available_capital}\n\nIPO details:\n{ipo}"
        try:
            result = self.llm.chat_json(system, user)
        except Exception as exc:  # noqa: BLE001
            logger.error("IPO analysis failed: %s", exc)
            result = {"recommendation": "RESEARCH_MORE", "confidence": 0, "reasoning": str(exc), "key_risks": []}
        result["ipo"] = ipo
        return result

    def daily_ipo_scan(self, available_capital: float) -> List[dict]:
        """Fetch open/upcoming IPOs and analyze each. Called by the orchestrator."""
        ipos = self.fetch_ipos()
        analyses = []
        for ipo in ipos[:5]:  # cap LLM calls — free-tier discipline
            analyses.append(self.analyze_ipo(ipo, available_capital))
        return analyses
