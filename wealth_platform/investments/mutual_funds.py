"""Mutual fund module: free AMFI NAV data + LLM-driven fund recommendations.

Data source: mftool (wraps the free AMFI India NAV feed — no API key needed).

IMPORTANT LIMITATION: no broker exposes a free public API to PLACE mutual
fund orders (Groww's trading API covers equities/F&O only). So this module:
  1. Tracks your MF holdings (you enter them once in data/mf_holdings.json)
  2. Computes live portfolio value from AMFI NAVs
  3. Runs an LLM advisor over candidate funds and your goals
  4. Emits BUY/SIP/REDEEM recommendations to the dashboard + Telegram
You execute the recommendation in the Groww app in ~30 seconds. Everything
else (research, monitoring, valuation) is automated.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from wealth_platform.llm.llm_client import LLMClient, extract_json

logger = logging.getLogger("wealth_platform.mutual_funds")

from wealth_platform.paths import MF_HOLDINGS_PATH

HOLDINGS_PATH = MF_HOLDINGS_PATH

# A small curated universe of liquid, low-expense funds for the LLM to pick from.
# Scheme codes are AMFI codes (look up any fund's code with mftool or amfiindia.com).
DEFAULT_FUND_UNIVERSE = {
    "120716": "UTI Nifty 50 Index Fund - Direct - Growth",
    "147622": "Navi Nifty 50 Index Fund - Direct - Growth",
    "122639": "Parag Parikh Flexi Cap Fund - Direct - Growth",
    "118989": "HDFC Mid-Cap Opportunities Fund - Direct - Growth",
    "119598": "SBI Small Cap Fund - Direct - Growth",
    "118825": "ICICI Prudential Liquid Fund - Direct - Growth",
}


class MutualFundManager:
    def __init__(self, llm: LLMClient, holdings_path: str = HOLDINGS_PATH):
        self.llm = llm
        self.holdings_path = holdings_path
        self._mf = None

    @property
    def mf(self):
        if self._mf is None:
            from mftool import Mftool

            self._mf = Mftool()
        return self._mf

    # ------------------------------------------------------------------
    # Holdings & valuation
    # ------------------------------------------------------------------

    def load_holdings(self) -> List[dict]:
        if not os.path.exists(self.holdings_path):
            return []
        with open(self.holdings_path) as f:
            return json.load(f)

    def save_holdings(self, holdings: List[dict]):
        os.makedirs(os.path.dirname(self.holdings_path), exist_ok=True)
        with open(self.holdings_path, "w") as f:
            json.dump(holdings, f, indent=2)

    def add_holding(self, scheme_code: str, units: float, avg_nav: float):
        holdings = self.load_holdings()
        holdings.append(
            {"scheme_code": scheme_code, "units": units, "avg_nav": avg_nav, "added": str(datetime.now().date())}
        )
        self.save_holdings(holdings)

    def get_nav(self, scheme_code: str) -> Optional[float]:
        try:
            quote = self.mf.get_scheme_quote(scheme_code)
            return float(quote["nav"]) if quote else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("NAV fetch failed for %s: %s", scheme_code, exc)
            return None

    def portfolio_snapshot(self) -> dict:
        holdings = self.load_holdings()
        rows, total_value, total_cost = [], 0.0, 0.0
        for h in holdings:
            nav = self.get_nav(h["scheme_code"]) or h["avg_nav"]
            value = nav * h["units"]
            cost = h["avg_nav"] * h["units"]
            total_value += value
            total_cost += cost
            rows.append(
                {
                    "scheme_code": h["scheme_code"],
                    "units": h["units"],
                    "avg_nav": h["avg_nav"],
                    "current_nav": nav,
                    "value": round(value, 2),
                    "pnl_pct": round((nav / h["avg_nav"] - 1) * 100, 2) if h["avg_nav"] else 0,
                }
            )
        return {
            "holdings": rows,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl_pct": round((total_value / total_cost - 1) * 100, 2) if total_cost else 0,
        }

    # ------------------------------------------------------------------
    # LLM advisory
    # ------------------------------------------------------------------

    def recommend(self, monthly_investable: float, risk_profile: str = "moderate") -> dict:
        """LLM picks an allocation across the fund universe given goals + current holdings."""
        universe_data = []
        for code, name in DEFAULT_FUND_UNIVERSE.items():
            nav = self.get_nav(code)
            universe_data.append(f"- {name} (code {code}), latest NAV: {nav}")
        snapshot = self.portfolio_snapshot()

        system = (
            "You are a SEBI-aware mutual fund advisor for an Indian retail investor. "
            "Recommend a simple monthly SIP allocation across the given fund universe. "
            "Principles: index funds as core, diversify across cap sizes per risk profile, "
            "liquid fund for any short-term parking, keep it to 2-4 funds maximum. "
            "Respond with JSON: "
            '{"allocations": [{"scheme_code": "...", "fund_name": "...", "monthly_amount": <int>, '
            '"reasoning": "..."}], "summary": "<2-3 sentences>"}'
        )
        user = (
            f"Monthly investable amount: Rs.{monthly_investable}\n"
            f"Risk profile: {risk_profile}\n"
            f"Current MF holdings: {snapshot['holdings'] or 'none'}\n\n"
            f"Fund universe:\n" + "\n".join(universe_data)
        )
        try:
            return self.llm.chat_json(system, user)
        except Exception as exc:  # noqa: BLE001
            logger.error("MF recommendation failed: %s", exc)
            return {"allocations": [], "summary": f"Recommendation unavailable: {exc}"}
