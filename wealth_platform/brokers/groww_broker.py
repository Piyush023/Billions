"""Groww adapter using the official `growwapi` SDK.

Groww's trading API is FREE (no monthly API fee, unlike Kite Connect),
which makes it the recommended live broker for the free-tier setup.

Auth: Groww issues an API key plus a secret. Depending on how the key pair
was generated, the secret is either a base32 TOTP seed or a plain secret used
in Groww's key+secret (checksum) flow. connect() detects which one you have
and uses the right flow automatically. Set env vars:
  GROWW_API_KEY, GROWW_API_SECRET

Docs: https://groww.in/trade-api/docs
"""

import logging
import os
from typing import Dict, Optional

from wealth_platform.brokers.base_broker import BaseBroker, Funds, OrderResult, Position, Quote

logger = logging.getLogger("wealth_platform.brokers.groww")


class GrowwBroker(BaseBroker):
    name = "groww"
    supports_mutual_funds = True  # Groww is MF-first; holdings visible via API

    def __init__(self):
        self.client = None

    def connect(self) -> bool:
        try:
            from growwapi import GrowwAPI
        except ImportError:
            logger.error("growwapi not installed: pip install growwapi")
            return False
        api_key = os.getenv("GROWW_API_KEY")
        api_secret = os.getenv("GROWW_API_SECRET")
        if not api_key:
            logger.error("GROWW_API_KEY not set")
            return False
        try:
            access_token = self._get_access_token(GrowwAPI, api_key, api_secret)
            self.client = GrowwAPI(access_token)
            logger.info("Groww connected")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww auth failed: %s", exc)
            return False

    @staticmethod
    def _get_access_token(GrowwAPI, api_key: str, api_secret: str) -> str:
        if not api_secret:
            return api_key  # direct access-token mode

        def _is_base32(s: str) -> bool:
            import re

            return bool(re.fullmatch(r"[A-Z2-7]+=*", s.strip().upper().replace(" ", "")))

        if _is_base32(api_secret):
            # TOTP-seed flavoured secret
            import pyotp

            totp = pyotp.TOTP(api_secret).now()
            return GrowwAPI.get_access_token(api_key=api_key, totp=totp)
        # Plain key+secret (checksum) flavoured secret — SDK signature has
        # varied across versions, so try the known parameter names.
        try:
            return GrowwAPI.get_access_token(api_key=api_key, secret=api_secret)
        except TypeError:
            return GrowwAPI.get_access_token(api_key, api_secret)

    def get_quote(self, symbol: str) -> Quote:
        quote = self.client.get_quote(
            exchange=self.client.EXCHANGE_NSE,
            segment=self.client.SEGMENT_CASH,
            trading_symbol=symbol,
        )
        return Quote(symbol=symbol, last_price=quote["last_price"])

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: str = "CNC",
    ) -> OrderResult:
        try:
            response = self.client.place_order(
                trading_symbol=symbol,
                quantity=quantity,
                validity=self.client.VALIDITY_DAY,
                exchange=self.client.EXCHANGE_NSE,
                segment=self.client.SEGMENT_CASH,
                product=self.client.PRODUCT_CNC if product == "CNC" else self.client.PRODUCT_MIS,
                order_type=self.client.ORDER_TYPE_MARKET if order_type == "MARKET" else self.client.ORDER_TYPE_LIMIT,
                transaction_type=(
                    self.client.TRANSACTION_TYPE_BUY if side == "BUY" else self.client.TRANSACTION_TYPE_SELL
                ),
                price=price or 0,
            )
            return OrderResult(success=True, order_id=str(response.get("groww_order_id")), message="Order placed")
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww order failed: %s", exc)
            return OrderResult(success=False, message=str(exc))

    def get_positions(self) -> Dict[str, Position]:
        result = {}
        try:
            positions = self.client.get_positions_for_user(segment=self.client.SEGMENT_CASH)
            for item in positions.get("positions", []):
                qty = item.get("quantity", 0)
                if qty:
                    symbol = item.get("trading_symbol")
                    result[symbol] = Position(
                        symbol=symbol,
                        quantity=qty,
                        average_price=item.get("net_price", 0.0),
                        last_price=self.get_quote(symbol).last_price,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww positions fetch failed: %s", exc)
        return result

    def get_holdings(self):
        holdings = []
        try:
            data = self.client.get_holdings_for_user()
            for item in data.get("holdings", []):
                holdings.append(
                    Position(
                        symbol=item.get("trading_symbol"),
                        quantity=item.get("quantity", 0),
                        average_price=item.get("average_price", 0.0),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww holdings fetch failed: %s", exc)
        return holdings

    def get_funds(self) -> Funds:
        try:
            margin = self.client.get_available_margin_details()
            return Funds(available_cash=margin.get("clear_cash", 0.0))
        except Exception as exc:  # noqa: BLE001
            logger.error("Groww funds fetch failed: %s", exc)
            return Funds(available_cash=0.0)
