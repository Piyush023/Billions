"""Broker factory: returns the configured broker, always falling back to paper."""

import logging

from wealth_platform.brokers.base_broker import BaseBroker
from wealth_platform.brokers.paper_broker import PaperBroker

logger = logging.getLogger("wealth_platform.brokers")


def get_broker(name: str = "paper", starting_cash: float = 15000.0) -> BaseBroker:
    name = (name or "paper").lower()
    if name == "zerodha":
        from wealth_platform.brokers.zerodha_broker import ZerodhaBroker

        broker = ZerodhaBroker()
        if broker.connect():
            return broker
        logger.warning("Zerodha connection failed — falling back to paper broker")
    elif name == "groww":
        from wealth_platform.brokers.groww_broker import GrowwBroker

        broker = GrowwBroker()
        if broker.connect():
            return broker
        logger.warning("Groww connection failed — falling back to paper broker")
    return PaperBroker(starting_cash=starting_cash)
