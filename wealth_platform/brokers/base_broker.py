"""Common broker interface. Every broker (Zerodha, Groww, paper) implements this,
so the orchestrator never cares which broker is active."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Quote:
    symbol: str
    last_price: float
    exchange: str = "NSE"


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    filled_price: Optional[float] = None


@dataclass
class Position:
    symbol: str
    quantity: int
    average_price: float
    last_price: float = 0.0

    @property
    def pnl(self) -> float:
        return (self.last_price - self.average_price) * self.quantity

    @property
    def value(self) -> float:
        return self.last_price * self.quantity


@dataclass
class Funds:
    available_cash: float
    used_margin: float = 0.0


class BaseBroker(ABC):
    """Interface all broker adapters implement."""

    name = "base"
    supports_mutual_funds = False

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate / validate session. Returns True when ready to trade."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,  # "BUY" | "SELL"
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: str = "CNC",  # CNC = delivery, MIS = intraday
    ) -> OrderResult:
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        ...

    @abstractmethod
    def get_funds(self) -> Funds:
        ...

    def get_holdings(self) -> List[Position]:
        """Long-term holdings (delivery). Defaults to positions if not distinct."""
        return list(self.get_positions().values())
