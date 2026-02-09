from dataclasses import dataclass
from typing import Optional

from src.backtester.core.trade import TradeDirection, OrderType


@dataclass
class Order:
    """Simple order emitted by strategies — engine executes realistically."""

    direction: TradeDirection
    order_type: OrderType = "market"
    limit_price: Optional[float] = None  # For limit orders; None = market
    notional: float = 0.0  # $ exposure desired
    asset: str = ""
    timeframe: str = ""
    strategy_name: str = ""
