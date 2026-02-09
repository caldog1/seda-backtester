"""
Models realistic order execution: maker/taker fees and volume-based slippage.
Provides pluggable FeeModel and SlippageModel abstractions.
"""

from abc import ABC, abstractmethod
from typing import Literal

OrderType = Literal["market", "limit"]


class FeeModel(ABC):
    @abstractmethod
    def get_fee_rate(self, order_type: OrderType) -> float:
        """Return fee rate (decimal) for the order type."""
        pass


class DefaultFeeModel(FeeModel):
    """Standard fee schedule.

    Taker (market): 4 bps
    Maker (limit): 2 bps
    Easily configurable via subclassing or instantiation args.
    """

    def __init__(self, taker_rate: float = 0.0004, maker_rate: float = 0.0002) -> None:
        self.taker_rate = taker_rate
        self.maker_rate = maker_rate

    def get_fee_rate(self, order_type: OrderType) -> float:
        return self.taker_rate if order_type == "market" else self.maker_rate


class SlippageModel(ABC):
    @abstractmethod
    def get_slippage_bps(
        self,
        notional: float,
        bar_liquidity: float,
        asset: str,
        timeframe: str,
        order_type: OrderType,
    ) -> float:
        """Return adverse slippage in basis points."""
        pass

    def apply_slippage(self, price: float, direction: str, bps: float) -> float:
        """Apply adverse slippage to price."""
        factor = bps / 10_000
        if direction.upper() == "LONG":
            return price * (1 + factor)
        return price * (1 - factor)


class HybridSlippageModel(SlippageModel):
    """Institutional-grade slippage: fixed noise + volume-share component."""

    def __init__(
        self,
        fixed_market_bps: float = 1.0,
        bps_per_full_bar: float = 50.0,
    ) -> None:
        self.fixed_market_bps = fixed_market_bps
        self.bps_per_full_bar = bps_per_full_bar

    def get_slippage_bps(
        self,
        notional: float,
        bar_liquidity: float,
        asset: str,
        timeframe: str,
        order_type: OrderType,
    ) -> float:
        if order_type == "limit":
            return 0.0

        bar_liquidity = max(bar_liquidity, 1.0)
        volume_share = notional / bar_liquidity
        variable_bps = volume_share * self.bps_per_full_bar
        total_bps = self.fixed_market_bps + variable_bps

        if volume_share > 0.1:
            print(
                f"\nWARNING: High slippage risk — {asset} {timeframe}\n"
                f"  Notional ${notional:,.0f} = {volume_share:.1%} of bar liquidity\n"
                f"  Estimated slippage: {total_bps:.1f} bps"
            )

        return total_bps
