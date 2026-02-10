"""
RSI Mean Reversion Strategy

A classic overbought/oversold strategy:
- Enter long when RSI(14) crosses below 30
- Enter short when RSI(14) crosses above 70
- Exit on opposite signal (reversal) or fixed SL/TP

Serves as a template for new strategy development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from backtester.strategies.base import Strategy, ExitDecision
from backtester.sizers.sizers import PositionSizer
from backtester.core.trade import Trade, TradeDirection
from backtester.core.order import Order
from backtester.core.execution import OrderType


@dataclass
class RSIMeanReversionParams:
    """Parameter container — enables future Optuna optimisation."""

    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    sl_pct: Optional[float] = 3.0
    tp_pct: Optional[float] = 9.0
    long_only: bool = False
    short_only: bool = False

    class Meta:
        ranges = {
            "rsi_period": (7, 21),
            "oversold": (20.0, 40.0),
            "overbought": (60.0, 80.0),
            "sl_pct": (1.0, 6.0),
            "tp_pct": (4.0, 15.0),
        }
        int_params = ["rsi_period"]


class RSIMeanReversionStrategy(Strategy):
    """RSI-based mean reversion with reversal exits."""

    Params = RSIMeanReversionParams

    def __init__(
        self,
        name: str = "RSI Mean Reversion",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        sl_pct: Optional[float] = 3.0,
        tp_pct: Optional[float] = 9.0,
        long_only: bool = False,
        short_only: bool = False,
        sizer: Optional[PositionSizer] = None,
        allow_pyramiding: bool = True,
        entry_order_type: OrderType = "limit",
        exit_order_type: OrderType = "market",
    ) -> None:
        params = RSIMeanReversionParams(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            long_only=long_only,
            short_only=short_only,
        )
        super().__init__(
            name=name,
            sizer=sizer,
            allow_pyramiding=allow_pyramiding,
            entry_order_type=entry_order_type,
            exit_order_type=exit_order_type,
        )
        self.params = params

    def _rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Vectorised RSI calculation."""
        delta = np.diff(close)
        up = np.maximum(delta, 0)
        down = np.abs(np.minimum(delta, 0))

        roll_up = np.convolve(up, np.ones(period), mode="valid")
        roll_down = np.convolve(down, np.ones(period), mode="valid")

        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        # Pad front with nan
        return np.concatenate([np.full(period - 1, np.nan), rsi])

    def _get_rsi_values(
        self, arrays: dict, idx: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return current and previous RSI values."""
        if idx < self.params.rsi_period:
            return None, None

        close = arrays["Close"][: idx + 1]
        rsi_series = self._rsi(close, self.params.rsi_period)

        current = rsi_series[-1]
        previous = rsi_series[-2] if len(rsi_series) >= 2 else None
        return current, previous

    def handle_entries(
        self,
        key: Tuple[str, str],
        idx: int,
        arrays: dict,
        timeframe: str,
        asset: str,
        current_equity: float,
        max_additional_notional: float,
        backtester: "Backtester",
    ) -> List[Order]:
        current_rsi, prev_rsi = self._get_rsi_values(arrays, idx)
        if current_rsi is None:
            return []

        direction: Optional[TradeDirection] = None

        # Oversold → long entry
        if prev_rsi is not None and prev_rsi >= self.params.oversold > current_rsi:
            if not self.params.short_only:
                direction = TradeDirection.LONG

        # Overbought → short entry
        elif prev_rsi is not None and prev_rsi <= self.params.overbought < current_rsi:
            if not self.params.long_only:
                direction = TradeDirection.SHORT

        if direction is None:
            return []

        return [
            Order(
                direction=direction,
                order_type=self.entry_order_type,
                notional=0.0,  # Sizer decides
                asset=asset,
                timeframe=timeframe,
                strategy_name=self.name,
            )
        ]

    def check_exit(
        self,
        idx: int,
        trade: Trade,
        arrays: dict,
        asset: str,
        timeframe: str,
        backtester: "Backtester",
    ) -> ExitDecision:
        current_rsi, _ = self._get_rsi_values(arrays, idx)
        if current_rsi is None:
            return ExitDecision(False, 0.0, 0.0, self.exit_order_type, 0.0)

        reverse = False
        if (
            trade.direction == TradeDirection.LONG
            and current_rsi >= self.params.overbought
        ):
            reverse = True
        elif (
            trade.direction == TradeDirection.SHORT
            and current_rsi <= self.params.oversold
        ):
            reverse = True

        if reverse:
            intended_price = arrays["Close"][idx]
            return ExitDecision(
                should_close=True,
                intended_price=intended_price,
                actual_price=intended_price,
                order_type=self.exit_order_type,
                fee_rate=0.0,
            )

        return ExitDecision(False, 0.0, 0.0, self.exit_order_type, 0.0)
