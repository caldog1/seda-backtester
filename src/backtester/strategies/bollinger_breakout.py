"""
Bollinger Bands Breakout Strategy

- Enter long on close above upper band
- Enter short on close below lower band
- Exit on opposite breakout (reversal) or midline cross

Excellent template for volatility-based strategies.
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
class BollingerBreakoutParams:
    period: int = 20
    std_dev: float = 2.0
    sl_pct: Optional[float] = 4.0
    tp_pct: Optional[float] = 12.0

    class Meta:
        ranges = {
            "period": (10, 50),
            "std_dev": (1.5, 3.0),
            "sl_pct": (2.0, 8.0),
            "tp_pct": (6.0, 20.0),
        }
        int_params = ["period"]


class BollingerBreakoutStrategy(Strategy):
    Params = BollingerBreakoutParams

    def __init__(
        self,
        name: str = "Bollinger Breakout",
        period: int = 20,
        std_dev: float = 2.0,
        sl_pct: Optional[float] = 4.0,
        tp_pct: Optional[float] = 12.0,
        sizer: Optional[PositionSizer] = None,
        allow_pyramiding: bool = False,
        entry_order_type: OrderType = "market",
        exit_order_type: OrderType = "market",
    ) -> None:
        params = BollingerBreakoutParams(
            period=period, std_dev=std_dev, sl_pct=sl_pct, tp_pct=tp_pct
        )
        super().__init__(
            name=name,
            sizer=sizer,
            allow_pyramiding=allow_pyramiding,
            entry_order_type=entry_order_type,
            exit_order_type=exit_order_type,
        )
        self.params = params

    def _bollinger(
        self, close: np.ndarray, period: int, std_dev: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis = np.convolve(close, np.ones(period) / period, mode="valid")
        dev = std_dev * np.std(
            [close[i : i + period] for i in range(len(close) - period + 1)], axis=1
        )
        upper = basis + dev
        lower = basis - dev
        # Pad front
        pad = period - 1
        basis = np.concatenate([np.full(pad, np.nan), basis])
        upper = np.concatenate([np.full(pad, np.nan), upper])
        lower = np.concatenate([np.full(pad, np.nan), lower])
        return basis, upper, lower

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
        if idx < self.params.period:
            return []

        close = arrays["Close"][: idx + 1]
        _, upper, lower = self._bollinger(
            close, self.params.period, self.params.std_dev
        )

        current_close = close[-1]
        prev_close = close[-2]
        current_upper = upper[-1]
        current_lower = lower[-1]
        prev_upper = upper[-2]
        prev_lower = lower[-2]

        direction: Optional[TradeDirection] = None

        # Breakout above upper band
        if prev_close <= prev_upper and current_close > current_upper:
            direction = TradeDirection.LONG
        # Breakdown below lower band
        elif prev_close >= prev_lower and current_close < current_lower:
            direction = TradeDirection.SHORT

        if direction is None:
            return []

        return [
            Order(
                direction=direction,
                order_type=self.entry_order_type,
                notional=0.0,
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
        if idx < self.params.period:
            return ExitDecision(False, 0.0, 0.0, self.exit_order_type, 0.0)

        close = arrays["Close"][: idx + 1]
        basis, upper, lower = self._bollinger(
            close, self.params.period, self.params.std_dev
        )

        current_close = close[-1]
        current_basis = basis[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]

        reverse = False
        if trade.direction == TradeDirection.LONG and current_close < current_basis:
            reverse = True
        elif trade.direction == TradeDirection.SHORT and current_close > current_basis:
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
