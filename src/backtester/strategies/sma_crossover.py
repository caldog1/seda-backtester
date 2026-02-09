"""Simple SMA crossover baseline strategy.

Fast SMA crosses above slow SMA → long entry (and vice versa for shorts).
On opposite crossover → exit current position and reverse (classic reversal system).
Included as an easy-to-understand example, sanity check, and optimization demo.

Features:
- Configurable fast/slow periods and optional protective SL/TP
- Long-only, short-only, or both directions (reversal mode)
- Uses PositionSizer for realistic exposure
- Fully compatible with multi-timeframe / multi-asset runs
- Direct param passing in __init__ for quick examples
- Params dataclass + Meta for Optuna optimization
- Pure signal logic — no execution details (plug-and-play with centralised engine)
- Classic reversal behaviour: exit + reverse on opposite crossover
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from src.backtester.strategies.base import Strategy, ExitDecision
from src.backtester.sizers.sizers import FixedNotionalSizer, PositionSizer
from src.backtester.core.trade import Trade, TradeDirection
from src.backtester.core.order import Order
from src.backtester.core.execution import OrderType


@dataclass
class SMACrossoverParams:
    """Parameter container for SMA Crossover strategy (enables Optuna optimization)."""

    fast_period: int = 50
    slow_period: int = 200
    sl_pct: Optional[float] = 2.0  # Protective stop-loss % (None = disabled)
    tp_pct: Optional[float] = 6.0  # Protective take-profit % (None = disabled)
    long_only: bool = True
    short_only: bool = False

    class Meta:
        """Optuna search space definition."""

        ranges = {
            "fast_period": (10, 100),
            "slow_period": (100, 300),
            "sl_pct": (0.5, 5.0),
            "tp_pct": (2.0, 15.0),
        }
        int_params = ["fast_period", "slow_period"]
        categoricals = {
            "long_only": [True, False],
            "short_only": [True, False],
        }


class SMACrossoverStrategy(Strategy):
    """Classic dual SMA crossover strategy — pure signal logic with reversal exits."""

    Params = SMACrossoverParams

    def __init__(
        self,
        name: str = "SMA Crossover",
        params: Optional[SMACrossoverParams] = None,
        sizer: Optional[PositionSizer] = None,
        allow_pyramiding: bool = False,
        entry_order_type: OrderType = "limit",
        exit_order_type: OrderType = "market",
        **kwargs,
    ) -> None:
        """
        Initialize the strategy.

        Supports two styles:
        - Direct parameters (quickstarts/examples): pass fast_period, slow_period, etc.
        - Pre-built Params (optimization): pass a SMACrossoverParams instance via 'params'.

        If both are provided, 'params' takes precedence.
        """
        if params is None:
            # Fall back to direct kwargs or defaults
            params = SMACrossoverParams(
                fast_period=kwargs.get("fast_period", 50),
                slow_period=kwargs.get("slow_period", 200),
                sl_pct=kwargs.get("sl_pct", 2.0),
                tp_pct=kwargs.get("tp_pct", 6.0),
                long_only=kwargs.get("long_only", True),
                short_only=kwargs.get("short_only", False),
            )

        self.params = params

        super().__init__(
            name=name,
            sizer=sizer or FixedNotionalSizer(notional=100_000),  # sensible default if None
            allow_pyramiding=allow_pyramiding,
            entry_order_type=entry_order_type,
            exit_order_type=exit_order_type,
        )

    def _get_sma_values(
        self,
        arrays: dict,
        idx: int,
    ) -> Tuple[float | None, float | None, float | None, float | None]:
        """Helper to compute current/previous fast & slow SMA values."""
        if idx < self.params.slow_period:
            return None, None, None, None

        closes = pd.Series(arrays["Close"])

        fast_series = closes.rolling(self.params.fast_period).mean()
        slow_series = closes.rolling(self.params.slow_period).mean()

        fast = fast_series.iloc[idx]
        slow = slow_series.iloc[idx]
        prev_fast = fast_series.iloc[idx - 1]
        prev_slow = slow_series.iloc[idx - 1]

        if pd.isna(fast) or pd.isna(slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return None, None, None, None

        return fast, slow, prev_fast, prev_slow

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
        """Emit entry Order on SMA crossover (engine handles sizing/execution)."""
        fast, slow, prev_fast, prev_slow = self._get_sma_values(arrays, idx)
        if fast is None:
            return []

        direction: Optional[TradeDirection] = None

        cross_up = prev_fast <= prev_slow and fast > slow
        cross_down = prev_fast >= prev_slow and fast < slow

        if cross_up and not self.params.short_only:
            direction = TradeDirection.LONG
        elif cross_down and not self.params.long_only:
            direction = TradeDirection.SHORT

        if direction is None:
            return []

        # Engine enforces pyramiding — we just emit the desired direction
        # If opposite position exists, the engine will have closed it first (via check_exit)
        return [
            Order(
                direction=direction,
                order_type=self.entry_order_type,
                notional=0.0,  # Let sizer decide
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
        """
        Custom exit logic: exit on opposite SMA crossover (reversal signal).
        Engine will also apply automatic intrabar SL/TP if sl_pct/tp_pct provided.
        """
        fast, slow, prev_fast, prev_slow = self._get_sma_values(arrays, idx)
        if fast is None:
            return ExitDecision(False, 0.0, 0.0, self.exit_order_type, 0.0)

        # Detect opposite crossover this bar
        reverse_signal = False
        if trade.direction == TradeDirection.LONG:
            if prev_fast >= prev_slow and fast < slow:  # cross down
                reverse_signal = True
        else:  # SHORT
            if prev_fast <= prev_slow and fast > slow:  # cross up
                reverse_signal = True

        if reverse_signal:
            intended_price = arrays["Close"][idx]  # Signal exit at close (conservative)
            return ExitDecision(
                should_close=True,
                intended_price=intended_price,
                actual_price=intended_price,  # Engine will apply slippage if market order
                order_type=self.exit_order_type,
                fee_rate=0.0,  # Engine fills fee_rate
            )

        return ExitDecision(False, 0.0, 0.0, self.exit_order_type, 0.0)
