"""Position sizing strategies for perpetual futures backtesting.

All sizers inherit from PositionSizer and return a notional $ exposure.
Supports fixed notional, fractional equity, risk-based, and fractional Kelly sizing
with optional per-trade caps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class PositionSizer(ABC):
    """Abstract base class for all position sizers."""

    def __init__(self, trade_size_cap: Optional[float] = 100_000.0) -> None:
        """
        Args:
            trade_size_cap: Maximum notional exposure per individual trade ($).
                            Set to None for unlimited.
        """
        self.trade_size_cap = trade_size_cap

    @abstractmethod
    def get_notional(
        self,
        entry_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        current_equity: float,
        stats: Dict[str, float],
    ) -> float:
        """Return the desired notional value ($) for a new trade."""
        raise NotImplementedError

    def _apply_cap(self, notional: float) -> float:
        """Apply the optional per-trade notional cap."""
        if self.trade_size_cap is None:
            return notional
        return min(notional, self.trade_size_cap)


class FixedNotionalSizer(PositionSizer):
    """Fixed dollar notional per trade."""

    def __init__(
        self, notional: float = 10_000.0, trade_size_cap: Optional[float] = 100_000.0
    ) -> None:
        super().__init__(trade_size_cap=trade_size_cap)
        self.notional = notional

    def get_notional(
        self,
        entry_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        current_equity: float,
        stats: Dict[str, float],
    ) -> float:
        return self._apply_cap(self.notional)


class FixedFractionalSizer(PositionSizer):
    """Fixed percentage of current equity (no risk adjustment)."""

    def __init__(
        self, fraction: float = 0.02, trade_size_cap: Optional[float] = 100_000.0
    ) -> None:
        super().__init__(trade_size_cap=trade_size_cap)
        self.fraction = fraction

    def get_notional(
        self,
        entry_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        current_equity: float,
        stats: Dict[str, float],
    ) -> float:
        notional = current_equity * self.fraction
        return self._apply_cap(notional)


class FixedRiskSizer(PositionSizer):
    """Classic risk-based sizing – risk a fixed % of equity on the stop-loss distance."""

    def __init__(
        self, risk_fraction: float = 0.01, trade_size_cap: Optional[float] = 100_000.0
    ) -> None:
        super().__init__(trade_size_cap=trade_size_cap)
        self.risk_fraction = risk_fraction

    def get_notional(
        self,
        entry_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        current_equity: float,
        stats: Dict[str, float],
    ) -> float:
        if sl_price is None:
            return 0.0
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit <= 0:
            return 0.0

        risk_amount = current_equity * self.risk_fraction
        notional = (risk_amount * entry_price) / risk_per_unit
        return self._apply_cap(max(0.0, notional))


class KellyRiskSizer(PositionSizer):
    """Fractional Kelly criterion sizing with fallback for low sample size."""

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        min_trades: int = 30,
        fallback_notional: float = 1_000.0,
        trade_size_cap: Optional[float] = 100_000.0,
    ) -> None:
        super().__init__(trade_size_cap=trade_size_cap)
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        self.fallback_notional = fallback_notional

    def get_notional(
        self,
        entry_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        current_equity: float,
        stats: Dict[str, float],
    ) -> float:
        total = stats.get("total_trades", 0)
        if total < self.min_trades or sl_price is None:
            return self._apply_cap(self.fallback_notional)

        wins = stats.get("wins", 0)
        win_rate = wins / total if total > 0 else 0.0
        avg_win = stats.get("total_win_pnl", 0.0) / wins if wins > 0 else 0.0
        losses = total - wins
        avg_loss_abs = -stats.get("total_loss_pnl", 0.0) / losses if losses > 0 else 0.0

        if avg_loss_abs <= 0:
            return self._apply_cap(self.fallback_notional)

        payoff_ratio = avg_win / avg_loss_abs
        kelly_f = win_rate - (1 - win_rate) / payoff_ratio
        kelly_f = max(0.0, min(1.0, kelly_f))

        fraction = kelly_f * self.kelly_fraction

        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit <= 0:
            return self._apply_cap(self.fallback_notional)

        risk_amount = current_equity * fraction
        notional = (risk_amount * entry_price) / risk_per_unit
        return self._apply_cap(max(notional, self.fallback_notional))
