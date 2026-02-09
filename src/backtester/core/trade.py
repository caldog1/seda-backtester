"""
Represents a single trade with full lifecycle: entry/exit, slippage, fees,
PnL calculation, and metadata. Immutable where possible for safety.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Optional, Literal

OrderType = Literal["market", "limit"]


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    OPEN = "Open"
    CLOSED = "Closed"


class Trade:
    def __init__(self) -> None:
        self.entry_price: float = 0.0
        self.direction: TradeDirection = TradeDirection.LONG  # placeholder
        self.stoploss: Optional[float] = None
        self.takeprofit: Optional[float] = None
        self.exit_price: float = 0.0
        self.quantity: float = 0.0
        self.trade_value: float = 0.0  # notional $ exposure
        self.asset: Optional[str] = None
        self.timeframe: Optional[str] = None

        self.entry_time: Optional[dt.datetime] = None
        self.exit_time: Optional[dt.datetime] = None

        self.pnl: float = 0.0
        self.fee: float = 0.0  # total round-trip fee in $

        self.entry_fee_rate: float = 0.0
        self.exit_fee_rate: float = 0.0

        self.entry_order_type: Optional[OrderType] = None
        self.exit_order_type: Optional[OrderType] = None

        self.status: TradeStatus = TradeStatus.OPEN
        self.duration: Optional[dt.timedelta] = None

        self.strategy_name: str = ""
        self.intended_entry_price: float = 0.0
        self.entry_slippage_bps: float = 0.0
        self.entry_slippage_dollar: float = 0.0
        self.entry_fee_dollar: float = 0.0

        self.intended_exit_price: Optional[float] = None
        self.exit_slippage_bps: float = 0.0
        self.exit_slippage_dollar: float = 0.0
        self.exit_fee_dollar: float = 0.0

    def open_trade(
        self,
        direction: str | TradeDirection,
        entry_price: float,
        entry_time: dt.datetime,
        *,
        quantity: Optional[float] = None,
        notional_value: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        entry_order_type: OrderType = "market",
        entry_fee_rate: float = 0.0,
    ) -> None:
        """Open a new trade with comprehensive validation and SL/TP setup."""
        # Normalize direction
        if isinstance(direction, str):
            direction = TradeDirection[direction.upper()]
        self.direction = direction

        self.entry_price = entry_price
        self.entry_time = entry_time
        self.entry_order_type = entry_order_type
        self.entry_fee_rate = entry_fee_rate
        self.intended_entry_price = entry_price  # for slippage tracking

        # Position sizing — exactly one of quantity or notional_value
        if (quantity is None) == (notional_value is None):
            raise ValueError("Must provide exactly one of quantity or notional_value")

        if entry_price <= 0:
            raise ValueError("entry_price must be > 0")

        if notional_value is not None:
            if notional_value <= 0:
                raise ValueError("notional_value must be > 0")
            self.quantity = notional_value / entry_price
            self.trade_value = notional_value
        else:
            if quantity <= 0:
                raise ValueError("quantity must be > 0")
            self.quantity = quantity
            self.trade_value = quantity * entry_price

        # SL/TP handling
        if sl is not None and sl_pct is not None:
            raise ValueError("Cannot provide both absolute sl and sl_pct")
        if tp is not None and tp_pct is not None:
            raise ValueError("Cannot provide both absolute tp and tp_pct")

        if sl is not None:
            self.stoploss = sl
        elif sl_pct is not None:
            self.stoploss = entry_price * (
                1 - sl_pct / 100
                if direction is TradeDirection.LONG
                else 1 + sl_pct / 100
            )

        if tp is not None:
            self.takeprofit = tp
        elif tp_pct is not None:
            self.takeprofit = entry_price * (
                1 + tp_pct / 100
                if direction is TradeDirection.LONG
                else 1 - tp_pct / 100
            )

        # Reset exit fields
        self.exit_price = 0.0
        self.exit_time = None
        self.exit_order_type = None
        self.exit_fee_rate = 0.0
        self.pnl = 0.0
        self.fee = 0.0
        self.duration = None
        self.status = TradeStatus.OPEN

    def _calculate_fee(self) -> None:
        """Calculate round-trip fee using original notional."""
        self.entry_fee_dollar = self.trade_value * self.entry_fee_rate
        self.exit_fee_dollar = self.trade_value * self.exit_fee_rate
        self.fee = self.entry_fee_dollar + self.exit_fee_dollar

    def _calculate_pnl(self) -> None:
        """Calculate gross PnL minus fees."""
        if self.direction is TradeDirection.LONG:
            gross = (self.exit_price - self.entry_price) * self.quantity
        else:
            gross = (self.entry_price - self.exit_price) * self.quantity
        self.pnl = gross - self.fee

    def close_trade(
        self,
        exit_price: float,
        exit_time: dt.datetime,
        *,
        exit_order_type: OrderType = "market",
        exit_fee_rate: float = 0.0,
    ) -> None:
        """Close the trade and finalize PnL (exit_price assumed slippage-adjusted)."""
        if self.status is not TradeStatus.OPEN:
            raise ValueError("Trade is not open")

        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_order_type = exit_order_type
        self.exit_fee_rate = exit_fee_rate

        self.duration = exit_time - self.entry_time
        self._calculate_fee()
        self._calculate_pnl()
        self.status = TradeStatus.CLOSED

    def __repr__(self) -> str:
        return (
            f"<Trade {self.direction.value} {self.quantity:.4f} @ {self.entry_price:.2f} "
            f"PnL=${self.pnl:+.2f} status={self.status.value}>"
        )
