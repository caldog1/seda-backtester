"""Abstract base class for all trading strategies.

Provides shared infrastructure:
- Position sizing integration
- Ongoing/completed trade tracking per asset/timeframe
- Centralized exit handling and execution recording (slippage/fees)
- Per-strategy performance and leverage impact counters
- Pyramiding control

Concrete strategies implement entry logic and exit checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, List, Tuple, NamedTuple

import datetime as dt

from src.backtester.core.execution import OrderType
from src.backtester.core.order import Order
from src.backtester.core.trade import Trade
from src.backtester.sizers.sizers import PositionSizer


class ExitDecision(NamedTuple):
    """Return type for check_exit — makes intent clear."""

    should_close: bool
    intended_price: float
    actual_price: float
    order_type: OrderType
    fee_rate: float


class Strategy(ABC):
    """Base class for all strategies — extend and implement entry/exit logic."""

    def __init__(
        self,
        name: str,
        sizer: PositionSizer,
        allow_pyramiding: bool = True,
        entry_order_type: OrderType = "limit",
        exit_order_type: OrderType = "limit",
    ) -> None:
        if not name.strip():
            raise ValueError("Strategy name must be non-empty and unique.")

        self.name = name
        self.sizer = sizer
        self.allow_pyramiding = allow_pyramiding
        self.entry_order_type = entry_order_type
        self.exit_order_type = exit_order_type

        # Trade tracking: (asset, timeframe) → list of trades
        self.ongoing_trades: DefaultDict[Tuple[str, str], List[Trade]] = defaultdict(
            list
        )
        self.completed_trades: DefaultDict[Tuple[str, str], List[Trade]] = defaultdict(
            list
        )

        # Per-strategy stats (for reporting/optimization)
        self.trade_stats = {
            "total_trades": 0,
            "wins": 0,
            "total_win_pnl": 0.0,
            "total_loss_pnl": 0.0,
        }

        # Leverage/pyramiding impact counters
        self.total_entry_signals = 0
        self.leverage_fully_prevented = 0
        self.leverage_partially_reduced = 0
        self.pyramiding_prevented = 0
        self.post_liquidation_prevented = 0

        self.entry_times: List[dt.datetime] = []

    def _update_trade_stats(self, trade: Trade) -> None:
        """Update running win/loss stats."""
        self.trade_stats["total_trades"] += 1
        if trade.pnl > 0:
            self.trade_stats["wins"] += 1
            self.trade_stats["total_win_pnl"] += trade.pnl
        elif trade.pnl < 0:
            self.trade_stats["total_loss_pnl"] += trade.pnl

    @abstractmethod
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
        """Return list of Orders to execute this bar. Engine handles filling."""
        return []

    def _record_entry_execution(
        self,
        trade: Trade,
        intended_price: float,
        actual_price: float,
        fee_rate: float,
        notional: float,
    ) -> None:
        """Centralized entry execution recording (slippage + fees)."""
        trade.strategy_name = self.name
        trade.intended_entry_price = intended_price
        trade.entry_fee_rate = fee_rate
        trade.entry_fee_dollar = notional * fee_rate

        # Adverse slippage only (positive bps/dollar)
        if trade.direction.value == "LONG":
            slippage_bps = (actual_price - intended_price) / intended_price * 10_000
            dollar_impact = (actual_price - intended_price) * trade.quantity
        else:
            slippage_bps = (intended_price - actual_price) / intended_price * 10_000
            dollar_impact = (intended_price - actual_price) * trade.quantity

        trade.entry_slippage_bps = max(0.0, slippage_bps)
        trade.entry_slippage_dollar = dollar_impact

    def _record_exit_execution(
        self,
        trade: Trade,
        intended_price: float,
        actual_price: float,
        fee_rate: float,
    ) -> None:
        """Centralized exit execution recording (slippage + fees)."""
        trade.intended_exit_price = intended_price
        trade.exit_fee_rate = fee_rate
        trade.exit_fee_dollar = trade.trade_value * fee_rate

        # Adverse slippage only
        if trade.direction.value == "LONG":
            slippage_bps = (intended_price - actual_price) / intended_price * 10_000
            dollar_impact = (intended_price - actual_price) * trade.quantity
        else:
            slippage_bps = (actual_price - intended_price) / intended_price * 10_000
            dollar_impact = (actual_price - intended_price) * trade.quantity

        trade.exit_slippage_bps = max(0.0, slippage_bps)
        trade.exit_slippage_dollar = dollar_impact

    @abstractmethod
    def check_exit(
        self,
        idx: int,
        trade: Trade,
        arrays: dict,
        asset: str,
        timeframe: str,
        backtester: "Backtester",
    ) -> ExitDecision:
        """Return exit decision for a single open trade.

        Concrete strategies implement specific logic (SL/TP, trailing, signals, etc.).
        Return ExitDecision with should_close=False if no exit.
        """
        return ExitDecision(False, 0.0, 0.0, "market", 0.0)

    def handle_exits(
        self,
        key: Tuple[str, str],
        idx: int,
        arrays: dict,
        asset: str,
        timeframe: str,
        backtester: "Backtester",
    ) -> float:
        """Shared exit processing — called by simulation loop."""
        trades_list = self.ongoing_trades[key]
        pnl_this_bar = 0.0
        to_close: List[Trade] = []

        current_price = arrays["Close"][idx]
        current_time = arrays["Close Time"][idx]

        for trade in trades_list[:]:
            decision = self.check_exit(idx, trade, arrays, asset, timeframe, backtester)
            if not decision.should_close:
                continue

            # Record execution
            self._record_exit_execution(
                trade=trade,
                intended_price=decision.intended_price,
                actual_price=decision.actual_price,
                fee_rate=decision.fee_rate,
            )

            trade.close_trade(
                exit_price=decision.actual_price,
                exit_time=current_time,
                exit_order_type=decision.order_type,
                exit_fee_rate=decision.fee_rate,
            )

            to_close.append(trade)
            pnl_this_bar += trade.pnl
            self._update_trade_stats(trade)

        # Cleanup
        for trade in to_close:
            trades_list.remove(trade)
            self.completed_trades[key].append(trade)

        return pnl_this_bar

    def __repr__(self) -> str:
        return f"<Strategy {self.name} ongoing={sum(len(v) for v in self.ongoing_trades.values())}>"
