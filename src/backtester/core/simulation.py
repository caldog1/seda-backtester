"""
Core event-driven simulation loop.

This module contains the main simulation engine for SEDA. It advances synchronously
through a unified timeline of all asset/timeframe bar close times, processes exits
first (to free capital), then entries, updates portfolio state (realized equity,
unrealized PnL, leverage, margin ratio), checks for liquidation, and records
comprehensive histories.

All performance metrics are computed vectorized with NumPy for speed and clarity.
Execution (fees + slippage) is fully centralized here — strategies only emit clean
Order objects for entries and optional custom ExitDecision for exits. Automatic
SL/TP handling is provided by the engine if the trade was opened with sl_pct/tp_pct.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtester.core.results import BacktestResults
from src.backtester.core.trade import Trade, TradeDirection, TradeStatus
from src.backtester.core.order import Order  # New minimal order abstraction
from src.backtester.core.execution import OrderType


def _collect_all_trades(backtester: "Backtester") -> List[Trade]:
    """Collect and chronologically sort all completed trades across strategies."""
    all_trades: List[Trade] = []
    for strategy in backtester.strategies:
        for trades in strategy.completed_trades.values():
            all_trades.extend(trades)
    all_trades.sort(key=lambda t: t.exit_time or t.entry_time or dt.datetime.min)
    return all_trades


def _collect_entry_times(backtester: "Backtester") -> List[dt.datetime]:
    """Collect all entry timestamps from completed and ongoing trades only.

    This avoids double-counting that previously occurred when extending
    strategy.entry_times and then adding the same times from trades.
    """
    entry_times: List[dt.datetime] = []
    for strategy in backtester.strategies:
        # Completed trades
        for trades_list in strategy.completed_trades.values():
            for trade in trades_list:
                if trade.entry_time:
                    entry_times.append(trade.entry_time)
        # Ongoing trades (will be empty after final forced close)
        for trades_list in strategy.ongoing_trades.values():
            for trade in trades_list:
                if trade.entry_time:
                    entry_times.append(trade.entry_time)
    entry_times.sort()
    return entry_times


def _compute_trade_metrics(trades: List[Trade]) -> Dict[str, float]:
    """Compute standard trade-level performance metrics."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_trade": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "total_fees": 0.0,
            "avg_duration_hours": 0.0,
            "std_duration_hours": 0.0,
            "min_duration_hours": 0.0,
            "max_duration_hours": 0.0,
            "median_duration_hours": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    total_trades = len(trades)
    pnls = np.array([tr.pnl for tr in trades])
    wins = pnls > 0
    losses = pnls < 0

    num_wins = np.sum(wins)
    num_losses = np.sum(losses)
    win_rate = num_wins / total_trades * 100 if total_trades > 0 else 0.0

    gross_profit = np.sum(pnls[wins])
    gross_loss = np.sum(pnls[losses])
    net_pnl = gross_profit + gross_loss
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
    avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
    avg_trade = net_pnl / total_trades if total_trades > 0 else 0.0

    largest_win = np.max(pnls[wins]) if num_wins > 0 else 0.0
    largest_loss = np.min(pnls[losses]) if num_losses > 0 else 0.0

    total_fees = sum(t.fee for t in trades)

    # Duration statistics (in hours) — robust for np.timedelta64 or pd.Timedelta
    durations_hours = [
        (
            pd.Timedelta(t.duration).total_seconds() / 3600.0
            if t.duration is not None
            else 0.0
        )
        for t in trades
    ]

    if durations_hours:
        durations_arr = np.array(durations_hours)
        avg_duration_hours = float(durations_arr.mean())
        std_duration_hours = (
            float(durations_arr.std(ddof=1)) if len(durations_arr) > 1 else 0.0
        )
        min_duration_hours = float(durations_arr.min())
        max_duration_hours = float(durations_arr.max())
        median_duration_hours = float(np.median(durations_arr))
    else:
        avg_duration_hours = std_duration_hours = min_duration_hours = (
            max_duration_hours
        ) = median_duration_hours = 0.0

    # === Win/loss streak calculation – robust version ===
    # Sort chronologically by exit time (defensive)
    sorted_trades = sorted(
        trades,
        key=lambda t: t.exit_time or dt.datetime.max,
    )

    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_type: Optional[str] = None  # "win" or "loss"

    TOLERANCE = 1e-6  # Handle floating-point near-zero PnL

    for trade in sorted_trades:
        if abs(trade.pnl) <= TOLERANCE:
            # Breakeven – reset streak
            current_streak = 0
            current_type = None
            continue

        if trade.pnl > 0:
            if current_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:  # trade.pnl < 0
            if current_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    metrics = {
        "total_trades": int(total_trades),
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": avg_trade,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "total_fees": total_fees,
        "avg_duration_hours": avg_duration_hours,
        "std_duration_hours": std_duration_hours,
        "min_duration_hours": min_duration_hours,
        "max_duration_hours": max_duration_hours,
        "median_duration_hours": median_duration_hours,
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
    }

    return metrics


def run_simulation(backtester: "Backtester") -> BacktestResults:
    """Main synchronous event-driven simulation loop with centralised execution."""
    results = BacktestResults()

    equity_history: List[Tuple[dt.datetime, float]] = []
    realized_equity_history: List[Tuple[dt.datetime, float]] = []
    open_notional_history: List[Tuple[dt.datetime, float]] = []
    open_trade_history: List[Tuple[dt.datetime, int]] = []
    maintenance_margin_history: List[Tuple[dt.datetime, float]] = []
    margin_ratio_history: List[Tuple[dt.datetime, float]] = []

    current_equity: float = backtester.initial_capital
    last_prices: Dict[Tuple[str, str], float] = {}
    open_notional: float = 0.0

    for current_time in tqdm(backtester.master_time_list, desc="Backtesting"):

        # Identify all series closing at this exact timestamp
        active_series: List[Tuple[Tuple[str, str], int, dict]] = []
        for key in backtester.pointers:
            ptr = backtester.pointers[key]
            arrays = backtester.arrays[key]
            if (
                ptr < len(arrays["Close Time"])
                and arrays["Close Time"][ptr] == current_time
            ):
                active_series.append((key, ptr, arrays))
                last_prices[key] = arrays["Close"][ptr]
                backtester.pointers[key] += 1

        if not active_series:
            continue

        # ---- Mark-to-market for leverage / liquidation ----
        unreal_pnl = 0.0
        open_trade_count = 0
        for strategy in backtester.strategies:
            for key, trades in strategy.ongoing_trades.items():
                open_trade_count += len(trades)
                price = last_prices.get(key)
                if price is None:
                    continue
                for trade in trades:
                    if trade.direction == TradeDirection.LONG:
                        unreal_pnl += (price - trade.entry_price) * trade.quantity
                    else:
                        unreal_pnl += (trade.entry_price - price) * trade.quantity

        mtm_equity = current_equity + unreal_pnl

        # Record histories
        equity_history.append((current_time, mtm_equity))
        realized_equity_history.append((current_time, current_equity))
        open_notional_history.append((current_time, open_notional))
        open_trade_history.append((current_time, open_trade_count))
        maintenance_margin = open_notional * backtester.liq_tracker.mmr_rate
        maintenance_margin_history.append((current_time, maintenance_margin))
        margin_ratio = (
            mtm_equity / maintenance_margin if maintenance_margin > 0 else float("inf")
        )
        margin_ratio_history.append((current_time, margin_ratio))

        # Liquidation check
        all_ongoing = defaultdict(list)
        for strategy in backtester.strategies:
            for k, v in strategy.ongoing_trades.items():
                all_ongoing[k].extend(v)
        backtester.liq_tracker.check_and_handle_liquidation(
            mtm_equity=mtm_equity,
            open_notional=open_notional,
            current_time=current_time,
            current_equity=current_equity,
            ongoing_trades=all_ongoing,
        )

        max_additional_notional = (
            backtester.liq_tracker.get_max_additional_notional(
                current_equity, open_notional
            )
            if not (
                backtester.liq_tracker.liquidated
                and backtester.liq_tracker.mode == "stop"
            )
            else 0.0
        )

        # ---- Exits (custom + automatic SL/TP) ----
        pnl_this_bar = 0.0
        for strategy in backtester.strategies:
            for key, idx, arrays in active_series:
                asset, timeframe = key
                trades = strategy.ongoing_trades[key][:]
                for trade in trades:
                    decision = strategy.check_exit(
                        idx, trade, arrays, asset, timeframe, backtester
                    )

                    intended_price: Optional[float] = (
                        decision.intended_price if decision.should_close else None
                    )

                    # Automatic SL/TP if strategy didn't provide a custom exit
                    if intended_price is None:
                        if trade.stoploss is not None:
                            if (
                                trade.direction == TradeDirection.LONG
                                and arrays["Low"][idx] <= trade.stoploss
                            ) or (
                                trade.direction == TradeDirection.SHORT
                                and arrays["High"][idx] >= trade.stoploss
                            ):
                                intended_price = trade.stoploss
                        if trade.takeprofit is not None and intended_price is None:
                            if (
                                trade.direction == TradeDirection.LONG
                                and arrays["High"][idx] >= trade.takeprofit
                            ) or (
                                trade.direction == TradeDirection.SHORT
                                and arrays["Low"][idx] <= trade.takeprofit
                            ):
                                intended_price = trade.takeprofit

                    if intended_price is not None:
                        order_type: OrderType = (
                            decision.order_type
                            if decision.should_close
                            else strategy.exit_order_type
                        )
                        exit_dir = (
                            TradeDirection.SHORT
                            if trade.direction == TradeDirection.LONG
                            else TradeDirection.LONG
                        )

                        actual_price, fee_rate = backtester.execute_order(
                            intended_price=intended_price,
                            direction=exit_dir.value,
                            notional=trade.trade_value,
                            order_type=order_type,
                            arrays=arrays,
                            idx=idx,
                            asset=asset,
                            timeframe=timeframe,
                        )

                        trade.close_trade(
                            exit_price=actual_price,
                            exit_time=current_time,
                            exit_order_type=order_type,
                            exit_fee_rate=fee_rate,
                        )

                        strategy._record_exit_execution(
                            trade=trade,
                            intended_price=intended_price,
                            actual_price=actual_price,
                            fee_rate=fee_rate,
                        )

                        pnl_this_bar += trade.pnl
                        strategy._update_trade_stats(trade)
                        strategy.ongoing_trades[key].remove(trade)
                        strategy.completed_trades[key].append(trade)
                        open_notional -= trade.trade_value

        current_equity += pnl_this_bar

        # ---- Entries (centralised execution) ----
        remaining_additional = max_additional_notional
        for strategy in backtester.strategies:
            for key, idx, arrays in active_series:
                asset, timeframe = key

                orders: List[Order] = strategy.handle_entries(
                    key=key,
                    idx=idx,
                    arrays=arrays,
                    timeframe=timeframe,
                    asset=asset,
                    current_equity=current_equity,
                    max_additional_notional=remaining_additional,
                    backtester=backtester,
                )

                for order in orders:
                    # Pyramiding check
                    if strategy.ongoing_trades[key] and not strategy.allow_pyramiding:
                        strategy.pyramiding_prevented += 1
                        continue

                    intended_price = (
                        arrays["Close"][idx]
                        if order.order_type == "market"
                        else (order.limit_price or arrays["Close"][idx])
                    )

                    # Optional SL/TP percentages from strategy params (for automatic trade-level SL/TP and risk sizers)
                    sl_pct: Optional[float] = None
                    tp_pct: Optional[float] = None
                    sl_price: Optional[float] = None
                    tp_price: Optional[float] = None

                    if hasattr(strategy, "params"):
                        params = strategy.params
                        sl_pct = getattr(params, "sl_pct", None)
                        tp_pct = getattr(params, "tp_pct", None)

                        # Calculate absolute prices for risk-based sizers (if pct provided)
                        if sl_pct is not None:
                            sl_price = intended_price * (
                                1 - sl_pct / 100
                                if order.direction == TradeDirection.LONG
                                else 1 + sl_pct / 100
                            )
                        if tp_pct is not None:
                            tp_price = intended_price * (
                                1 + tp_pct / 100
                                if order.direction == TradeDirection.LONG
                                else 1 - tp_pct / 100
                            )

                    notional = strategy.sizer.get_notional(
                        entry_price=intended_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        current_equity=current_equity,
                        stats=strategy.trade_stats,
                    )

                    # Leverage cap
                    if notional > remaining_additional:
                        if remaining_additional <= 0:
                            strategy.leverage_fully_prevented += 1
                        else:
                            notional = remaining_additional
                            strategy.leverage_partially_reduced += 1

                    if notional <= 0:
                        continue

                    actual_price, fee_rate = backtester.execute_order(
                        intended_price=intended_price,
                        direction=order.direction.value,
                        notional=notional,
                        order_type=order.order_type,
                        arrays=arrays,
                        idx=idx,
                        asset=asset,
                        timeframe=timeframe,
                    )

                    trade = Trade()
                    trade.open_trade(
                        direction=order.direction,
                        entry_price=actual_price,
                        entry_time=current_time,
                        notional_value=notional,
                        sl_pct=sl_pct,
                        tp_pct=tp_pct,
                        entry_order_type=order.order_type,
                        entry_fee_rate=fee_rate,
                    )
                    trade.asset = asset
                    trade.timeframe = timeframe
                    trade.strategy_name = strategy.name

                    strategy._record_entry_execution(
                        trade=trade,
                        intended_price=intended_price,
                        actual_price=actual_price,
                        fee_rate=fee_rate,
                        notional=notional,
                    )

                    strategy.ongoing_trades[key].append(trade)
                    strategy.total_entry_signals += 1
                    strategy.entry_times.append(current_time)

                    remaining_additional -= notional
                    open_notional += notional

    # ==== FINAL CLOSE: Close all open positions at the end of the backtest ====
    if backtester.master_time_list:
        final_time = pd.Timestamp(backtester.master_time_list[-1])
        final_prices = {
            key: arrays["Close"][-1] for key, arrays in backtester.arrays.items()
        }

        pnl_this_bar = 0.0

        for strategy in backtester.strategies:
            for key, trades in list(strategy.ongoing_trades.items()):
                asset, timeframe = key
                price = final_prices.get(key)
                if price is None:
                    continue

                for trade in trades[:]:
                    actual_price, fee_rate = backtester.execute_order(
                        intended_price=price,
                        direction=(
                            "SHORT"
                            if trade.direction == TradeDirection.LONG
                            else "LONG"
                        ),
                        notional=trade.trade_value,
                        order_type="market",
                        arrays=backtester.arrays[key],
                        idx=-1,
                        asset=asset,
                        timeframe=timeframe,
                    )

                    exit_time = final_time + pd.Timedelta(seconds=1)

                    trade.close_trade(
                        exit_price=actual_price,
                        exit_time=exit_time,
                        exit_order_type="market",
                        exit_fee_rate=fee_rate,
                    )

                    strategy._record_exit_execution(
                        trade=trade,
                        intended_price=price,
                        actual_price=actual_price,
                        fee_rate=fee_rate,
                    )

                    strategy._update_trade_stats(trade)
                    strategy.completed_trades[key].append(trade)
                    strategy.ongoing_trades[key].remove(trade)

                    pnl_this_bar += trade.pnl
                    open_notional -= trade.trade_value

        current_equity += pnl_this_bar

        final_record_time = final_time + pd.Timedelta(seconds=1)
        equity_history.append((final_record_time, current_equity))
        realized_equity_history.append((final_record_time, current_equity))
        open_notional_history.append((final_record_time, open_notional))
        open_trade_history.append((final_record_time, 0))
        maintenance_margin_history.append((final_record_time, 0.0))
        margin_ratio_history.append((final_record_time, float("inf")))

    # ---- Final results assembly ----
    results.equity_history = equity_history
    results.realized_equity_history = realized_equity_history
    results.open_notional_history = open_notional_history
    results.open_trade_history = open_trade_history
    results.maintenance_margin_history = maintenance_margin_history
    results.margin_ratio_history = margin_ratio_history

    results.all_trades = _collect_all_trades(backtester)
    results.trade_pnls = [t.pnl for t in results.all_trades]
    results.entry_times = _collect_entry_times(backtester)

    results.liquidated = backtester.liq_tracker.liquidated
    results.liq_time = backtester.liq_tracker.liq_time

    # Aggregate leverage / pyramiding counters across strategies
    for strategy in backtester.strategies:
        results.total_entry_signals += strategy.total_entry_signals
        results.leverage_fully_prevented += strategy.leverage_fully_prevented
        results.leverage_partially_reduced += strategy.leverage_partially_reduced
        results.pyramiding_prevented += strategy.pyramiding_prevented
        results.post_liquidation_prevented += strategy.post_liquidation_prevented

    results.overall_metrics = _compute_trade_metrics(results.all_trades)

    # Vectorised portfolio metrics (same as original)
    if len(equity_history) > 1:
        times_arr = np.array([t for t, _ in equity_history], dtype="datetime64[ns]")
        equity_arr = np.array([e for _, e in equity_history])

        results.total_return_pct = (
            (equity_arr[-1] - backtester.initial_capital)
            / backtester.initial_capital
            * 100
        )

        days = float((times_arr[-1] - times_arr[0]) / np.timedelta64(1, "D"))
        if days > 0:
            results.cagr_pct = (
                (equity_arr[-1] / backtester.initial_capital) ** (365 / days) - 1
            ) * 100

        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peak) / peak
        results.max_drawdown_pct = float(drawdowns.min() * 100)
        results.calmar_ratio = (
            results.cagr_pct / abs(results.max_drawdown_pct)
            if results.max_drawdown_pct != 0
            else 0.0
        )

        max_dd_idx = int(np.argmin(drawdowns))
        if max_dd_idx < len(equity_arr) - 1:
            recovery_slice = equity_arr[max_dd_idx:]
            recovery_idx = max_dd_idx + int(
                np.argmax(recovery_slice >= peak[max_dd_idx])
            )
            recovery_delta = times_arr[recovery_idx] - times_arr[max_dd_idx]
            results.max_dd_recovery_days = int(recovery_delta / np.timedelta64(1, "D"))

        returns = np.diff(equity_arr) / equity_arr[:-1]
        if len(returns) > 1:
            std = np.std(returns)
            results.sharpe_ratio = (
                float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0
            )
            downside = returns[returns < 0]
            downside_std = np.std(downside)
            results.sortino_ratio = (
                float(np.mean(returns) / downside_std * np.sqrt(252))
                if len(downside) and downside_std > 0
                else 0.0
            )

        notional_arr = np.array([n for _, n in open_notional_history])
        realized_arr = np.array([e for _, e in realized_equity_history])
        lev_series = notional_arr / np.where(realized_arr > 0, realized_arr, np.nan)
        results.avg_leverage = float(np.nanmean(lev_series))
        if backtester.liq_tracker.leverage is not None:
            high_thresh = 0.8 * backtester.liq_tracker.leverage
            results.pct_time_high_lev = float(np.mean(lev_series > high_thresh)) * 100

    # Buy-and-hold benchmark (first asset/timeframe)
    if backtester.asset_list and backtester.timeframes:
        key = (backtester.asset_list[0], backtester.timeframes[0])
        if key in backtester.arrays:
            arrays = backtester.arrays[key]
            if len(arrays["Close"]) > 0:
                results.initial_price = arrays["Close"][0]
                results.final_price = arrays["Close"][-1]
                if results.initial_price > 0:
                    results.bh_return_pct = (
                        (results.final_price - results.initial_price)
                        / results.initial_price
                        * 100
                    )

    # Expectancy & SQN
    if results.overall_metrics.get("total_trades", 0) > 0:
        win_rate_pct = results.overall_metrics["win_rate"]
        avg_win = results.overall_metrics["avg_win"]
        avg_loss_abs = abs(results.overall_metrics["avg_loss"])
        results.expectancy = (win_rate_pct / 100 * avg_win) - ((100 - win_rate_pct) / 100 * avg_loss_abs)

        if len(results.trade_pnls) > 1:
            std = np.std(results.trade_pnls)
            if std != 0:
                results.sqn = results.expectancy / std * math.sqrt(results.overall_metrics["total_trades"])

    notional_arr = np.array([n for _, n in open_notional_history])
    realized_arr = np.array([e for _, e in realized_equity_history])
    lev_series = notional_arr / np.where(realized_arr > 0, realized_arr, np.nan)

    # Existing avg_leverage (keep or replace)
    results.avg_leverage = float(np.nanmean(lev_series)) if np.any(~np.isnan(lev_series)) else 0.0

    # Max leverage
    results.max_leverage_achieved = float(np.nanmax(lev_series)) if np.any(~np.isnan(lev_series)) else 0.0

    # Min margin ratio (only over periods where positions were actually open)
    margin_vals = np.array([mr for _, mr in margin_ratio_history])
    finite_margin = margin_vals[np.isfinite(margin_vals)]
    if len(finite_margin) > 0:
        results.min_margin_ratio = float(np.min(finite_margin))
    else:
        results.min_margin_ratio = float("inf")

    # Total bars (for % calculations)
    results.total_bars = len(backtester.master_time_list)

    # Danger zone (high leverage usage close to cap)
    if backtester.liq_tracker.leverage is not None:
        # Danger = leverage > 95% of cap (stricter than the 80% used for % time high lev)
        danger_thresh = 0.95 * backtester.liq_tracker.leverage
        danger_mask = lev_series > danger_thresh
        results.total_danger_bars = int(np.sum(~np.isnan(lev_series) & danger_mask))

        # Count distinct episodes (transitions into danger zone)
        in_danger = ~np.isnan(lev_series) & danger_mask
        transitions = np.diff(np.concatenate(([False], in_danger.astype(int)))) == 1
        results.danger_episodes = int(np.sum(transitions))
    else:
        results.total_danger_bars = 0
        results.danger_episodes = 0

    return results
