"""
Container for all backtest outputs: time-series histories, trade lists,
aggregated metrics (Sharpe, Calmar, SQN, etc.), and risk/liquidation statistics.
Designed for easy analysis and reporting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import datetime as dt

from src.backtester.core.trade import Trade


@dataclass
class BacktestResults:
    # Core Histories
    equity_history: List[Tuple[dt.datetime, float]] = field(default_factory=list)
    realized_equity_history: List[Tuple[dt.datetime, float]] = field(
        default_factory=list
    )
    open_notional_history: List[Tuple[dt.datetime, float]] = field(default_factory=list)
    open_trade_history: List[Tuple[dt.datetime, int]] = field(default_factory=list)
    maintenance_margin_history: List[Tuple[dt.datetime, float]] = field(
        default_factory=list
    )
    margin_ratio_history: List[Tuple[dt.datetime, float]] = field(default_factory=list)

    # Trade-level data
    all_trades: List[Trade] = field(default_factory=list)
    trade_pnls: List[float] = field(default_factory=list)
    trade_r_multiples: List[float] = field(default_factory=list)
    entry_times: List[dt.datetime] = field(default_factory=list)

    # Activity & dormancy
    dormant_periods: List[dt.timedelta] = field(default_factory=list)
    active_bars: int = 0
    total_bars: int = 0

    # Per-timeframe stats
    timeframe_entries: Dict[str, List[dt.datetime]] = field(default_factory=dict)

    # Risk & liquidation
    liquidated: bool = False
    liq_time: Optional[dt.datetime] = None
    danger_episodes: int = 0
    total_danger_bars: int = 0
    min_buffer_fraction: float = float("inf")
    min_margin_ratio: float = float("inf")
    max_leverage_achieved: float = 0.0
    danger_buffer: float = 0.30

    # Leverage impact
    total_entry_signals: int = 0
    leverage_fully_prevented: int = 0
    leverage_partially_reduced: int = 0
    pyramiding_prevented: int = 0
    post_liquidation_prevented: int = 0

    # Buy & Hold benchmark
    initial_price: Optional[float] = None
    final_price: Optional[float] = None

    # Aggregated & derived metrics
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    per_series_metrics: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=dict
    )

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    max_dd_recovery_days: int = 0
    avg_leverage: float = 0.0
    pct_time_high_lev: float = 0.0
    total_return_pct: float = 0.0
    bh_return_pct: float = 0.0
    expectancy: float = 0.0
    sqn: float = 0.0
    avg_r_multiple: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    activity_pct: float = 0.0
    num_dormant_periods: int = 0
    avg_dormant_hours: float = 0.0
    max_dormant_hours: float = 0.0
    avg_gap_hours: float = 0.0
    max_gap_hours: float = 0.0
