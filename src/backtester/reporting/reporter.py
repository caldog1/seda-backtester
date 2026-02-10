"""Rich reporting for backtest results.

Generates detailed console summaries and beautiful standalone HTML reports
with configuration, performance metrics, execution analysis, rolling statistics,
and embedded Matplotlib dashboards.
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from backtester.core.engine import Backtester
from backtester.reporting.plotter import Plotter
from backtester.core.results import BacktestResults

# Clean console output — removes "INFO:module:" prefix globally
logging.basicConfig(level=logging.INFO)
if logging.getLogger().handlers:
    logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger(__name__)


class Reporter:
    """Generates console and HTML reports from backtest results."""

    def __init__(
        self,
        results: BacktestResults,
        backtester: Backtester,
        plotter: Optional[Plotter] = None,
    ) -> None:
        self.results = results
        self.backtester = backtester
        self.plotter = plotter or Plotter(results, backtester)

    def _fig_to_base64(self, fig) -> str:
        """Convert Matplotlib figure to base64 PNG for HTML embedding."""
        if fig is None:
            return ""
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    def _format_config_for_console(self) -> None:
        """Print detailed configuration summary (changed to print for test capture)."""
        bt = self.backtester
        print("\n" + "═" * 80)
        print(" BACKTEST CONFIGURATION")
        print("═" * 80)
        print(
            f"{'Period':<30} {bt.start_date} → {bt.end_date or 'latest available'}"
        )
        print(f"{'Initial Capital':<30} ${bt.initial_capital:,.2f}")
        lev = bt.liq_tracker.leverage
        mode = bt.liq_tracker.mode or "none"
        print(
            f"{'Leverage Cap':<30} {lev if lev is not None else 'Unlimited'}x (mode: {mode})"
        )
        print(f"{'Assets':<30} {', '.join(sorted(bt.asset_list)) or 'None'}")
        print(f"{'Timeframes':<30} {', '.join(sorted(bt.timeframes))}")
        print(f"{'Total Strategies':<30} {len(bt.strategies)}")
        print("-" * 80)
        for i, strategy in enumerate(bt.strategies, 1):
            print(
                f"\nStrategy {i}: {strategy.name} ({strategy.__class__.__name__})"
            )
            print(f" {'Sizer':<20} {strategy.sizer.__class__.__name__}")
            sizer_params = {
                k: v
                for k, v in strategy.sizer.__dict__.items()
                if not k.startswith("_")
            }
            for k, v in sizer_params.items():
                print(f" {k:<18} : {v}")
            print(
                f" {'Pyramiding':<20} {getattr(strategy, 'allow_pyramiding', 'N/A')}"
            )
            print(" Parameters:")
            params = (
                strategy.params.__dict__
                if hasattr(strategy.params, "__dict__")
                else strategy.params
            )
            for k, v in params.items():
                if not k.startswith("_"):
                    print(f" {k:<25} : {v}")

    def _format_config_for_html(self) -> str:
        """Return HTML snippet with full configuration."""
        bt = self.backtester
        config_html = f"""
        <div class="card p-4 mb-5">
            <h2>Backtest Configuration</h2>
            <table class="metrics-table">
                <tr><th>Period</th><td>{bt.start_date} → {bt.end_date or 'latest available'}</td></tr>
                <tr><th>Initial Capital</th><td>${bt.initial_capital:,.2f}</td></tr>
                <tr><th>Leverage Cap</th><td>{bt.liq_tracker.leverage if bt.liq_tracker.leverage is not None else 'Unlimited'}x 
                    (mode: {bt.liq_tracker.mode or 'none'})</td></tr>
                <tr><th>Assets</th><td>{', '.join(sorted(bt.asset_list)) or 'None'}</td></tr>
                <tr><th>Timeframes</th><td>{', '.join(sorted(bt.timeframes))}</td></tr>
                <tr><th>Number of Strategies</th><td>{len(bt.strategies)}</td></tr>
            </table>

            <h3 class="mt-4">Strategies</h3>
        """
        for strategy in bt.strategies:
            model_str = ""
            if hasattr(strategy, "model_names") and strategy.model_names:
                model_str = (
                    f"<br><strong>AI Models:</strong> {', '.join(strategy.model_names)}"
                )

            config_html += f"""
            <div class="card p-3 mb-3" style="background:#252525;">
                <h4>{strategy.name} ({strategy.__class__.__name__}){model_str}</h4>
                <div class="row">
                    <div class="col-md-4">
                        <strong>Sizer:</strong> {strategy.sizer.__class__.__name__}<br>
            """
            sizer_params = {
                k: v
                for k, v in strategy.sizer.__dict__.items()
                if not k.startswith("_")
            }
            for k, v in sizer_params.items():
                config_html += f"<strong>{k}:</strong> {v}<br>"

            config_html += f"<br><strong>Pyramiding Allowed:</strong> {getattr(strategy, 'allow_pyramiding', 'N/A')}<br>"

            config_html += """
                    </div>
                    <div class="col-md-8">
                        <strong>Parameters:</strong>
                        <table class="metrics-table mt-2">
                            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                            <tbody>
            """
            params = (
                strategy.params.__dict__
                if hasattr(strategy.params, "__dict__")
                else strategy.params
            )
            for k, v in params.items():
                if not k.startswith("_"):
                    config_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            config_html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            """
        config_html += "</div>"
        return config_html

    def _compute_position_activity(self) -> Dict[str, float]:
        """Compute time-weighted position activity and dormant period statistics.

        Robust handling for timestamps that may be datetime.datetime, pandas.Timestamp,
        or numpy.datetime64[ns] (common in OHLCV arrays).
        """
        history = self.results.open_trade_history
        if len(history) < 2:
            return {
                "pct_time_active": 0.0,
                "num_dormant_periods": 0,
                "avg_dormant_hrs": 0.0,
                "max_dormant_hrs": 0.0,
            }

        # Convert all timestamps to pandas.Timestamp for uniform handling
        times = [pd.to_datetime(t) for t, _ in history]
        counts = [c for _, c in history]

        # Convert to numpy arrays for reliable vectorised timedelta operations
        times_np = np.array(times, dtype="datetime64[ns]")
        counts_np = np.array(counts, dtype=int)

        # Total backtest duration in hours
        total_duration_h = float(
            (times_np[-1] - times_np[0]) / np.timedelta64(1, "h")
        )
        if total_duration_h <= 0:
            return {
                "pct_time_active": 0.0,
                "num_dormant_periods": 0,
                "avg_dormant_hrs": 0.0,
                "max_dormant_hrs": 0.0,
            }

        # Interval durations in hours (between consecutive snapshots)
        interval_hours = np.diff(times_np) / np.timedelta64(1, "h")
        interval_hours = interval_hours.astype(float)  # convert to float for summing

        # Active duration: the state at snapshot i-1 holds until snapshot i
        active_mask = counts_np[:-1] > 0
        active_duration_h = np.sum(interval_hours[active_mask])

        pct_time_active = round(active_duration_h / total_duration_h * 100, 2)

        # Dormant periods — only count periods that span at least one full interval
        dormant_periods_h: List[float] = []
        i = 0
        n = len(counts_np)
        while i < n - 1:  # need room for at least one interval
            if counts_np[i] == 0:
                start_i = i
                i += 1
                while i < n - 1 and counts_np[i] == 0:
                    i += 1
                # Duration from start of dormant run to end of last zero-count interval
                duration_h = float(
                    (times_np[i] - times_np[start_i]) / np.timedelta64(1, "h")
                )
                if duration_h > 0:
                    dormant_periods_h.append(duration_h)
            else:
                i += 1

        num_dormant = len(dormant_periods_h)
        avg_dormant_hrs = (
            round(float(np.mean(dormant_periods_h)), 2)
            if dormant_periods_h
            else 0.0
        )
        max_dormant_hrs = (
            round(float(np.max(dormant_periods_h)), 2)
            if dormant_periods_h
            else 0.0
        )

        return {
            "pct_time_active": pct_time_active,
            "num_dormant_periods": int(num_dormant),
            "avg_dormant_hrs": avg_dormant_hrs,
            "max_dormant_hrs": max_dormant_hrs,
        }

    def _compute_frequency_stats(
        self, entry_times_list: List[dt.datetime]
    ) -> Dict[str, float]:
        if not entry_times_list:
            return {"entries": 0, "avg_gap_hrs": 0.0, "max_gap_hrs": 0.0}
        entries = len(entry_times_list)
        entry_times_list = sorted(entry_times_list)
        gaps = []
        for i in range(len(entry_times_list) - 1):
            delta = entry_times_list[i + 1] - entry_times_list[i]
            hours = (
                delta.total_seconds() / 3600
                if hasattr(delta, "total_seconds")
                else float(delta / np.timedelta64(1, "h"))
            )
            gaps.append(hours)
        avg_gap = np.mean(gaps) if gaps else 0.0
        max_gap = np.max(gaps) if gaps else 0.0
        return {"entries": entries, "avg_gap_hrs": avg_gap, "max_gap_hrs": max_gap}

    def _compute_initial_period_performance(self, days: int) -> Dict[str, float]:
        if len(self.results.equity_history) < 2:
            return {
                "return_pct": 0.0,
                "max_dd_pct": 0.0,
                "trades_entered": 0,
                "completed_trades": 0,
                "win_rate": 0.0,
            }
        df = pd.DataFrame(self.results.equity_history, columns=["time", "equity"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        start_time = df.index[0]
        end_time = start_time + pd.Timedelta(days=days)
        df_period = df[df.index <= end_time]
        if len(df_period) < 2:
            return {
                "return_pct": 0.0,
                "max_dd_pct": 0.0,
                "trades_entered": 0,
                "completed_trades": 0,
                "win_rate": 0.0,
            }
        equity_start = df_period["equity"].iloc[0]
        equity_end = df_period["equity"].iloc[-1]
        return_pct = (equity_end / equity_start - 1) * 100
        peak = df_period["equity"].cummax()
        dd = (df_period["equity"] - peak) / peak
        max_dd_pct = dd.min() * 100
        trades_in_period = [
            tr
            for tr in self.results.all_trades
            if start_time <= tr.entry_time <= end_time
        ]
        trades_entered = len(trades_in_period)
        completed = [tr for tr in trades_in_period if tr.exit_time is not None]
        win_rate = (
            len([tr for tr in completed if tr.pnl > 0]) / len(completed) * 100
            if completed
            else 0.0
        )
        return {
            "return_pct": return_pct,
            "max_dd_pct": max_dd_pct,
            "trades_entered": trades_entered,
            "completed_trades": len(completed),
            "win_rate": win_rate,
        }

    def _compute_periodic_stats(self, freq: str = "W") -> Dict[str, float]:
        if len(self.results.equity_history) < 2:
            return {}
        df = pd.DataFrame(self.results.equity_history, columns=["time", "equity"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        rule = "W-FRI" if freq == "W" else "ME"
        resampled = df["equity"].resample(rule).last().ffill()
        returns_pct = resampled.pct_change().dropna() * 100
        if len(returns_pct) == 0:
            return {}
        stats = {
            "num_periods": len(returns_pct),
            "avg_return_pct": returns_pct.mean(),
            "median_return_pct": returns_pct.median(),
            "std_return_pct": returns_pct.std(),
            "best_return_pct": returns_pct.max(),
            "worst_return_pct": returns_pct.min(),
            "positive_pct": (returns_pct > 0).mean() * 100,
        }
        if self.results.all_trades:
            entry_times = [tr.entry_time for tr in self.results.all_trades]
            entry_series = pd.Series(1, index=pd.to_datetime(entry_times))
            trades_per_period = entry_series.resample(rule).count()
            stats.update(
                {
                    "avg_trades": trades_per_period.mean(),
                    "max_trades": trades_per_period.max(),
                    "periods_with_trades_pct": (trades_per_period > 0).mean() * 100,
                }
            )
        return stats

    def _compute_execution_metrics(self) -> Tuple[Dict, Dict, List]:
        if not self.results.all_trades:
            return {}, {}, []
        per_strat = defaultdict(
            lambda: {
                "trades": 0,
                "entry_slip_bps": [],
                "exit_slip_bps": [],
                "entry_fee_dollar": 0.0,
                "exit_fee_dollar": 0.0,
                "entry_slip_dollar": 0.0,
                "exit_slip_dollar": 0.0,
            }
        )
        high_slip_events = []
        for trade in self.results.all_trades:
            strat = trade.strategy_name or "Unknown"
            data = per_strat[strat]
            data["trades"] += 1
            data["entry_slip_bps"].append(trade.entry_slippage_bps or 0.0)
            data["exit_slip_bps"].append(trade.exit_slippage_bps or 0.0)
            data["entry_slip_dollar"] += trade.entry_slippage_dollar or 0.0
            data["exit_slip_dollar"] += trade.exit_slippage_dollar or 0.0
            data["entry_fee_dollar"] += trade.entry_fee_dollar or 0.0
            data["exit_fee_dollar"] += trade.exit_fee_dollar or 0.0
            asset_tf = f"{trade.asset or 'Unknown'}/{trade.timeframe or 'Unknown'}"
            if trade.entry_slippage_bps > 15:
                high_slip_events.append(
                    (
                        trade.entry_slippage_bps,
                        "Entry",
                        strat,
                        asset_tf,
                        trade.entry_time,
                        trade.entry_slippage_dollar,
                        trade.trade_value,
                    )
                )
            if trade.exit_slippage_bps > 15:
                high_slip_events.append(
                    (
                        trade.exit_slippage_bps,
                        "Exit",
                        strat,
                        asset_tf,
                        trade.exit_time,
                        trade.exit_slippage_dollar,
                        trade.trade_value,
                    )
                )
        total_trades = len(self.results.all_trades)
        overall = {
            "total_fee_dollar": sum(
                d["entry_fee_dollar"] + d["exit_fee_dollar"] for d in per_strat.values()
            ),
            "total_slip_dollar": sum(
                d["entry_slip_dollar"] + d["exit_slip_dollar"]
                for d in per_strat.values()
            ),
            "avg_entry_slip_bps": (
                np.mean([b for d in per_strat.values() for b in d["entry_slip_bps"]])
                if total_trades
                else 0.0
            ),
            "avg_exit_slip_bps": (
                np.mean([b for d in per_strat.values() for b in d["exit_slip_bps"]])
                if total_trades
                else 0.0
            ),
            "pct_trades_with_slip": 100
            * sum(
                1
                for t in self.results.all_trades
                if (t.entry_slippage_bps or 0) + (t.exit_slippage_bps or 0) > 0
            )
            / total_trades,
        }
        overall["total_cost_dollar"] = (
            overall["total_fee_dollar"] + overall["total_slip_dollar"]
        )
        high_slip_events.sort(reverse=True, key=lambda x: x[0])
        high_slip_events = high_slip_events[:20]
        per_strat_summary = {}
        for strat, data in per_strat.items():
            per_strat_summary[strat] = {
                "trades": data["trades"],
                "total_fee": data["entry_fee_dollar"] + data["exit_fee_dollar"],
                "total_slip": data["entry_slip_dollar"] + data["exit_slip_dollar"],
                "avg_entry_slip_bps": (
                    np.mean(data["entry_slip_bps"]) if data["entry_slip_bps"] else 0.0
                ),
                "avg_exit_slip_bps": (
                    np.mean(data["exit_slip_bps"]) if data["exit_slip_bps"] else 0.0
                ),
            }
        return overall, per_strat_summary, high_slip_events

    def generate_console_report(self, verbose: bool = True) -> None:
        """Generate a clean, polished, and professional console report."""
        r = self.results

        print("\n" + "═" * 80)
        print("SEDA Backtest Report")
        print("═" * 80)

        if not r.equity_history or r.overall_metrics.get("total_trades", 0) == 0:
            print("No trades executed — no performance data available")
            print("=" * 80)
            return

        # Headline metrics (always shown)
        print(f"Total Return       : {r.total_return_pct:+.2f}%")
        print(f"Sharpe Ratio       : {r.sharpe_ratio:.2f}")
        print(f"Calmar Ratio       : {r.calmar_ratio:.2f}")
        print(f"Max Drawdown (MTM) : {r.max_drawdown_pct:.2f}%")

        if r.liquidated and r.liq_time:
            print(f"\n!!! ACCOUNT LIQUIDATED at {r.liq_time.strftime('%Y-%m-%d')} !!!")

        if not verbose:
            print("═" * 80)
            return

        bt = self.backtester
        initial_capital = bt.initial_capital
        leverage_cap = bt.liq_tracker.leverage
        self._format_config_for_console()
        # Summary Dashboard
        total_return_pct = r.total_return_pct
        net_pnl = r.overall_metrics.get("net_pnl", 0.0)
        bh_return_pct = r.bh_return_pct
        out_under = (
            "Outperformed" if total_return_pct > bh_return_pct else "Underperformed"
        )
        print("\n" + "═" * 80)
        print(" BACKTEST SUMMARY DASHBOARD")
        print("═" * 80)
        print(f"{'Total Return':<35} {total_return_pct:+8.2f}%")
        print(f"{'Net PnL':<35} ${net_pnl:>15,.2f}")
        if bh_return_pct not in (0.0, None):
            print(f"{'vs Buy & Hold':<35} {bh_return_pct:+8.2f}% ({out_under})")
        print(f"{'CAGR':<35} {r.cagr_pct:+8.2f}%")
        print(f"{'Sharpe Ratio':<35} {r.sharpe_ratio:8.2f}")
        print(f"{'Max Drawdown':<35} {r.max_drawdown_pct:+8.2f}%")
        print(f"{'Calmar Ratio':<35} {r.calmar_ratio:8.2f}")
        print(
            f"{'Profit Factor':<35} {r.overall_metrics.get('profit_factor', 0.0):8.2f}"
        )
        print(f"{'Win Rate':<35} {r.overall_metrics.get('win_rate', 0.0):8.2f}%")
        print(f"{'Min Margin Ratio':<35} {r.min_margin_ratio:8.2f}x")

        # Detailed Performance
        om = r.overall_metrics
        print("\n" + "═" * 80)
        print(" DETAILED PERFORMANCE METRICS")
        print("═" * 80)
        print(f"{'Total Trades':<35} {om.get('total_trades', 0):>10}")
        print(f"{'Win Rate':<35} {om.get('win_rate', 0.0):>10.2f}%")
        print(f"{'Expectancy ($/trade)':<35} {r.expectancy:>10.2f}")
        print(f"{'Average R-Multiple':<35} {r.avg_r_multiple:>10.2f}")
        print(f"{'SQN':<35} {r.sqn:>10.2f}")
        print(f"{'Max Win Streak':<35} {om['max_win_streak']:>10}")
        print(f"{'Max Loss Streak':<35} {om['max_loss_streak']:>10}")
        print(f"{'Average Leverage':<35} {r.avg_leverage:>10.2f}x")
        if leverage_cap is not None:
            print(f"{'% Time >80% of Cap':<35} {r.pct_time_high_lev:>10.2f}%")
        print(f"{'Ulcer Index':<35} {r.ulcer_index:>10.2f}")
        print(f"{'Max DD Recovery (days)':<35} {r.max_dd_recovery_days:>10}")

        # Trade Frequency & Activity
        entry_stats = self._compute_frequency_stats(self.results.entry_times)
        activity_stats = self._compute_position_activity()

        print("\n" + "═" * 80)
        print(" TRADE FREQUENCY & ACTIVITY")
        print("═" * 80)
        print(f"{'Total Entries':<35} {entry_stats['entries']:>10}")
        print(f"{'Avg Gap Between Entries (hrs)':<35} {entry_stats['avg_gap_hrs']:>10.2f}")
        print(f"{'Max Gap Between Entries (hrs)':<35} {entry_stats['max_gap_hrs']:>10.2f}")
        print(f"{'% Time with Active Positions':<35} {activity_stats['pct_time_active']:>10.2f}%")
        print(f"{'Number of Dormant Periods':<35} {activity_stats['num_dormant_periods']:>10}")
        print(f"{'Avg Dormant Period (hrs)':<35} {activity_stats['avg_dormant_hrs']:>10.2f}")
        print(f"{'Max Dormant Period (hrs)':<35} {activity_stats['max_dormant_hrs']:>10.2f}")

        # Per Asset/Timeframe Trade Statistics & Frequency (computed on-the-fly for accuracy)
        from collections import defaultdict

        series_stats = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
            "entry_times": []
        })

        for trade in r.all_trades:
            key = f"{trade.asset.upper()}/{trade.timeframe}" if trade.asset and trade.timeframe else "Unknown"
            data = series_stats[key]
            data["trades"] += 1
            if trade.pnl > 0:
                data["wins"] += 1
            data["pnl"] += trade.pnl
            if trade.entry_time:
                data["entry_times"].append(trade.entry_time)

        print("\n" + "═" * 80)
        print(" TRADE STATISTICS & FREQUENCY – PER ASSET/TIMEFRAME")
        print("═" * 80)

        if series_stats:
            for key in sorted(series_stats.keys()):
                data = series_stats[key]
                count = data["trades"]
                win_rate = (data["wins"] / count * 100) if count > 0 else 0.0
                pnl = data["pnl"]

                # Frequency stats
                entry_list = sorted(data["entry_times"])
                freq = self._compute_frequency_stats(entry_list)

                print(f"{key}")
                print(f" {'Trades':<30} {count:>10}")
                print(f" {'Win Rate':<30} {win_rate:>10.2f}%")
                print(f" {'Net PnL Contribution':<30} ${pnl:>10,.2f}")
                print(f" {'Total Entries':<30} {freq['entries']:>10}")
                print(f" {'Avg Gap Between Entries (hrs)':<30} {freq['avg_gap_hrs']:>10.2f}")
                print(f" {'Max Gap Between Entries (hrs)':<30} {freq['max_gap_hrs']:>10.2f}")
                print(" " + "-" * 80)
        else:
            print("No trades executed – no per asset/timeframe data available.")

        # Risk & Liquidation
        print("\n" + "═" * 80)
        print(" RISK & LIQUIDATION STATISTICS")
        print("═" * 80)
        print(f"{'Max Effective Leverage':<35} {r.max_leverage_achieved:>10.2f}x")
        if leverage_cap is not None:
            print(f"{'(Cap set at)':<35} {leverage_cap:>10.2f}x")
        print(f"{'Danger Zone Episodes':<35} {getattr(r, 'danger_episodes', 0):>10}")
        danger_pct = (getattr(r, 'total_danger_bars', 0) / (r.total_bars or 1)) * 100
        print(f"{'Time in Danger Zone':<35} {getattr(r, 'total_danger_bars', 0):>8} bars ({danger_pct:>6.2f}%)")
        if r.min_margin_ratio == float("inf"):
            print(f"{'Minimum Margin Ratio':<35} {'Never below maintenance':>20}")
        else:
            print(f"{'Minimum Margin Ratio':<35} {r.min_margin_ratio:>10.2f}x")
        if r.liquidated:
            print(f"{'Liquidation Time':<35} {r.liq_time.strftime('%Y-%m-%d %H:%M') if r.liq_time else 'Unknown'}")

        # Execution Impact
        overall_exec, per_strat_exec, high_slips = self._compute_execution_metrics()
        print("\n" + "═" * 80)
        print(" EXECUTION IMPACT ANALYSIS")
        print("═" * 80)
        print(
            f"{'Total Fees Paid':<40} ${overall_exec.get('total_fee_dollar', 0.0):>12,.2f}"
        )
        print(
            f"{'Total Slippage Cost':<40} ${overall_exec.get('total_slip_dollar', 0.0):>12,.2f}"
        )
        total_cost = overall_exec.get("total_cost_dollar", 0.0)
        print(
            f"{'Total Execution Cost':<40} ${total_cost:>12,.2f} ({total_cost/initial_capital*100:>6.2f}% of capital)"
        )
        print(
            f"{'Avg Entry Slippage':<40} {overall_exec.get('avg_entry_slip_bps', 0.0):>12.2f} bps"
        )
        print(
            f"{'Avg Exit Slippage':<40} {overall_exec.get('avg_exit_slip_bps', 0.0):>12.2f} bps"
        )
        print(
            f"{'% Trades Affected':<40} {overall_exec.get('pct_trades_with_slip', 0.0):>12.1f}%"
        )
        print("\nPer-Strategy Execution Breakdown")
        print("-" * 80)
        header = f"{'Strategy':<25} {'Trades':<8} {'Total Cost':<12} {'Fees':<12} {'Slippage':<12} {'Entry Slip':<12} {'Exit Slip':<12}"
        print(header)
        print("-" * 80)
        for strat, data in per_strat_exec.items():
            print(
                f"{strat:<25} {data['trades']:<8} "
                f"${data['total_fee'] + data['total_slip']:>10,.0f} "
                f"${data['total_fee']:>10,.0f} "
                f"${data['total_slip']:>10,.0f} "
                f"{data['avg_entry_slip_bps']:>10.2f} bps "
                f"{data['avg_exit_slip_bps']:>10.2f} bps"
            )
        if high_slips:
            print("\nTop 20 Worst Slippage Events (>15 bps)")
            print("-" * 80)
            header = f"{'BPS':>8} {'Side':<6} {'Strategy':<20} {'Asset/TF':<15} {'Date':<12} {'$ Impact':>12} {'Notional':>12}"
            print(header)
            print("-" * len(header))
            for bps, side, strat, asset_tf, time, dollar, notional in high_slips:
                date_str = pd.Timestamp(time).strftime("%Y-%m-%d")
                print(
                    f"{bps:>8.1f} {side:<6} {strat:<20} {asset_tf:<15} {date_str:<12} "
                    f"${dollar:>11,.0f} ${notional:>11,.0f}"
                )
        # Leverage Cap Impact
        print("\n" + "═" * 80)
        print(" LEVERAGE CAP IMPACT ON ENTRIES")
        print("═" * 80)
        print(f"{'Total Entry Signals':<40} {r.total_entry_signals:>10}")
        print(f"{'Trades Opened':<40} {om.get('total_trades', 0):>10}")
        print(f"{'Fully Prevented':<40} {r.leverage_fully_prevented:>10}")
        print(f"{'Partially Reduced':<40} {r.leverage_partially_reduced:>10}")
        print(f"{'Prevented by Pyramiding':<40} {r.pyramiding_prevented:>10}")
        if r.total_entry_signals > 0:
            prevented_pct = (r.leverage_fully_prevented / r.total_entry_signals) * 100
            print(f"{'% Fully Blocked by Leverage':<40} {prevented_pct:>9.2f}%")
        # Trade Statistics
        print("\n" + "═" * 80)
        print(" TRADE STATISTICS")
        print("═" * 80)
        print(f"{'Gross Profit':<35} ${om.get('gross_profit', 0.0):>15,.2f}")
        print(f"{'Gross Loss':<35} ${om.get('gross_loss', 0.0):>15,.2f}")
        print(f"{'Net PnL':<35} ${om.get('net_pnl', 0.0):>15,.2f}")
        print(f"{'Average Win':<35} ${om.get('avg_win', 0.0):>15,.2f}")
        print(f"{'Average Loss':<35} ${om.get('avg_loss', 0.0):>15,.2f}")
        print(f"{'Average Trade':<35} ${om.get('avg_trade', 0.0):>15,.2f}")
        print(f"{'Largest Win':<35} ${om.get('largest_win', 0.0):>15,.2f}")
        print(f"{'Largest Loss':<35} ${om.get('largest_loss', 0.0):>15,.2f}")
        print(f"{'Total Fees':<35} ${om.get('total_fees', 0.0):>15,.2f}")
        print(
            f"{'Avg Hold Time (hrs)':<35} {om.get('avg_duration_hours', 0.0):>15.2f}"
        )
        print(
            f"{'Median Hold Time (hrs)':<35} {om.get('median_duration_hours', 0.0):>15.2f}"
        )
        # Initial Period + Rolling Stats
        week_init = self._compute_initial_period_performance(7)
        month_init = self._compute_initial_period_performance(30)
        weekly = self._compute_periodic_stats("W")
        monthly = self._compute_periodic_stats("M")
        print("\n" + "═" * 80)
        print(" INITIAL PERIOD PERFORMANCE")
        print("═" * 80)
        print("First 7 Days")
        print(f" {'Return':<25} {week_init['return_pct']:+8.2f}%")
        print(f" {'Max Drawdown':<25} {week_init['max_dd_pct']:+8.2f}%")
        print(f" {'Trades Entered':<25} {week_init['trades_entered']:>8}")
        if week_init["completed_trades"] > 0:
            print(f" {'Win Rate':<25} {week_init['win_rate']:>8.2f}%")
        print("\nFirst 30 Days")
        print(f" {'Return':<25} {month_init['return_pct']:+8.2f}%")
        print(f" {'Max Drawdown':<25} {month_init['max_dd_pct']:+8.2f}%")
        print(f" {'Trades Entered':<25} {month_init['trades_entered']:>8}")
        if month_init["completed_trades"] > 0:
            print(f" {'Win Rate':<25} {month_init['win_rate']:>8.2f}%")
        if weekly:
            print("\n" + "═" * 80)
            print(" WEEKLY PERFORMANCE STATISTICS")
            print("═" * 80)
            print(f"{'Number of Weeks':<30} {weekly['num_periods']:>10}")
            print(f"{'Average Return':<30} {weekly['avg_return_pct']:+10.2f}%")
            print(f"{'Median Return':<30} {weekly['median_return_pct']:+10.2f}%")
            print(f"{'Std Dev':<30} {weekly['std_return_pct']:>10.2f}%")
            print(f"{'Best Week':<30} {weekly['best_return_pct']:+10.2f}%")
            print(f"{'Worst Week':<30} {weekly['worst_return_pct']:+10.2f}%")
            print(f"{'% Positive Weeks':<30} {weekly['positive_pct']:>10.1f}%")
            if "avg_trades" in weekly:
                print(f"{'Avg Trades/Week':<30} {weekly['avg_trades']:>10.2f}")
                print(f"{'Max Trades/Week':<30} {weekly['max_trades']:>10}")
                print(
                    f"{'% Weeks with Trades':<30} {weekly['periods_with_trades_pct']:>10.1f}%"
                )
        if monthly:
            print("\n" + "═" * 80)
            print(" MONTHLY PERFORMANCE STATISTICS")
            print("═" * 80)
            print(f"{'Number of Months':<30} {monthly['num_periods']:>10}")
            print(f"{'Average Return':<30} {monthly['avg_return_pct']:+10.2f}%")
            print(f"{'Median Return':<30} {monthly['median_return_pct']:+10.2f}%")
            print(f"{'Std Dev':<30} {monthly['std_return_pct']:>10.2f}%")
            print(f"{'Best Month':<30} {monthly['best_return_pct']:+10.2f}%")
            print(f"{'Worst Month':<30} {monthly['worst_return_pct']:+10.2f}%")
            print(f"{'% Positive Months':<30} {monthly['positive_pct']:>10.1f}%")
            if "avg_trades" in monthly:
                print(f"{'Avg Trades/Month':<30} {monthly['avg_trades']:>10.2f}")
                print(f"{'Max Trades/Month':<30} {monthly['max_trades']:>10}")
                print(
                    f"{'% Months with Trades':<30} {monthly['periods_with_trades_pct']:>10.1f}%"
                )
        # Monthly / Yearly Returns Tables
        if len(r.equity_history) > 1:
            df_eq = pd.DataFrame(r.equity_history, columns=["time", "mtm_equity"])
            df_eq["time"] = pd.to_datetime(df_eq["time"])
            df_eq = df_eq.set_index("time").sort_index()
            monthly_ret = df_eq["mtm_equity"].resample("ME").last().pct_change() * 100
            print("\n" + "═" * 80)
            print(" MONTHLY RETURNS (%)")
            print("═" * 80)
            for date, ret in monthly_ret.dropna().items():
                print(f" {date.strftime('%Y-%m'):>8} {ret:>12.2f}%")

            print("\n" + "═" * 80)
            print(" YEARLY RETURNS (%)")
            print("═" * 80)

            df_eq["year"] = df_eq.index.year
            yearly_groups = df_eq.groupby("year")

            if len(yearly_groups) == 0:
                print(" Insufficient data for yearly returns")
            else:
                for year, group in yearly_groups:
                    if len(group) < 2:
                        continue  # skip if no movement in that year

                    start_eq = group["mtm_equity"].iloc[0]
                    end_eq = group["mtm_equity"].iloc[-1]
                    ret_pct = (end_eq / start_eq - 1) * 100

                    # Detect if this is the current/partial year
                    year_end = pd.Timestamp(f"{year}-12-31")
                    is_partial = group.index[-1] < year_end

                    label = f"{year} (YTD)" if is_partial else str(year)
                    print(f" {label:>12} {ret_pct:+12.2f}%")
        # Liquidation Warning
        if r.liquidated:
            print("\n" + "!" * 80)
            print("!!! ACCOUNT WAS LIQUIDATED !!!")
            print(f"Liquidation occurred at: {r.liq_time}")
            print(f"Mode: {bt.liq_tracker.mode}")
            print("!" * 80)

        print('\n\n')

    def generate_html_report(self, output_path: str = "backtest_report.html") -> None:
        """Generate comprehensive standalone HTML report with full theme consistency."""
        r = self.results
        bt = self.backtester
        initial_capital = bt.initial_capital
        leverage_cap = bt.liq_tracker.leverage

        # Use the active Plotter theme for full report consistency
        theme = self.plotter.theme

        # Embedded plots
        fig_dashboard = self.plotter.plot_performance_dashboard(return_fig=True)
        fig_activity = self.plotter.plot_activity_dashboard(return_fig=True)
        fig_execution = self.plotter.plot_execution_dashboard(return_fig=True)
        fig_correlations = self.plotter.plot_correlations(return_fig=True)
        img_dashboard = self._fig_to_base64(fig_dashboard)
        img_activity = self._fig_to_base64(fig_activity)
        img_execution = self._fig_to_base64(fig_execution)
        img_correlations = self._fig_to_base64(fig_correlations)

        config_html = self._format_config_for_html()

        # Execution metrics
        overall_exec, per_strat_exec, high_slips = self._compute_execution_metrics()

        # Activity stats (used in top card)
        activity_stats = self._compute_position_activity()

        # Top cards — warnings + key metrics (fully theme-aware)
        top_cards_html = ""
        if r.liquidated:
            top_cards_html += f"""
            <div class="card text-danger border-danger mb-5">
                <h2>⚠️ ACCOUNT LIQUIDATED ⚠️</h2>
                <p><strong>Liquidation Date:</strong> {r.liq_time.strftime('%Y-%m-%d') if r.liq_time else 'Unknown'}</p>
                <p>Final equity near zero — backtest terminated early due to margin call.</p>
            </div>
            """

        if len(r.all_trades) == 0:
            top_cards_html += """
            <div class="card mb-5">
                <h2>No trades executed</h2>
                <p>No entry signals generated during the backtest period.</p>
            </div>
            """

        top_cards_html += f"""
        <div class="card highlight-card mb-5">
            <h2>Key Metrics</h2>
            <div class="row g-4">
                <div class="col-md-4"><strong>Total Return:</strong><br><span class="value">{r.total_return_pct:+.2f}%</span></div>
                <div class="col-md-4"><strong>Sharpe Ratio:</strong><br><span class="value">{r.sharpe_ratio:.2f}</span></div>
                <div class="col-md-4"><strong>Calmar Ratio:</strong><br><span class="value">{r.calmar_ratio:.2f}</span></div>
            </div>
            <div class="row g-4 mt-2">
                <div class="col-md-4"><strong>Max Drawdown:</strong><br><span class="value">{r.max_drawdown_pct:.2f}%</span></div>
                <div class="col-md-4"><strong>CAGR:</strong><br><span class="value">{r.cagr_pct:+.2f}%</span></div>
                <div class="col-md-4"><strong>Sortino Ratio:</strong><br><span class="value">{r.sortino_ratio:.2f}</span></div>
            </div>
            <div class="row g-4 mt-2">
                <div class="col-md-4"><strong>Total Trades:</strong><br><span class="value">{r.overall_metrics.get('total_trades', 0)}</span></div>
                <div class="col-md-4"><strong>Win Rate:</strong><br><span class="value">{r.overall_metrics.get('win_rate', 0.0):.1f}%</span></div>
                <div class="col-md-4"><strong>Profit Factor:</strong><br><span class="value">{r.overall_metrics.get('profit_factor', 0.0):.2f}</span></div>
            </div>
            <div class="row g-4 mt-2">
                <div class="col-md-4"><strong>Average Leverage:</strong><br><span class="value">{r.avg_leverage:.2f}x</span></div>
                <div class="col-md-4"><strong>Time at High Leverage:</strong><br><span class="value">{r.pct_time_high_lev:.1f}%</span></div>
                <div class="col-md-4"><strong>Position Activity %:</strong><br><span class="value">{activity_stats['pct_time_active']:.1f}%</span></div>
            </div>
            <div class="row g-4 mt-2">
                <div class="col-md-12"><strong>Buy & Hold Return:</strong><br><span class="value large">{r.bh_return_pct:+.2f}%</span></div>
            </div>
        </div>
        """

        # All remaining cards (unchanged from your original implementation)
        total_return_pct = r.total_return_pct
        net_pnl = r.overall_metrics.get("net_pnl", 0.0)
        bh_return_pct = r.bh_return_pct
        out_under = (
            "Outperformed" if total_return_pct > bh_return_pct else "Underperformed"
        )
        summary_html = f"""
        <div class="card p-4 mb-5">
            <h2>Backtest Summary Dashboard</h2>
            <table class="metrics-table">
                <tr><th>Total Return</th><td>{total_return_pct:+.2f}%</td></tr>
                <tr><th>Net PnL</th><td>${net_pnl:,.2f}</td></tr>
                {"<tr><th>vs Buy & Hold</th><td>{bh_return_pct:+.2f}% ({out_under})</td></tr>" if bh_return_pct not in (0.0, None) else ""}
                <tr><th>CAGR</th><td>{r.cagr_pct:+.2f}%</td></tr>
                <tr><th>Sharpe Ratio</th><td>{r.sharpe_ratio:.2f}</td></tr>
                <tr><th>Max Drawdown</th><td>{r.max_drawdown_pct:+.2f}%</td></tr>
                <tr><th>Calmar Ratio</th><td>{r.calmar_ratio:.2f}</td></tr>
                <tr><th>Profit Factor</th><td>{r.overall_metrics.get('profit_factor', 0.0):.2f}</td></tr>
                <tr><th>Win Rate</th><td>{r.overall_metrics.get('win_rate', 0.0):.2f}%</td></tr>
                <tr><th>Min Margin Ratio</th><td>{r.min_margin_ratio:.2f}x</td></tr>
            </table>
        </div>
        """

        om = r.overall_metrics
        detailed_html = f"""
        <div class="card p-4 mb-5">
            <h2>Detailed Performance Metrics</h2>
            <table class="metrics-table">
                <tr><th>Total Trades</th><td>{om.get('total_trades', 0)}</td></tr>
                <tr><th>Win Rate</th><td>{om.get('win_rate', 0.0):.2f}%</td></tr>
                <tr><th>Expectancy ($/trade)</th><td>{r.expectancy:.2f}</td></tr>
                <tr><th>Average R-Multiple</th><td>{r.avg_r_multiple:.2f}</td></tr>
                <tr><th>SQN</th><td>{r.sqn:.2f}</td></tr>
                <tr><th>Max Win Streak</th><td>{om.get('max_win_streak', 0)}</td></tr>
                <tr><th>Max Loss Streak</th><td>{om.get('max_loss_streak', 0)}</td></tr>
                <tr><th>Average Leverage</th><td>{r.avg_leverage:.2f}x</td></tr>
                {"<tr><th>% Time >80% of Cap</th><td>{r.pct_time_high_lev:.2f}%</td></tr>" if leverage_cap is not None else ""}
                <tr><th>Ulcer Index</th><td>{r.ulcer_index:.2f}</td></tr>
                <tr><th>Max DD Recovery (days)</th><td>{r.max_dd_recovery_days}</td></tr>
            </table>
        </div>
        """

        frequency_html = f"""
        <div class="card p-4 mb-5">
            <h2>Trade Frequency & Activity</h2>
            <table class="metrics-table">
                <tr><th>Total Entries</th><td>{len(r.entry_times)}</td></tr>
                <tr><th>Avg Gap Between Entries (hrs)</th><td>{r.avg_gap_hours:.2f}</td></tr>
                <tr><th>Max Gap Between Entries (hrs)</th><td>{r.max_gap_hours:.2f}</td></tr>
                <tr><th>% Time with Active Positions</th><td>{activity_stats['pct_time_active']:.2f}%</td></tr>
                <tr><th>Number of Dormant Periods</th><td>{activity_stats['num_dormant_periods']}</td></tr>
                <tr><th>Avg Dormant Period (hrs)</th><td>{activity_stats['avg_dormant_hrs']:.2f}</td></tr>
                <tr><th>Max Dormant Period (hrs)</th><td>{activity_stats['max_dormant_hrs']:.2f}</td></tr>
            </table>
        </div>
        """

        tf_rows = ""
        if r.timeframe_entries:
            for tf in sorted(r.timeframe_entries.keys()):
                stats = self._compute_frequency_stats(r.timeframe_entries[tf])
                tf_rows += f"""
                <tr><td colspan="2"><strong>Timeframe: {tf.upper()}</strong></td></tr>
                <tr><th>Total Entries</th><td>{int(stats['entries'])}</td></tr>
                <tr><th>Avg Gap (hrs)</th><td>{stats['avg_gap_hrs']:.2f}</td></tr>
                <tr><th>Max Gap (hrs)</th><td>{stats['max_gap_hrs']:.2f}</td></tr>
                """
        else:
            tf_rows = "<tr><td>No trades executed – no per-timeframe data available.</td></tr>"
        tf_html = f"""
        <div class="card p-4 mb-5">
            <h2>Trade Frequency & Activity – Per Timeframe</h2>
            <table class="metrics-table">{tf_rows}</table>
        </div>
        """

        total_bars = r.total_bars or 1
        danger_pct = (r.total_danger_bars / total_bars) * 100
        risk_html = f"""
        <div class="card p-4 mb-5">
            <h2>Risk & Liquidation Statistics</h2>
            <table class="metrics-table">
                <tr><th>Max Effective Leverage</th><td>{r.max_leverage_achieved:.2f}x</td></tr>
                {"<tr><th>(Cap set at)</th><td>{leverage_cap:.2f}x</td></tr>" if leverage_cap is not None else ""}
                <tr><th>Danger Zone Episodes</th><td>{r.danger_episodes}</td></tr>
                <tr><th>Time in Danger Zone</th><td>{r.total_danger_bars} bars ({danger_pct:.2f}%)</td></tr>
                <tr><th>Minimum Margin Ratio</th><td>{'Never open' if r.min_margin_ratio == float('inf') else f'{r.min_margin_ratio:.2f}x'}</td></tr>
                {"<tr><th>Liquidation Time</th><td>{r.liq_time.strftime('%Y-%m-%d') if r.liq_time else ''}</td></tr>" if r.liquidated else ""}
            </table>
            {"<p class='text-danger fw-bold mt-3'>!!! ACCOUNT WAS LIQUIDATED !!!</p>" if r.liquidated else ""}
        </div>
        """

        total_cost = overall_exec.get("total_cost_dollar", 0.0)
        exec_overall_html = f"""
        <table class="metrics-table">
            <tr><th>Total Fees Paid</th><td>${overall_exec.get('total_fee_dollar', 0.0):,.2f}</td></tr>
            <tr><th>Total Slippage Cost</th><td>${overall_exec.get('total_slip_dollar', 0.0):,.2f}</td></tr>
            <tr><th>Total Execution Cost</th><td class="text-warning fw-bold">${total_cost:,.2f} ({total_cost / initial_capital * 100:.2f}% of capital)</td></tr>
            <tr><th>Avg Entry Slippage</th><td>{overall_exec.get('avg_entry_slip_bps', 0.0):.2f} bps</td></tr>
            <tr><th>Avg Exit Slippage</th><td>{overall_exec.get('avg_exit_slip_bps', 0.0):.2f} bps</td></tr>
            <tr><th>% Trades Affected</th><td>{overall_exec.get('pct_trades_with_slip', 0.0):.1f}%</td></tr>
        </table>
        """

        per_strat_rows = ""
        for strat, data in per_strat_exec.items():
            total_cost_strat = data["total_fee"] + data["total_slip"]
            per_strat_rows += f"""
            <tr>
                <td>{strat}</td><td>{data['trades']}</td><td>${total_cost_strat:,.0f}</td>
                <td>${data['total_fee']:,.0f}</td><td>${data['total_slip']:,.0f}</td>
                <td>{data['avg_entry_slip_bps']:.2f}</td><td>{data['avg_exit_slip_bps']:.2f}</td>
            </tr>
            """
        per_strat_html = f"""
        <table class="metrics-table">
            <thead><tr><th>Strategy</th><th>Trades</th><th>Total Cost $</th><th>Fees $</th><th>Slippage $</th>
            <th>Entry Slip bps</th><th>Exit Slip bps</th></tr></thead>
            <tbody>{per_strat_rows or "<tr><td colspan='7'>No trades</td></tr>"}</tbody>
        </table>
        """

        high_slip_rows = ""
        for bps, side, strat, asset_tf, time, dollar, notional in high_slips:
            date_str = pd.Timestamp(time).strftime("%Y-%m-%d")
            high_slip_rows += f"""
            <tr>
                <td>{bps:.1f}</td><td>{side}</td><td>{strat}</td><td>{asset_tf}</td>
                <td>{date_str}</td><td>${dollar:,.0f}</td><td>${notional:,.0f}</td>
            </tr>
            """
        high_slip_html = f"""
        <table class="metrics-table">
            <thead><tr><th>BPS</th><th>Side</th><th>Strategy</th><th>Asset/TF</th><th>Date</th><th>$ Impact</th><th>Notional</th></tr></thead>
            <tbody>{high_slip_rows or "<tr><td colspan='7'>No events >15 bps</td></tr>"}</tbody>
        </table>
        """

        execution_html = f"""
        <div class="card p-4 mb-5">
            <h2>Execution Impact Analysis</h2>
            <div class="row mb-4">
                <div class="col-md-6">
                    <h4>Overall Execution Costs</h4>
                    {exec_overall_html}
                </div>
            </div>
            <h4>Per-Strategy Execution Breakdown</h4>
            {per_strat_html}
            <h4 class="mt-4">Top 20 Worst Slippage Events (>15 bps)</h4>
            {high_slip_html}
        </div>
        """

        leverage_impact_html = f"""
        <div class="card p-4 mb-5">
            <h2>Leverage Cap Impact on Entries</h2>
            <table class="metrics-table">
                <tr><th>Total Entry Signals</th><td>{r.total_entry_signals}</td></tr>
                <tr><th>Trades Opened</th><td>{om.get('total_trades', 0)}</td></tr>
                <tr><th>Fully Prevented</th><td>{r.leverage_fully_prevented}</td></tr>
                <tr><th>Partially Reduced</th><td>{r.leverage_partially_reduced}</td></tr>
                <tr><th>Prevented by Pyramiding</th><td>{r.pyramiding_prevented}</td></tr>
                {"<tr><th>% Fully Blocked by Leverage</th><td>{(r.leverage_fully_prevented / r.total_entry_signals * 100 if r.total_entry_signals > 0 else 0):.2f}%</td></tr>" if r.total_entry_signals > 0 else ""}
            </table>
        </div>
        """

        trade_stats_html = f"""
        <div class="card p-4 mb-5">
            <h2>Trade Statistics</h2>
            <table class="metrics-table">
                <tr><th>Gross Profit</th><td>${om.get('gross_profit', 0.0):,.2f}</td></tr>
                <tr><th>Gross Loss</th><td>${om.get('gross_loss', 0.0):,.2f}</td></tr>
                <tr><th>Net PnL</th><td>${om.get('net_pnl', 0.0):,.2f}</td></tr>
                <tr><th>Average Win</th><td>${om.get('avg_win', 0.0):,.2f}</td></tr>
                <tr><th>Average Loss</th><td>${om.get('avg_loss', 0.0):,.2f}</td></tr>
                <tr><th>Average Trade</th><td>${om.get('avg_trade', 0.0):,.2f}</td></tr>
                <tr><th>Largest Win</th><td>${om.get('largest_win', 0.0):,.2f}</td></tr>
                <tr><th>Largest Loss</th><td>${om.get('largest_loss', 0.0):,.2f}</td></tr>
                <tr><th>Total Fees</th><td>${om.get('total_fees', 0.0):,.2f}</td></tr>
                <tr><th>Avg Hold Time (hrs)</th><td>{om.get('avg_duration_hours', 0.0):.2f}</td></tr>
                <tr><th>Median Hold Time (hrs)</th><td>{om.get('median_duration_hours', 0.0):.2f}</td></tr>
            </table>
        </div>
        """

        week_init = self._compute_initial_period_performance(7)
        month_init = self._compute_initial_period_performance(30)
        initial_html = f"""
        <div class="card p-4 mb-5">
            <h2>Initial Period Performance</h2>
            <div class="row">
                <div class="col-md-6">
                    <h4>First 7 Days</h4>
                    <table class="metrics-table">
                        <tr><th>Return</th><td>{week_init['return_pct']:+.2f}%</td></tr>
                        <tr><th>Max DD</th><td>{week_init['max_dd_pct']:+.2f}%</td></tr>
                        <tr><th>Trades Entered</th><td>{week_init['trades_entered']}</td></tr>
                        {f"<tr><th>Win Rate</th><td>{week_init['win_rate']:.2f}%</td></tr>" if week_init['completed_trades'] > 0 else ""}
                    </table>
                </div>
                <div class="col-md-6">
                    <h4>First 30 Days</h4>
                    <table class="metrics-table">
                        <tr><th>Return</th><td>{month_init['return_pct']:+.2f}%</td></tr>
                        <tr><th>Max DD</th><td>{month_init['max_dd_pct']:+.2f}%</td></tr>
                        <tr><th>Trades Entered</th><td>{month_init['trades_entered']}</td></tr>
                        {f"<tr><th>Win Rate</th><td>{month_init['win_rate']:.2f}%</td></tr>" if month_init['completed_trades'] > 0 else ""}
                    </table>
                </div>
            </div>
        </div>
        """

        weekly = self._compute_periodic_stats("W")
        monthly = self._compute_periodic_stats("M")
        weekly_html = (
            f"""
        <div class="card p-4 mb-5">
            <h2>Weekly Performance Statistics</h2>
            <table class="metrics-table">
                <tr><th>Number of Weeks</th><td>{weekly.get('num_periods', 0)}</td></tr>
                <tr><th>Average Return</th><td>{weekly.get('avg_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>Median Return</th><td>{weekly.get('median_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>Std Dev</th><td>{weekly.get('std_return_pct', 0.0):.2f}%</td></tr>
                <tr><th>Best Week</th><td>{weekly.get('best_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>Worst Week</th><td>{weekly.get('worst_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>% Positive Weeks</th><td>{weekly.get('positive_pct', 0.0):.1f}%</td></tr>
                {f"<tr><th>Avg Trades/Week</th><td>{weekly.get('avg_trades', 0.0):.2f}</td></tr>" if 'avg_trades' in weekly else ""}
                {f"<tr><th>Max Trades/Week</th><td>{weekly.get('max_trades', 0)}</td></tr>" if 'max_trades' in weekly else ""}
                {f"<tr><th>% Weeks with Trades</th><td>{weekly.get('periods_with_trades_pct', 0.0):.1f}%</td></tr>" if 'periods_with_trades_pct' in weekly else ""}
            </table>
        </div>
        """
            if weekly
            else ""
        )

        monthly_stats_html = (
            f"""
        <div class="card p-4 mb-5">
            <h2>Monthly Performance Statistics</h2>
            <table class="metrics-table">
                <tr><th>Number of Months</th><td>{monthly.get('num_periods', 0)}</td></tr>
                <tr><th>Average Return</th><td>{monthly.get('avg_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>Median Return</th><td>{monthly.get('median_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>Std Dev</th><td>{monthly.get('std_return_pct', 0.0):.2f}%</td></tr>
                <tr><th>Best Month</th><td>{monthly.get('best_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>Worst Month</th><td>{monthly.get('worst_return_pct', 0.0):+.2f}%</td></tr>
                <tr><th>% Positive Months</th><td>{monthly.get('positive_pct', 0.0):.1f}%</td></tr>
                {f"<tr><th>Avg Trades/Month</th><td>{monthly.get('avg_trades', 0.0):.2f}</td></tr>" if 'avg_trades' in monthly else ""}
                {f"<tr><th>Max Trades/Month</th><td>{monthly.get('max_trades', 0)}</td></tr>" if 'max_trades' in monthly else ""}
                {f"<tr><th>% Months with Trades</th><td>{monthly.get('periods_with_trades_pct', 0.0):.1f}%</td></tr>" if 'periods_with_trades_pct' in monthly else ""}
            </table>
        </div>
        """
            if monthly
            else ""
        )

        monthly_table_html = "<p>No equity history available.</p>"
        yearly_table_html = "<p>No equity history available.</p>"
        if len(r.equity_history) > 1:
            df_eq = pd.DataFrame(r.equity_history, columns=["time", "mtm_equity"])
            df_eq["time"] = pd.to_datetime(df_eq["time"])
            df_eq = df_eq.set_index("time").sort_index()

            monthly_ret = df_eq["mtm_equity"].resample("ME").last().pct_change() * 100
            monthly_df = monthly_ret.dropna().reset_index()
            monthly_df["time"] = monthly_df["time"].dt.strftime("%Y-%m")
            monthly_df["mtm_equity"] = monthly_df["mtm_equity"].apply(
                lambda x: f"{x:+.2f}%"
            )
            monthly_table_html = monthly_df.to_html(
                index=False,
                classes="metrics-table",
                header=["Month", "Return (%)"],
                escape=False,
            )

            df_eq["year"] = df_eq.index.year
            yearly_groups = df_eq.groupby("year")
            yearly_rows = ""
            if len(yearly_groups) > 0:
                for year, group in yearly_groups:
                    if len(group) < 2:
                        continue
                    start_eq = group["mtm_equity"].iloc[0]
                    end_eq = group["mtm_equity"].iloc[-1]
                    ret_pct = (end_eq / start_eq - 1) * 100
                    year_end = pd.Timestamp(f"{year}-12-31")
                    is_partial = group.index[-1] < year_end
                    label = f"{year} (YTD)" if is_partial else str(year)
                    yearly_rows += f"<tr><td>{label}</td><td>{ret_pct:+.2f}%</td></tr>"
            yearly_table_html = f"""
            <table class="metrics-table">
                <thead><tr><th>Year</th><th>Return (%)</th></tr></thead>
                <tbody>{yearly_rows or "<tr><td colspan='2'>Insufficient data</td></tr>"}</tbody>
            </table>
            """

        returns_html = f"""
        <div class="card p-4 mb-5">
            <h2>Monthly Returns</h2>
            {monthly_table_html}
            <h2 class="mt-5">Yearly Returns</h2>
            {yearly_table_html}
        </div>
        """

        # Dynamic CSS — fully driven by the active Plotter theme
        dynamic_style = f"""
        <style>
            :root {{
                --bg-body: {theme["fig_bg"]};
                --bg-card: {theme["ax_bg"]};
                --text-main: {theme["text_color"]};
                --heading: {theme["mtm_color"]};
                --grid: {theme["grid_color"]};
                --danger: {theme["dd_line"]};
                --warning: {theme["danger_thresh"]};
                --highlight-strong: {theme["mtm_color"]};
            }}

            body {{
                background: var(--bg-body);
                color: var(--text-main);
                padding: 40px 0;
                font-family: 'Segoe UI', sans-serif;
            }}

            .container {{ max-width: 1400px; }}

            h1, h2, h3, h4 {{ color: var(--heading); }}

            .card {{
                background: var(--bg-card) !important;
                border: 1px solid var(--grid);
                margin-bottom: 40px;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            }}

            .highlight-card {{
                border: 2px solid var(--heading);
                box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            }}

            .highlight-card strong {{
                color: var(--highlight-strong);
                font-size: 1.1rem;
            }}

            .highlight-card .value {{
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--text-main);
            }}

            .highlight-card .value.large {{
                font-size: 1.8rem;
            }}

            img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.7);
                margin: 30px 0;
            }}

            .metrics-table, .metrics-table th, .metrics-table td {{
                background: transparent !important;
                border: none !important;
                padding: 10px 15px;
            }}

            .metrics-table th {{
                text-align: left;
                color: var(--heading);
                width: 45%;
            }}

            .metrics-table td {{ text-align: right; color: var(--text-main); }}

            .metrics-table tr {{ border-bottom: 1px solid var(--grid); }}
            .metrics-table tr:last-child {{ border-bottom: none; }}

            .text-warning {{ color: var(--warning) !important; }}
            .text-danger {{ color: var(--danger) !important; }}
        </style>
        """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>SEDA Backtest Report - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    {dynamic_style}
</head>
<body>
    <div class="container">
        <h1 class="text-center my-5">SEDA Backtest Performance Report</h1>

        {top_cards_html}

        {config_html}
        {summary_html}
        {detailed_html}
        {frequency_html}
        {tf_html}
        {risk_html}
        {execution_html}
        {leverage_impact_html}
        {trade_stats_html}
        {initial_html}
        {weekly_html}
        {monthly_stats_html}
        {returns_html}

        <div class="card">
            <h2>Performance Dashboard</h2>
            <img src="data:image/png;base64,{img_dashboard}">
        </div>

        <div class="card">
            <h2>Trade Activity Dashboard</h2>
            <img src="data:image/png;base64,{img_activity}">
        </div>

        <div class="card">
            <h2>Execution Impact Dashboard</h2>
            <img src="data:image/png;base64,{img_execution}">
        </div>

        <div class="card">
            <h2>Strategy Correlations</h2>
            <img src="data:image/png;base64,{img_correlations}">
        </div>
    </div>
</body>
</html>"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"\nHTML report successfully generated: {output_path}")

    def generate_report(
        self,
        format: str = "console",
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        if format == "console":
            self.generate_console_report(verbose=verbose)
        elif format == "html":
            self.generate_html_report(output_path or "backtest_report.html")