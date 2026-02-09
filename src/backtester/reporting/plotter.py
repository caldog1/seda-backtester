"""Matplotlib-based visualization suite for SEDA backtest reports.

Generates four comprehensive dashboards with configurable dark/light themes:
• Performance Dashboard – equity curve, drawdown, PnL distribution, leverage, margin ratio
• Trade Activity Dashboard – concurrent open trades, per asset-timeframe trade counts
• Execution Impact Dashboard – slippage distributions, cumulative costs, per-strategy breakdown
• Strategy Correlations – placeholder for future multi-strategy correlation heatmap

All plots support:
• Multiple professional color themes (pastel, neon, dark_pro, light, vibrant)
• return_fig=True for seamless embedding in HTML reports
• Graceful handling of edge cases (no trades, no leverage, etc.)

Designed for both interactive exploration and production-grade reporting.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from src.backtester.core.backtester import Backtester
from src.backtester.core.results import BacktestResults


class Plotter:
    """Handles all Matplotlib plotting with configurable themes."""

    THEMES = {
        "pastel": {
            "style": "dark_background",
            "fig_bg": "#0e1117",
            "ax_bg": "#1a1f2e",
            "grid_color": "#444444",
            "text_color": "#e0e0e0",
            "mtm_color": "#8ECAE6",
            "realized_color": "#A8DADC",
            "leverage_color": "#C7CEEA",
            "dd_fill": "#FF9999",
            "dd_line": "#FF6B6B",
            "hist_color": "#A7C7E7",
            "hist_edge": "#6baed6",
            "breakeven": "#FFB6C1",
            "liq_thresh": "#FFB6C1",
            "danger_thresh": "#FFD580",
            "margin_color": "#C7CEEA",
        },
        "neon": {
            "style": "dark_background",
            "fig_bg": "#0e1117",
            "ax_bg": "#1a1f2e",
            "grid_color": "#444444",
            "text_color": "#e0e0e0",
            "mtm_color": "#00ffea",
            "realized_color": "#39ff14",
            "leverage_color": "#ff00ff",
            "dd_fill": "#ff2d55",
            "dd_line": "#ff6b6b",
            "hist_color": "#00d2ff",
            "hist_edge": "#0083b0",
            "breakeven": "#ff0048",
            "liq_thresh": "#ff2d55",
            "danger_thresh": "#ffa500",
            "margin_color": "#ff00ff",
        },
        "dark_pro": {
            "style": "dark_background",
            "fig_bg": "#121212",
            "ax_bg": "#1e1e1e",
            "grid_color": "#333333",
            "text_color": "#e0e0e0",
            "mtm_color": "#64b5f6",
            "realized_color": "#81c784",
            "leverage_color": "#ba68c8",
            "dd_fill": "#e57373",
            "dd_line": "#f44336",
            "hist_color": "#4fc3f7",
            "hist_edge": "#029be5",
            "breakeven": "#ef5350",
            "liq_thresh": "#ef5350",
            "danger_thresh": "#ff9800",
            "margin_color": "#ba68c8",
        },
        "light": {
            "style": "default",
            "fig_bg": "#ffffff",
            "ax_bg": "#f5f5f5",
            "grid_color": "#e0e0e0",
            "text_color": "#000000",
            "mtm_color": "#1976d2",
            "realized_color": "#388e3c",
            "leverage_color": "#7b1fa2",
            "dd_fill": "#e57373",
            "dd_line": "#d32f2f",
            "hist_color": "#42a5f5",
            "hist_edge": "#1976d2",
            "breakeven": "#d32f2f",
            "liq_thresh": "#d32f2f",
            "danger_thresh": "#f57c00",
            "margin_color": "#7b1fa2",
        },
        "vibrant": {
            "style": "dark_background",
            "fig_bg": "#0a0e17",
            "ax_bg": "#161b22",
            "grid_color": "#2d3748",
            "text_color": "#e0e0e0",
            "mtm_color": "#00bfff",
            "realized_color": "#00ff7f",
            "leverage_color": "#ff1493",
            "dd_fill": "#ff4757",
            "dd_line": "#ff3838",
            "hist_color": "#1e90ff",
            "hist_edge": "#0066cc",
            "breakeven": "#ff0066",
            "liq_thresh": "#ff1744",
            "danger_thresh": "#ff8c00",
            "margin_color": "#ff1493",
        },
    }

    def __init__(
        self,
        results: BacktestResults,
        backtester: Optional[Backtester] = None,
        theme: str = "light",
    ) -> None:
        self.results = results
        self.backtester = backtester
        self.theme = self.THEMES.get(theme.lower(), self.THEMES["pastel"])

    def _apply_theme(self, fig: plt.Figure) -> None:
        """Apply selected theme to figure and axes."""
        plt.style.use(self.theme["style"])
        fig.patch.set_facecolor(self.theme["fig_bg"])
        for ax in fig.axes:
            ax.set_facecolor(self.theme["ax_bg"])
            ax.tick_params(colors=self.theme["text_color"])
            ax.xaxis.label.set_color(self.theme["text_color"])
            ax.yaxis.label.set_color(self.theme["text_color"])
            ax.title.set_color(self.theme["text_color"])
            ax.grid(True, alpha=0.3, color=self.theme["grid_color"])

    def plot_performance_dashboard(
        self, return_fig: bool = False
    ) -> Optional[plt.Figure]:
        """Main performance dashboard: equity, drawdown, leverage, margin ratio."""
        if len(self.results.equity_history) <= 1:
            return None

        fig = plt.figure(figsize=(26, 22), constrained_layout=True)
        self._apply_theme(fig)
        fig.suptitle(
            "Backtest Performance Dashboard",
            fontsize=22,
            fontweight="bold",
            color=self.theme["text_color"],
        )

        # DataFrames
        df_eq = pd.DataFrame(
            self.results.equity_history, columns=["time", "mtm_equity"]
        )
        df_eq["time"] = pd.to_datetime(df_eq["time"])
        df_eq = df_eq.set_index("time")

        df_real = pd.DataFrame(
            self.results.realized_equity_history, columns=["time", "realized"]
        )
        df_real["time"] = pd.to_datetime(df_real["time"])
        df_real = df_real.set_index("time")

        df_notional = pd.DataFrame(
            self.results.open_notional_history, columns=["time", "notional"]
        )
        df_notional["time"] = pd.to_datetime(df_notional["time"])
        df_notional = df_notional.set_index("time")

        leverage = df_notional["notional"] / df_real["realized"].replace(0, np.nan)

        gs = GridSpec(3, 2, figure=fig)

        # Equity Curve
        ax_eq = fig.add_subplot(gs[0, 0])
        ax_eq.plot(
            df_eq.index,
            df_eq["mtm_equity"],
            label="MTM Equity",
            color=self.theme["mtm_color"],
            linewidth=2,
        )
        ax_eq.plot(
            df_real.index,
            df_real["realized"],
            label="Realized Equity",
            color=self.theme["realized_color"],
            linestyle="--",
            linewidth=2,
        )
        ax_eq.set_title("Equity Curve", fontsize=16, pad=20)
        ax_eq.set_ylabel("Equity ($)", fontsize=14)
        ax_eq.legend(frameon=True, fancybox=True, shadow=True)

        # Drawdown
        ax_dd = fig.add_subplot(gs[0, 1])
        mtm_dd = (df_eq["mtm_equity"] / df_eq["mtm_equity"].cummax() - 1) * 100
        ax_dd.fill_between(
            mtm_dd.index,
            mtm_dd,
            0,
            color=self.theme["dd_fill"],
            alpha=0.4,
            label="Drawdown Area",
        )
        ax_dd.plot(
            mtm_dd.index,
            mtm_dd,
            label="Drawdown (%)",
            color=self.theme["dd_line"],
            linewidth=2,
        )
        ax_dd.set_title("MTM Drawdown (%)", fontsize=16, pad=20)
        ax_dd.set_ylabel("Drawdown (%)", fontsize=14)
        ax_dd.legend(frameon=True, fancybox=True, shadow=True)

        # PnL Histogram
        ax_hist = fig.add_subplot(gs[1, 0])
        if self.results.trade_pnls:
            ax_hist.hist(
                self.results.trade_pnls,
                bins=50,
                color=self.theme["hist_color"],
                edgecolor=self.theme["hist_edge"],
                alpha=0.8,
                label="Trade PnL",
            )
            ax_hist.axvline(
                0,
                color=self.theme["breakeven"],
                linewidth=2,
                linestyle="--",
                label="Breakeven",
            )
            ax_hist.set_title("Trade PnL Distribution", fontsize=16, pad=20)
            ax_hist.set_xlabel("PnL ($)", fontsize=14)
            ax_hist.set_ylabel("Frequency", fontsize=14)
            ax_hist.legend(frameon=True, fancybox=True, shadow=True)
        else:
            ax_hist.text(
                0.5,
                0.5,
                "No Closed Trades",
                transform=ax_hist.transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
            ax_hist.set_title("Trade PnL Distribution", fontsize=16, pad=20)

        # Margin Ratio
        ax_margin = fig.add_subplot(gs[1, 1])
        if self.results.margin_ratio_history:
            ratio_series = pd.Series(
                [
                    r if not np.isinf(r) else np.nan
                    for _, r in self.results.margin_ratio_history
                ],
                index=df_eq.index,
            )
            ax_margin.plot(
                ratio_series.index,
                ratio_series,
                color=self.theme["margin_color"],
                linewidth=2,
                label="Margin Ratio",
            )
            ax_margin.axhline(
                1.0,
                color=self.theme["liq_thresh"],
                linewidth=2,
                linestyle="-",
                label="Liquidation Threshold",
            )
            danger_level = 1 + self.results.danger_buffer
            ax_margin.axhline(
                danger_level,
                color=self.theme["danger_thresh"],
                linewidth=2,
                linestyle="--",
                label="Danger Zone Threshold",
            )
            ax_margin.fill_between(
                ratio_series.index,
                1.0,
                danger_level,
                color=self.theme["danger_thresh"],
                alpha=0.15,
            )
            ax_margin.set_yscale("log")
            ax_margin.set_title("Margin Ratio (Log Scale)", fontsize=16, pad=20)
            ax_margin.set_ylabel("Margin Ratio (x)", fontsize=14)
            ax_margin.legend(frameon=True, fancybox=True, shadow=True)
        else:
            ax_margin.text(
                0.5,
                0.5,
                "No Leverage Cap",
                transform=ax_margin.transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
            ax_margin.set_title("Margin Ratio (Log Scale)", fontsize=16, pad=20)

        # Leverage
        ax_lev = fig.add_subplot(gs[2, 0])
        ax_lev.plot(
            leverage.index,
            leverage,
            color=self.theme["leverage_color"],
            linewidth=2,
            label="Effective Leverage",
        )
        if self.backtester and self.backtester.liq_tracker.leverage:
            ax_lev.axhline(
                self.backtester.liq_tracker.leverage,
                color=self.theme["danger_thresh"],
                linestyle="--",
                linewidth=2,
                label="Leverage Cap",
            )
        ax_lev.set_title("Effective Leverage", fontsize=16, pad=20)
        ax_lev.set_ylabel("Leverage (x)", fontsize=14)
        ax_lev.legend(frameon=True, fancybox=True, shadow=True)

        # Buy & Hold (placeholder)
        ax_bh = fig.add_subplot(gs[2, 1])
        ax_bh.text(
            0.5,
            0.5,
            "Buy & Hold Comparison\n(Not implemented)",
            transform=ax_bh.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
        ax_bh.set_title("Buy & Hold", fontsize=16, pad=20)

        if return_fig:
            return fig
        plt.show()
        return None

    def plot_activity_dashboard(self, return_fig: bool = False) -> Optional[plt.Figure]:
        """Trade activity dashboard: open trades, per-series counts."""
        if len(self.results.equity_history) <= 1:
            return None

        fig = plt.figure(figsize=(28, 30))
        self._apply_theme(fig)
        fig.suptitle(
            "Trade Activity Dashboard",
            fontsize=22,
            fontweight="bold",
            color=self.theme["text_color"],
        )

        gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.4], hspace=0.5)

        # Concurrent open trades
        df_open = pd.DataFrame(
            self.results.open_trade_history, columns=["time", "count"]
        )
        df_open["time"] = pd.to_datetime(df_open["time"])
        df_open = df_open.set_index("time")

        ax_count = fig.add_subplot(gs[0, 0])
        ax_count.step(
            df_open.index,
            df_open["count"],
            where="post",
            color="#9C27B0",
            linewidth=2,
            label="Open Trades",
        )
        ax_count.fill_between(
            df_open.index,
            0,
            df_open["count"],
            where=df_open["count"] > 0,
            color="#9C27B0",
            alpha=0.15,
            label="Active Period",
        )
        ax_count.set_title("Concurrent Open Trades", fontsize=16, pad=20)
        ax_count.set_ylabel("Open Trades", fontsize=14)
        ax_count.legend(frameon=True, fancybox=True, shadow=True)

        # Per-series trade counts
        ax_bar = fig.add_subplot(gs[1, 0])
        if self.results.per_series_metrics:
            labels = [
                f"{a.upper()}-{tf.upper()}"
                for (a, tf), m in self.results.per_series_metrics.items()
            ]
            counts = [
                m["total_trades"] for m in self.results.per_series_metrics.values()
            ]
            sorted_idx = np.argsort(counts)[::-1]
            bars = ax_bar.bar(
                np.array(labels)[sorted_idx],
                np.array(counts)[sorted_idx],
                color=self.theme["mtm_color"],
                alpha=0.85,
            )
            ax_bar.set_title("Total Trades per Asset-Timeframe", fontsize=16, pad=20)
            ax_bar.set_ylabel("Trade Count", fontsize=14)
            ax_bar.tick_params(axis="x", rotation=60)
            # Simple legend for bar chart
            ax_bar.legend(
                [bars[0]], ["Trade Count"], frameon=True, fancybox=True, shadow=True
            )
        else:
            ax_bar.text(
                0.5,
                0.5,
                "No Trades",
                transform=ax_bar.transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
            ax_bar.set_title("Total Trades per Asset-Timeframe", fontsize=16, pad=20)

        # Placeholder for per-TF equity (future extension)
        ax_tf = fig.add_subplot(gs[2, 0])
        ax_tf.text(
            0.5,
            0.5,
            "Per-Timeframe Realized Equity\n(Not implemented yet)",
            transform=ax_tf.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
        ax_tf.set_title("Realized Equity by Timeframe", fontsize=16, pad=20)

        if return_fig:
            return fig
        plt.show()
        return None

    def plot_execution_dashboard(
        self, return_fig: bool = False
    ) -> Optional[plt.Figure]:
        """6-panel execution impact dashboard."""
        if not self.results.all_trades:
            fig, ax = plt.subplots(figsize=(20, 12))
            self._apply_theme(fig)
            ax.text(
                0.5,
                0.5,
                "No trades executed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=20,
            )
            ax.set_title("Execution Impact Dashboard", fontsize=16, pad=20)
            if return_fig:
                return fig
            plt.show()
            return None

        fig = plt.figure(figsize=(24, 18))
        self._apply_theme(fig)
        fig.suptitle(
            "Execution Impact Dashboard",
            fontsize=24,
            fontweight="bold",
            color=self.theme["text_color"],
        )

        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Entry slippage
        ax1 = fig.add_subplot(gs[0, 0])
        entry_bps = [t.entry_slippage_bps or 0 for t in self.results.all_trades]
        ax1.hist(
            entry_bps,
            bins=50,
            color="#00ffea",
            edgecolor="black",
            alpha=0.8,
            label="Entry Slippage",
        )
        ax1.set_title("Entry Slippage Distribution (bps)", fontsize=16)
        ax1.set_xlabel("Adverse Slippage (bps)")
        ax1.set_ylabel("Trade Count")
        ax1.legend(frameon=True, fancybox=True, shadow=True)

        # Exit slippage
        ax2 = fig.add_subplot(gs[0, 1])
        exit_bps = [t.exit_slippage_bps or 0 for t in self.results.all_trades]
        ax2.hist(
            exit_bps,
            bins=50,
            color="#ff00ff",
            edgecolor="black",
            alpha=0.8,
            label="Exit Slippage",
        )
        ax2.set_title("Exit Slippage Distribution (bps)", fontsize=16)
        ax2.set_xlabel("Adverse Slippage (bps)")
        ax2.set_ylabel("Trade Count")
        ax2.legend(frameon=True, fancybox=True, shadow=True)

        # Cumulative cost
        ax3 = fig.add_subplot(gs[1, :])
        trades_sorted = sorted(self.results.all_trades, key=lambda t: t.entry_time)
        cum_cost = np.cumsum(
            [
                t.entry_slippage_dollar
                + t.exit_slippage_dollar
                + t.entry_fee_dollar
                + t.exit_fee_dollar
                for t in trades_sorted
            ]
        )
        times = [t.entry_time for t in trades_sorted]
        ax3.plot(
            times,
            cum_cost,
            color="#ff6b6b",
            linewidth=2,
            label="Cumulative Execution Cost",
        )
        ax3.set_title("Cumulative Execution Cost Over Time ($)", fontsize=16)
        ax3.set_ylabel("Cumulative Cost ($)")
        ax3.legend(frameon=True, fancybox=True, shadow=True)

        # Per-strategy breakdown
        ax4 = fig.add_subplot(gs[2, 0])
        strats = sorted(
            set(t.strategy_name for t in self.results.all_trades if t.strategy_name)
        )
        fees = []
        slips = []
        for s in strats:
            strat_trades = [t for t in self.results.all_trades if t.strategy_name == s]
            fees.append(
                sum(t.entry_fee_dollar + t.exit_fee_dollar for t in strat_trades)
            )
            slips.append(
                sum(
                    t.entry_slippage_dollar + t.exit_slippage_dollar
                    for t in strat_trades
                )
            )
        x = np.arange(len(strats))
        bar1 = ax4.bar(x - 0.2, fees, 0.4, label="Fees", color="#64b5f6")
        bar2 = ax4.bar(x + 0.2, slips, 0.4, label="Slippage", color="#e57373")
        ax4.set_xticks(x)
        ax4.set_xticklabels(strats, rotation=45)
        ax4.set_title("Per-Strategy Execution Cost Breakdown ($)", fontsize=16)
        ax4.set_ylabel("Total Cost ($)")
        ax4.legend(frameon=True, fancybox=True, shadow=True)

        # Slippage vs Notional
        ax5 = fig.add_subplot(gs[2, 1])
        notionals = [t.trade_value for t in self.results.all_trades]
        avg_slips = [
            (t.entry_slippage_bps + t.exit_slippage_bps) / 2
            for t in self.results.all_trades
        ]
        scatter = ax5.scatter(notionals, avg_slips, alpha=0.6, color="#ba68c8")
        ax5.set_title("Average Slippage vs Trade Notional", fontsize=16)
        ax5.set_xlabel("Notional ($)")
        ax5.set_ylabel("Avg Slippage (bps)")
        ax5.legend(
            [scatter], ["Average Slippage"], frameon=True, fancybox=True, shadow=True
        )

        if return_fig:
            return fig
        plt.show()
        return None

    def plot_correlations(self, return_fig: bool = False) -> Optional[plt.Figure]:
        """Strategy correlation dashboard (placeholder for future implementation)."""
        fig = plt.figure(figsize=(20, 12))
        self._apply_theme(fig)
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Strategy Correlations\n(Not implemented yet)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=20,
        )
        ax.set_title("Strategy Correlations", fontsize=16)

        if return_fig:
            return fig
        plt.show()
        return None

    def generate_all_plots(self) -> None:
        """Show all dashboards sequentially."""
        self.plot_performance_dashboard()
        self.plot_activity_dashboard()
        self.plot_execution_dashboard()
        self.plot_correlations()
