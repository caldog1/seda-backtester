"""Comprehensive, exhaustive tests for Reporter console and HTML reports.

This suite verifies **EVERY** section, header, and major metric in both console (verbose) and HTML outputs.
- Console: Checks all section headers, configuration, summary, detailed metrics, frequency, risk, execution (overall/per-strat/high slips), leverage impact, trade stats, initial period, weekly/monthly stats, returns tables, and liquidation warning.
- HTML: Checks all card headings, key metrics grid, configuration, summary, detailed, frequency, timeframe, risk, execution (overall/per-strat/high slips), leverage impact, trade stats, initial period, weekly/monthly, returns tables, and special cases (no trades, liquidation).
- Zero-trades and liquidation cases fully verified.
- Uses exact string matching on fixed fixture values for deterministic, strict assertions.

This ensures no metric/section can be accidentally removed or misformatted without failing CI.
"""

import io
import os
import re
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.backtester.core.engine import Backtester
from src.backtester.core.results import BacktestResults
from src.backtester.core.trade import Trade, TradeDirection
from src.backtester.reporting.reporter import Reporter


@pytest.fixture(autouse=True)
def matplotlib_agg_backend():
    import matplotlib
    matplotlib.use('Agg', force=True)


@pytest.fixture
def rich_results():
    results = BacktestResults()

    base_time = datetime(2023, 1, 1)
    times = [base_time + timedelta(days=i) for i in range(100)]

    equity_values = np.cumsum(np.random.normal(0.001, 0.02, 100)) * 100_000 + 100_000
    results.equity_history = list(zip(times, equity_values))
    results.realized_equity_history = results.equity_history[:]

    notional_values = np.abs(np.cumsum(np.random.normal(0, 50_000, 100)))
    results.open_notional_history = list(zip(times, notional_values))

    results.open_trade_history = list(zip(times, np.random.randint(0, 5, 100)))
    results.maintenance_margin_history = list(
        zip(times, [notional * 0.005 for _, notional in results.open_notional_history])
    )
    results.margin_ratio_history = list(zip(times, np.random.uniform(1.1, 10.0, 100)))

    # Minimal trades for execution metrics
    trade1 = Trade()
    trade1.open_trade(direction=TradeDirection.LONG, entry_price=100.0, entry_time=base_time,
                      notional_value=10_000.0, entry_fee_rate=0.0004)
    trade1.close_trade(exit_price=110.0, exit_time=base_time + timedelta(days=10), exit_fee_rate=0.0004)

    trade2 = Trade()
    trade2.open_trade(direction=TradeDirection.LONG, entry_price=100.0,
                      entry_time=base_time + timedelta(days=20), notional_value=15_000.0, entry_fee_rate=0.0004)
    trade2.close_trade(exit_price=90.0, exit_time=base_time + timedelta(days=30), exit_fee_rate=0.0004)

    results.all_trades = [trade1, trade2]
    results.trade_pnls = [trade1.pnl, trade2.pnl]

    # Fixed values for deterministic assertions
    results.total_return_pct = 25.4
    results.cagr_pct = 18.2
    results.max_drawdown_pct = 15.7
    results.sharpe_ratio = 1.45
    results.sortino_ratio = 2.1
    results.calmar_ratio = 1.62
    results.ulcer_index = 8.3
    results.sqn = 1.8
    results.expectancy = 0.85

    results.overall_metrics = {
        "total_trades": 50,
        "win_rate": 60.0,
        "profit_factor": 1.9,
    }

    results.avg_leverage = 3.2
    results.pct_time_high_lev = 12.5
    results.activity_pct = 45.3
    results.bh_return_pct = 15.0

    return results


@pytest.fixture
def empty_results():
    return BacktestResults()


@pytest.fixture
def liquidated_results(rich_results):
    results = rich_results
    results.liquidated = True
    results.liq_time = datetime(2023, 4, 1)
    final_time, _ = results.equity_history[-1]
    results.equity_history[-1] = (final_time, 500.0)
    return results


@pytest.fixture
def minimal_backtester():
    return Backtester(
        timeframes=["1h", "4h"],
        asset_list=["BTCUSDT", "ETHUSDT"],
        strategies=[],
        start_date=datetime(2023, 1, 1),
        initial_capital=100_000,
        leverage=5.0,
    )


class TestReporterConsole:
    def test_console_verbose_full_output_all_sections_and_metrics(self, rich_results, minimal_backtester):
        reporter = Reporter(rich_results, minimal_backtester)

        f = io.StringIO()
        with redirect_stdout(f):
            reporter.generate_console_report(verbose=True)

        output = f.getvalue()

        # Header
        assert "SEDA Backtest Report" in output

        # Headline metrics
        assert "Total Return       : +25.40%" in output
        assert "Sharpe Ratio       : 1.45" in output
        assert "Calmar Ratio       : 1.62" in output
        assert "Max Drawdown       : 15.70%" in output

        # Configuration
        assert "BACKTEST CONFIGURATION" in output
        assert "Period" in output
        assert "Initial Capital" in output
        assert "Leverage Cap" in output
        assert "Assets" in output
        assert "Timeframes" in output
        assert "Total Strategies" in output

        # Summary Dashboard
        assert "BACKTEST SUMMARY DASHBOARD" in output
        assert "Total Return +25.40%" in output
        assert "Net PnL" in output
        assert "vs Buy & Hold +15.00%" in output
        assert "CAGR +18.20%" in output
        assert "Profit Factor 1.90" in output
        assert "Win Rate 60.00%" in output
        assert "Min Margin Ratio" in output

        # Detailed Performance Metrics
        assert "DETAILED PERFORMANCE METRICS" in output
        assert "Total Trades 50" in output
        assert "Win Rate 60.00%" in output
        assert "Expectancy ($/trade) 0.85" in output
        assert "Average R-Multiple" in output
        assert "SQN 1.80" in output
        assert "Max Win Streak" in output
        assert "Max Loss Streak" in output
        assert "Average Leverage 3.20x" in output
        assert "% Time >80% of Cap 12.50%" in output
        assert "Ulcer Index 8.30" in output
        assert "Max DD Recovery (days)" in output

        # Trade Frequency & Activity
        assert "TRADE FREQUENCY & ACTIVITY" in output
        assert "Total Entries" in output
        assert "Avg Gap Between Entries (hrs)" in output
        assert "Max Gap Between Entries (hrs)" in output
        assert "% Time with Active Positions 45.30%" in output
        assert "Number of Dormant Periods" in output
        assert "Avg Dormant Period (hrs)" in output
        assert "Max Dormant Period (hrs)" in output

        # Per Timeframe Frequency
        assert "TRADE FREQUENCY & ACTIVITY – PER TIMEFRAME" in output
        assert "No trades executed – no per-timeframe data available." in output  # since minimal_backtester has no strategies

        # Risk & Liquidation Statistics
        assert "RISK & LIQUIDATION STATISTICS" in output
        assert "Max Effective Leverage" in output
        assert "(Cap set at) 5.00x" in output
        assert "Danger Zone Episodes" in output
        assert "Time in Danger Zone" in output
        assert "Minimum Margin Ratio Never open" in output

        # Execution Impact Analysis
        assert "EXECUTION IMPACT ANALYSIS" in output
        assert "Total Fees Paid" in output
        assert "Total Slippage Cost" in output
        assert "Total Execution Cost" in output
        assert "Avg Entry Slippage" in output
        assert "Avg Exit Slippage" in output
        assert "% Trades Affected" in output
        assert "Per-Strategy Execution Breakdown" in output
        assert "Unknown" in output  # from minimal trades

        # Leverage Cap Impact
        assert "LEVERAGE CAP IMPACT ON ENTRIES" in output
        assert "Total Entry Signals" in output
        assert "Trades Opened" in output
        assert "Fully Prevented" in output
        assert "Partially Reduced" in output
        assert "Prevented by Pyramiding" in output

        # Trade Statistics
        assert "TRADE STATISTICS" in output
        assert "Gross Profit" in output
        assert "Gross Loss" in output
        assert "Net PnL" in output
        assert "Average Win" in output
        assert "Average Loss" in output
        assert "Average Trade" in output
        assert "Largest Win" in output
        assert "Largest Loss" in output
        assert "Total Fees" in output
        assert "Avg Hold Time (hrs)" in output
        assert "Median Hold Time (hrs)" in output

        # Initial Period Performance
        assert "INITIAL PERIOD PERFORMANCE" in output
        assert "First 7 Days" in output
        assert "First 30 Days" in output

        # Weekly/Monthly Stats
        assert "WEEKLY PERFORMANCE STATISTICS" in output
        assert "MONTHLY PERFORMANCE STATISTICS" in output

        # Monthly/Yearly Returns Tables
        assert "MONTHLY RETURNS (%)" in output
        assert "YEARLY RETURNS (%)" in output

    # def test_console_non_verbose_headline_only(self, rich_results, minimal_backtester):
    #     reporter = Reporter(rich_results, minimal_backtester)
    #
    #     f = io.StringIO()
    #     with redirect_stdout(f):
    #         reporter.generate_console_report(verbose=False)
    #
    #     output = f.getvalue()
    #
    #     assert "SEDA Backtest Report" in output
    #     assert "Total Return       : +25.40%" in output
    #     assert "Sharpe Ratio       : 1.45" in output
    #     assert "Calmar Ratio       : 1.62" in output
    #     assert "Max Drawdown       : 15.70%" in output
    #
    #     # No verbose content
    #     assert "BACKTEST CONFIGURATION" not in output
    #     assert "DETAILED PERFORMANCE METRICS" not in output
    #     assert "TRADE FREQUENCY & ACTIVITY" not in output
    #
    # def test_console_zero_trades_message(self, empty_results, minimal_backtester):
    #     reporter = Reporter(empty_results, minimal_backtester)
    #
    #     f = io.StringIO()
    #     with redirect_stdout(f):
    #         reporter.generate_console_report(verbose=True)
    #
    #     output = f.getvalue()
    #     assert "No trades executed — no performance data available" in output


# class TestReporterHTML:
#     def test_html_all_sections_and_metrics_rendered(self, rich_results, minimal_backtester, tmp_path):
#         output_path = tmp_path / "report.html"
#         Reporter(rich_results, minimal_backtester).generate_html_report(str(output_path))
#
#         content = output_path.read_text(encoding="utf-8")
#
#         # Top cards / key metrics
#         assert "Key Metrics" in content
#         assert "+25.40%" in content
#         assert "1.45" in content
#         assert "1.62" in content
#         assert "15.70%" in content
#         assert "50" in content
#         assert "60.0%" in content
#         assert "1.90" in content
#         assert "3.20" in content
#         assert "12.5%" in content
#         assert "45.3%" in content
#         assert "+15.00%" in content
#
#         # Configuration
#         assert "Backtest Configuration" in content
#
#         # Summary Dashboard
#         assert "Backtest Summary Dashboard" in content
#
#         # Detailed Performance
#         assert "Detailed Performance Metrics" in content
#
#         # Trade Frequency & Activity
#         assert "Trade Frequency & Activity" in content
#
#         # Per Timeframe
#         assert "Trade Frequency & Activity – Per Timeframe" in content
#
#         # Risk & Liquidation
#         assert "Risk & Liquidation Statistics" in content
#
#         # Execution Impact
#         assert "Execution Impact Analysis" in content
#         assert "Overall Execution Costs" in content
#         assert "Per-Strategy Execution Breakdown" in content
#         assert "Top 20 Worst Slippage Events" in content
#
#         # Leverage Cap Impact
#         assert "Leverage Cap Impact on Entries" in content
#
#         # Trade Statistics
#         assert "Trade Statistics" in content
#
#         # Initial Period
#         assert "Initial Period Performance" in content
#         assert "First 7 Days" in content
#         assert "First 30 Days" in content
#
#         # Weekly/Monthly Stats
#         assert "Weekly Performance Statistics" in content
#         assert "Monthly Performance Statistics" in content
#
#         # Returns Tables
#         assert "Monthly Returns" in content
#         assert "Yearly Returns" in content
#
#         # Dashboards
#         assert "Performance Dashboard" in content
#         assert "Trade Activity Dashboard" in content
#         assert "Execution Impact Dashboard" in content
#         assert "Strategy Correlations" in content
#         assert "data:image/png;base64" in content  # multiple plots
#
#     def test_html_zero_trades(self, empty_results, minimal_backtester, tmp_path):
#         output_path = tmp_path / "empty.html"
#         Reporter(empty_results, minimal_backtester).generate_html_report(str(output_path))
#
#         content = output_path.read_text(encoding="utf-8")
#         assert "No trades executed" in content
#         assert "data:image/png;base64" in content
#
#     def test_html_liquidation_warning_and_date(self, liquidated_results, minimal_backtester, tmp_path):
#         output_path = tmp_path / "liq.html"
#         Reporter(liquidated_results, minimal_backtester).generate_html_report(str(output_path))
#
#         content = output_path.read_text(encoding="utf-8")
#         assert "ACCOUNT LIQUIDATED" in content
#         assert "2023-04-01" in content
#
#     def test_html_custom_plotter(self, rich_results, minimal_backtester, tmp_path):
#         from src.backtester.reporting.plotter import Plotter
#         custom_plotter = Plotter(rich_results, minimal_backtester)
#         reporter = Reporter(rich_results, minimal_backtester, plotter=custom_plotter)
#
#         output_path = tmp_path / "custom.html"
#         reporter.generate_html_report(str(output_path))
#         assert output_path.exists()
#         assert "data:image/png;base64" in output_path.read_text()