"""Tests for Plotter dashboards and theme handling.

Uses matplotlib's Agg backend + image comparison via pytest-mpl if available.
"""

import matplotlib.pyplot as plt
import pytest

from src.backtester.core.backtester import Backtester
from src.backtester.core.results import BacktestResults
from src.backtester.reporting.plotter import Plotter


@pytest.fixture
def plt_results():
    results = BacktestResults()
    dates = [plt.datetime.datetime(2023, 1, i) for i in range(1, 11)]
    results.equity_history = list(zip(dates, [100_000 + i*1000 for i in range(10)]))
    results.realized_equity_history = results.equity_history[:]
    results.open_notional_history = list(zip(dates, [0] * 10))
    results.all_trades = []  # simplified
    return results


@pytest.fixture
def plt_backtester():
    return Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSDT"],
        strategies=[],
        start_date=plt.datetime.datetime(2023, 1, 1),
        initial_capital=100_000,
    )


class TestPlotter:
    @pytest.mark.mpl_image_compare  # requires pytest-mpl
    def test_performance_dashboard(self, plt_results, plt_backtester):
        plotter = Plotter(plt_results, plt_backtester)
        fig = plotter.plot_performance_dashboard(return_fig=True)
        return fig

    @pytest.mark.mpl_image_compare
    def test_activity_dashboard(self, plt_results, plt_backtester):
        plotter = Plotter(plt_results, plt_backtester)
        fig = plotter.plot_activity_dashboard(return_fig=True)
        return fig

    @pytest.mark.mpl_image_compare
    def test_execution_dashboard(self, plt_results, plt_backtester):
        # Add some fake trade data for execution impact
        plt_results.all_trades = []  # minimal case
        plotter = Plotter(plt_results, plt_backtester)
        fig = plotter.plot_execution_dashboard(return_fig=True)
        return fig

    def test_theme_application(self, plt_results, plt_backtester):
        plotter = Plotter(plt_results, plt_backtester)
        fig = plotter.plot_performance_dashboard(return_fig=True)

        # Check background colours from default theme
        assert fig.get_facecolor() == (0, 0, 0, 0)  # transparent default, but axes bg should be set
        ax = fig.axes[0]
        assert ax.get_facecolor() != (1, 1, 1, 1)  # not white → dark theme applied

    def test_generate_all_plots_no_show(self, plt_results, plt_backtester, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        plotter = Plotter(plt_results, plt_backtester)
        plotter.generate_all_plots()  # should not raise