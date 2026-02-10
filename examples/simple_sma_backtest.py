"""
Quickstart Example – SMA Crossover on BTCUSDT (zero-config)

Run with:
    python examples/simple_sma_backtest.py

This script demonstrates the easiest way to get started:
• Auto-discovers sample_data/BTCUSDT_1h.csv
• Generates both console summary and full HTML report
"""

from datetime import datetime

from src.backtester.core.engine import Backtester
from src.backtester.reporting.reporter import Reporter
from src.backtester.strategies.sma_crossover import SMACrossoverStrategy
from src.backtester.sizers.sizers import FixedNotionalSizer

if __name__ == "__main__":
    # Simple SMA strategy
    strategy = SMACrossoverStrategy(
        name="SMA Crossover",
        fast_period=50,
        slow_period=200,
        sizer=FixedNotionalSizer(notional=1_000),
    )

    # Backtester auto-loads from sample_data/ using standard naming: {asset}_{timeframe}.csv
    # Adjust timeframe to match your sample CSV (e.g., "1h" for BTCUSDT_1h.csv)
    bt = Backtester(
        timeframes=[
            "1h"
        ],  # Change to match your sample file (e.g., "15m" if you have 15m data)
        asset_list=["BTCUSDT"],
        strategies=[strategy],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 7, 1),
        initial_capital=10_000,
        leverage=5.0,
    )

    results = bt.run()
    Reporter(results, bt).generate_report(
        format="console", output_path="../reports/simple_sma_report.html"
    )
    Reporter(results, bt).generate_report(
        format="html", output_path="../reports/simple_sma_report.html"
    )

    print("Backtest complete — open ../reports/simple_sma_report.html")
