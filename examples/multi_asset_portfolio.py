"""
Multi-Asset / Multi-Timeframe Portfolio Demo

Demonstrates SEDA's core strength: true synchronous event-driven portfolio simulation
across multiple perpetual futures contracts, multiple timeframes, and multiple strategies
sharing the same equity pool.

Features showcased:
- Simultaneous BTCUSDT + ETHUSDT
- 1h and 4h timeframes (unified timeline)
- Two independent strategies running in parallel:
  • SMA Crossover (reversal mode)
  • RSI Mean Reversion (classic overbought/oversold)
- Shared capital, realistic execution, leverage caps, and liquidation tracking
- Portfolio-level metrics and a single combined HTML report

This example highlights how SEDA naturally handles complex, real-world portfolio
setups with correct chronological ordering and capital allocation.

Run directly:
    python examples/multi_asset_portfolio.py
"""

from datetime import datetime

from src.backtester.core.backtester import Backtester
from src.backtester.strategies.sma_crossover import SMACrossoverStrategy
from src.backtester.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.backtester.sizers.sizers import FixedNotionalSizer, KellyRiskSizer
from src.backtester.reporting.reporter import Reporter

if __name__ == "__main__":
    # Strategy 1: Classic SMA crossover (reversal system)
    sma_strategy = SMACrossoverStrategy(
        name="Portfolio SMA Crossover",
        fast_period=10,
        slow_period=50,
        sizer=FixedNotionalSizer(notional=20_000),  # $20k per signal
        long_only=True,
        short_only=False,
    )

    # Strategy 2: RSI mean reversion (overbought/oversold with reversal exits)
    rsi_strategy = RSIMeanReversionStrategy(
        name="Portfolio RSI Mean Reversion",
        rsi_period=14,
        oversold=30.0,
        overbought=70.0,
        sizer=KellyRiskSizer(kelly_fraction=0.25),  # Use quarter kelly
        long_only=False,
        short_only=True,
    )

    # Backtester configuration — multiple assets, timeframes, and strategies
    bt = Backtester(
        timeframes=["1h", "4h"],                    # Multi-timeframe
        asset_list=["BTCUSDT", "ETHUSDT"],          # Multi-asset
        strategies=[sma_strategy, rsi_strategy],   # Multi-strategy portfolio
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 1),
        initial_capital=200_000.0,
        leverage=5.0,                               # Realistic 5x cap
        mmr_rate=0.005,                             # 0.5% maintenance margin
        liq_mode="log_only",
    )

    print("\nRunning multi-asset / multi-timeframe / multi-strategy portfolio backtest...")
    results = bt.run()

    # Generate reports
    output_html = "../reports/multi_asset_portfolio_report.html"
    Reporter(results, bt).generate_report(format="html", output_path=output_html)
    Reporter(results, bt).generate_report(format="console", verbose=True)

    print(f"\nBacktest complete!")
    print(f"Open {output_html} for the full interactive portfolio report.")
    print(f"   • Total return: {results.total_return_pct:+.2f}%")
    print(f"   • CAGR: {results.cagr_pct:+.2f}%")
    print(f"   • Max drawdown: {results.max_drawdown_pct:.2f}%")
    print(f"   • Calmar ratio: {results.calmar_ratio:.2f}")
    print(f"   • Sharpe ratio: {results.sharpe_ratio:.2f}")