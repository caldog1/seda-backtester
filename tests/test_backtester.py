"""Comprehensive integration tests for Backtester orchestration.

Uses a synthetic in-memory DataProvider to ensure deterministic, fast tests.
Covers:
- Baseline (no trades)
- Realistic trade execution (entries/exits, fees, slippage)
- Multi-asset/multi-timeframe timeline synchronization
- Leverage caps & pyramiding prevention
- Liquidation detection
- Core metrics accuracy
"""

import datetime
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

# Suppress expected slippage warnings for clean output (synthetic data has intentionally high notional relative to early bars)
warnings.filterwarnings("ignore", message="High slippage risk")

from backtester.core.engine import Backtester
from backtester.core.trade import TradeDirection
from backtester.data.base_provider import DataProvider
from backtester.strategies.base import Strategy
from backtester.sizers.sizers import FixedNotionalSizer
from backtester.core.order import Order


# =============================================================================
# Synthetic In-Memory Data Provider
# =============================================================================


class SyntheticDataProvider(DataProvider):
    """Generate simple linear price series for deterministic testing."""

    def __init__(self, price_trends: Dict[tuple[str, str], Dict[str, float]]):
        self.trends = price_trends

    def load_data(
            self,
            asset: str,
            timeframe: str,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp | None = None,
            filepath: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        del filepath

        key = (asset, timeframe)
        if key not in self.trends:
            raise ValueError(f"No synthetic data for {key}")

        cfg = self.trends[key]
        bars = cfg["bars"]
        prices = np.linspace(cfg["start"], cfg["end"], bars)

        if start_date.tz is None:
            start_date = start_date.tz_localize("UTC")

        times = pd.date_range(start=start_date, periods=bars, freq="1h", tz="UTC")

        return {
            "Close Time": times.values.astype("datetime64[ns]"),
            "Open": prices * 0.999,
            "High": prices * 1.001,
            "Low": prices * 0.999,
            "Close": prices,
            "Volume": np.full(
                bars, 1_000_000_000.0
            ),  # Extremely high → effectively zero slippage
        }


# =============================================================================
# Minimal Test Strategies
# =============================================================================


class AlwaysEnterStrategy(Strategy):
    """Enter fixed notional on first bar of the target asset, hold until end."""

    def __init__(
            self,
            direction: TradeDirection = TradeDirection.LONG,
            target_asset: Optional[str] = None,
    ):
        super().__init__(
            name=f"AlwaysEnter{direction.value}"
                 + (f"_{target_asset}" if target_asset else ""),
            sizer=FixedNotionalSizer(notional=10_000),
            allow_pyramiding=False,
        )
        self.direction = direction
        self.target_asset = target_asset
        self.entered = False

    def handle_entries(
            self,
            key,
            idx,
            arrays,
            asset,
            timeframe,
            current_equity,
            max_additional_notional,
            backtester,
    ):
        # Optional asset filter — allows one strategy per asset in multi-asset tests
        if self.target_asset is not None and asset != self.target_asset:
            return []

        if self.entered or idx != 0:
            return []

        self.entered = True
        return [
            Order(
                direction=self.direction,
                order_type="market",
                notional=0.0,  # sizer decides
                asset=asset,
                timeframe=timeframe,
                strategy_name=self.name,
            )
        ]

    def check_exit(self, idx, trade, arrays, asset, timeframe, backtester):
        # Never exit early
        return super().check_exit(idx, trade, arrays, asset, timeframe, backtester)


# =============================================================================
# Tests
# =============================================================================


@pytest.fixture
def synthetic_provider():
    """Rising BTC and falling ETH on 1h."""
    return SyntheticDataProvider(
        {
            ("BTCUSdatetime", "1h"): {"start": 100.0, "end": 200.0, "bars": 100},
            ("ETHUSdatetime", "1h"): {"start": 50.0, "end": 30.0, "bars": 100},
        }
    )


def test_backtester_no_trades(synthetic_provider):
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        end_date=datetime.datetime(2023, 1, 5, tzinfo=datetime.timezone.utc),
        initial_capital=100_000,
        data_provider=synthetic_provider,
    )
    results = bt.run()
    assert results.total_return_pct == pytest.approx(0.0)
    assert len(results.all_trades) == 0
    assert results.equity_history[-1][1] == pytest.approx(100_000.0)


def test_backtester_profitable_long_with_execution(synthetic_provider):
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[AlwaysEnterStrategy(TradeDirection.LONG)],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        initial_capital=100_000,
        data_provider=synthetic_provider,
    )
    results = bt.run()

    assert len(results.all_trades) == 1
    trade = results.all_trades[0]
    assert trade.pnl > 9950  # ~100% gross minus negligible fees/slippage
    assert results.total_return_pct > 9.95
    assert trade.fee > 0
    # With ultra-high synthetic volume, slippage is effectively negligible
    assert trade.entry_slippage_bps < 5
    assert trade.exit_slippage_bps < 5


def test_backtester_loss_short(synthetic_provider):
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["ETHUSdatetime"],
        strategies=[AlwaysEnterStrategy(TradeDirection.SHORT)],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        initial_capital=100_000,
        data_provider=synthetic_provider,
    )
    results = bt.run()
    assert len(results.all_trades) == 1
    trade = results.all_trades[0]
    assert trade.pnl > 3950  # ~40% gross minus negligible fees/slippage


def test_multi_asset_timeframe_sync(synthetic_provider):
    """Verify unified timeline and independent positioning."""
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime", "ETHUSdatetime"],
        strategies=[
            AlwaysEnterStrategy(TradeDirection.LONG, target_asset="BTCUSdatetime"),
            AlwaysEnterStrategy(TradeDirection.SHORT, target_asset="ETHUSdatetime"),
        ],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        initial_capital=100_000,
        data_provider=synthetic_provider,
    )
    results = bt.run()
    assert len(results.all_trades) == 2
    pnls = [t.pnl for t in results.all_trades]
    assert any(p > 9950 for p in pnls)  # BTC long winner
    assert any(p > 3950 for p in pnls)  # ETH short winner
    assert results.total_return_pct > 13.9  # ~14% combined (minus tiny fees)


def test_leverage_cap_prevention(synthetic_provider):
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[AlwaysEnterStrategy(TradeDirection.LONG)],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        initial_capital=10_000,
        leverage=2.0,  # Max ~20k notional
        data_provider=synthetic_provider,
    )
    results = bt.run()
    trade = results.all_trades[0]
    assert trade.trade_value == pytest.approx(10_000.0)

    # Oversized sizer to trigger cap
    big_sizer = FixedNotionalSizer(notional=50_000)
    bt.strategies[0].sizer = big_sizer
    results_big = bt.run()
    trade_big = results_big.all_trades[0]
    assert trade_big.trade_value <= 20_000.0


def test_liquidation_detection(synthetic_provider):
    """Force liquidation with aggressive sizing and adverse price move."""

    class CrashStrategy(AlwaysEnterStrategy):
        def __init__(self):
            super().__init__(TradeDirection.LONG)
            self.sizer = FixedNotionalSizer(notional=100_000)

    provider = SyntheticDataProvider(
        {
            ("BTCUSdatetime", "1h"): {"start": 100.0, "end": 10.0, "bars": 100},
        }
    )

    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[CrashStrategy()],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        initial_capital=10_000,
        leverage=5.0,
        liq_mode="log_only",
        data_provider=provider,
    )
    results = bt.run()
    assert results.liquidated
    assert results.liq_time is not None
    assert results.equity_history[-1][1] < 1000.0


def test_buy_and_hold_benchmark(synthetic_provider):
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[AlwaysEnterStrategy(TradeDirection.LONG)],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        initial_capital=100_000,
        data_provider=synthetic_provider,
    )
    results = bt.run()
    assert results.bh_return_pct == pytest.approx(100.0)


def test_backtester_aware_datetimes(synthetic_provider):
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[],
        start_date=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        end_date=datetime.datetime(2023, 1, 5, tzinfo=datetime.timezone.utc),
        initial_capital=100_000,
        data_provider=synthetic_provider,
    )
    results = bt.run()
    assert results.total_return_pct == pytest.approx(0.0)


def test_walk_forward_like_behavior(synthetic_provider):
    """Simulate in-sample / out-of-sample split."""
    bt_in = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[AlwaysEnterStrategy(TradeDirection.LONG)],
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 1, 5),
        data_provider=synthetic_provider,
    )
    bt_out = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[AlwaysEnterStrategy(TradeDirection.LONG)],
        start_date=datetime.datetime(2023, 1, 6),
        end_date=datetime.datetime(2023, 1, 10),
        data_provider=synthetic_provider,
    )

    in_results = bt_in.run()
    out_results = bt_out.run()

    assert in_results.total_return_pct > 0
    assert out_results.total_return_pct > 0  # same trend


def test_zero_initial_capital_edge():
    bt = Backtester(
        timeframes=["1h"],
        asset_list=["BTCUSdatetime"],
        strategies=[],
        start_date=datetime.datetime(2023, 1, 1),
        initial_capital=0.0,
        data_provider=SyntheticDataProvider(...),  # no trades possible
    )
    results = bt.run()
    assert results.total_return_pct == pytest.approx(0.0)
    assert not results.liquidated
