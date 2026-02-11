"""Comprehensive tests for LiquidationTracker.

Covers:
- No leverage (unlimited notional)
- Leverage cap calculation (various equity/open notional combinations)
- Liquidation trigger boundaries (strict < maintenance margin)
- Different modes ("stop" vs "log_only")
- Post-liquidation state persistence (flag remains set, time preserved)
- MMR rate variations
- Edge cases (zero notional, negative equity, already liquidated)
- Verbose output sanity (no crash)
"""

import datetime
from typing import Dict, List

import pytest

from backtester.core.liquidation_tracker import LiquidationTracker
from backtester.core.trade import Trade, TradeDirection


@pytest.fixture
def dummy_ongoing_trades() -> Dict[tuple[str, str], List[Trade]]:
    """Realistic ongoing trades (no price movement → unreal PnL = 0)."""
    trade1 = Trade()
    trade1.open_trade(
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        notional_value=50_000,
        entry_order_type="market",
        entry_fee_rate=0.0004,
    )
    trade1.asset = "BTCUSdatetime"
    trade1.timeframe = "1h"

    trade2 = Trade()
    trade2.open_trade(
        direction=TradeDirection.SHORT,
        entry_price=2000.0,
        entry_time=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
        notional_value=30_000,
        entry_order_type="market",
        entry_fee_rate=0.0004,
    )
    trade2.asset = "ETHUSdatetime"
    trade2.timeframe = "1h"

    return {
        ("BTCUSdatetime", "1h"): [trade1],
        ("ETHUSdatetime", "1h"): [trade2],
    }


class TestLiquidationTracker:
    @pytest.mark.parametrize("leverage", [None, float("inf")])
    def test_no_leverage_unlimited_notional(self, leverage):
        tracker = LiquidationTracker(leverage=leverage)
        max_additional = tracker.get_max_additional_notional(10_000, 50_000)
        assert max_additional == float("inf")

        tracker.check_and_handle_liquidation(
            mtm_equity=5000,
            open_notional=100_000,
            current_time=datetime.datetime.now(datetime.timezone.utc),
            current_equity=10_000,
            ongoing_trades={},
        )
        assert not tracker.liquidated
        assert tracker.liq_time is None

    @pytest.mark.parametrize(
        "equity, open_notional, leverage, expected_max",
        [
            (10_000, 0, 5.0, 50_000.0),
            (10_000, 30_000, 5.0, 20_000.0),
            (10_000, 50_000, 5.0, 0.0),
            (10_000, 60_000, 5.0, 0.0),
            (20_000, 50_000, 3.0, 10_000.0),
            (15_000, 0, 10.0, 150_000.0),
        ],
    )
    def test_leverage_cap_calculation(
            self, equity, open_notional, leverage, expected_max
    ):
        tracker = LiquidationTracker(leverage=leverage)
        max_additional = tracker.get_max_additional_notional(equity, open_notional)
        assert max_additional == pytest.approx(expected_max, rel=1e-9)

    @pytest.mark.parametrize("mode", ["stop", "log_only"])
    @pytest.mark.parametrize(
        "mtm_equity, open_notional, mmr_rate, should_liquidate",
        [
            (1050.0, 200_000.0, 0.005, False),  # Safely above
            (1000.0, 200_000.0, 0.005, False),  # Exactly at margin → safe (strict <)
            (999.99, 200_000.0, 0.005, True),  # Slightly below → trigger
            (5000.0, 500_000.0, 0.01, False),  # Safe with higher MMR
            (4999.99, 500_000.0, 0.01, True),  # Trigger with higher MMR
            (0.0, 0.0, 0.005, False),  # Zero notional → safe
            (-1000.0, 100_000.0, 0.005, True),  # Negative equity → trigger
        ],
    )
    def test_liquidation_trigger_and_mode_behavior(
            self,
            mode,
            mtm_equity,
            open_notional,
            mmr_rate,
            should_liquidate,
            dummy_ongoing_trades,
    ):
        current_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
        tracker = LiquidationTracker(leverage=5.0, mmr_rate=mmr_rate, mode=mode)

        # Always provide trades for verbose sanity check (current impl doesn't close them)
        tracker.check_and_handle_liquidation(
            mtm_equity=mtm_equity,
            open_notional=open_notional,
            current_time=current_time,
            current_equity=10_000,
            ongoing_trades=dummy_ongoing_trades,
            verbose=True,
        )

        assert tracker.liquidated == should_liquidate
        if should_liquidate:
            assert tracker.liq_time == current_time
        else:
            assert tracker.liq_time is None

        # Current implementation does NOT force-close trades
        # Trades remain open → exit_time/exit_price remain unset (None or default 0.0)
        for trade_list in dummy_ongoing_trades.values():
            for trade in trade_list:
                assert trade.exit_time is None
                # Relaxed: accept 0.0 as "unset" if default in Trade class
                assert trade.exit_price is None or trade.exit_price == 0.0

    def test_post_liquidation_state_persistence(self):
        tracker = LiquidationTracker(leverage=5.0, mode="log_only")
        trigger_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)

        tracker.check_and_handle_liquidation(
            mtm_equity=500.0,
            open_notional=200_000.0,
            current_time=trigger_time,
            current_equity=10_000,
            ongoing_trades={},
        )
        assert tracker.liquidated
        assert tracker.liq_time == trigger_time

        later_time = trigger_time + datetime.timedelta(hours=10)
        tracker.check_and_handle_liquidation(
            mtm_equity=20_000.0,
            open_notional=0.0,
            current_time=later_time,
            current_equity=15_000,
            ongoing_trades={},
        )
        assert tracker.liquidated
        assert tracker.liq_time == trigger_time

    def test_zero_open_notional_edge_case(self):
        tracker = LiquidationTracker(leverage=5.0, mmr_rate=0.005)
        tracker.check_and_handle_liquidation(
            mtm_equity=1000.0,
            open_notional=0.0,
            current_time=datetime.datetime.now(datetime.timezone.utc),
            current_equity=10_000,
            ongoing_trades={},
        )
        assert not tracker.liquidated
        assert tracker.get_max_additional_notional(10_000, 0.0) == 50_000.0

    @pytest.mark.parametrize("mmr_rate", [0.001, 0.005, 0.01, 0.02])
    def test_varying_mmr_rates(self, mmr_rate):
        tracker = LiquidationTracker(leverage=10.0, mmr_rate=mmr_rate)
        required_margin = 100_000 * mmr_rate
        tracker.check_and_handle_liquidation(
            mtm_equity=required_margin + 1,
            open_notional=100_000,
            current_time=datetime.datetime.now(datetime.timezone.utc),
            current_equity=20_000,
            ongoing_trades={},
        )
        assert not tracker.liquidated

        tracker.check_and_handle_liquidation(
            mtm_equity=required_margin - 1,
            open_notional=100_000,
            current_time=datetime.datetime.now(datetime.timezone.utc),
            current_equity=20_000,
            ongoing_trades={},
        )
        assert tracker.liquidated


@pytest.mark.parametrize(
    "leverage, mmr_rate, open_notional, mtm_equity, expect_liq",
    [
        (None, 0.005, 1_000_000, 100_000, False),  # unlimited leverage
        (10.0, 0.005, 100_000, 4_999, True),  # just below maintenance margin
        (10.0, 0.005, 100_000, 5_001, False),  # just above
        (5.0, 0.01, 0, 100_000, False),  # zero notional
    ],
)
def test_liquidation_boundary_cases(leverage, mmr_rate, open_notional, mtm_equity, expect_liq):
    tracker = LiquidationTracker(leverage=leverage, mmr_rate=mmr_rate)
    tracker.check_and_handle_liquidation(
        mtm_equity=mtm_equity,
        open_notional=open_notional,
        current_time=datetime.now(datetime.timezone.utc),
        current_equity=100_000,
        ongoing_trades={},
    )
    assert tracker.liquidated == expect_liq
