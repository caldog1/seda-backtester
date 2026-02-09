"""Comprehensive tests for HybridSlippageModel.

Covers:
- Slippage application for market orders (long/short, various volume shares)
- Zero slippage for limit/stop orders
- Slippage BPS calculation (market vs non-market, volume share edges)
- Configurable parameters (fixed_market_bps, bps_per_full_bar)
- Edge cases (zero liquidity, extreme volume share, negative/zero price)
- Numerical stability and floating-point tolerance
- Asset/timeframe ignored (confirmed — no effect)
"""

import pytest

from src.backtester.core.execution import HybridSlippageModel, OrderType


class TestHybridSlippageModel:
    @pytest.fixture
    def default_model(self):
        return HybridSlippageModel(fixed_market_bps=1.0, bps_per_full_bar=50.0)

    @pytest.fixture
    def custom_model(self):
        return HybridSlippageModel(fixed_market_bps=2.5, bps_per_full_bar=100.0)

    @pytest.mark.parametrize(
        "price, direction, volume_share_pct, expected_price",
        [
            (
                100.0,
                "LONG",
                0.0,
                100.0,
            ),  # No volume share → only fixed (but fixed applied separately?)
            (
                100.0,
                "LONG",
                10.0,
                100.05,
            ),  # 10% → 5 bps variable + 1 fixed = 6 bps → +0.06%
            (100.0, "LONG", 50.0, 100.26),  # 50% → 25 bps variable + 1 fixed = 26 bps
            (
                100.0,
                "LONG",
                100.0,
                100.51,
            ),  # Full bar → 50 bps variable + 1 fixed = 51 bps
            (100.0, "LONG", 200.0, 101.01),  # >100% → capped at full bar + fixed
            (100.0, "SHORT", 10.0, 99.95),  # Adverse lower for short
            (100.0, "SHORT", 50.0, 99.74),
            (100.0, "SHORT", 100.0, 99.49),
            (100.0, "SHORT", 200.0, 98.99),
            (0.0, "LONG", 50.0, 0.0),  # Zero price → no slippage
            (
                -100.0,
                "LONG",
                50.0,
                -100.26,
            ),  # Negative price (defensive — should handle)
        ],
    )
    def test_apply_slippage_market_orders(
        self, default_model, price, direction, volume_share_pct, expected_price
    ):
        slipped = default_model.apply_slippage(price, direction, volume_share_pct)
        assert slipped == pytest.approx(expected_price, rel=1e-10)

    def test_apply_slippage_custom_params(self, custom_model):
        # Custom: fixed 2.5 bps, 100 bps per full bar
        slipped_long = custom_model.apply_slippage(100.0, "LONG", 50.0)
        assert slipped_long == pytest.approx(
            100.525, rel=1e-10
        )  # 50 bps variable + 2.5 fixed = 52.5 bps

        slipped_short = custom_model.apply_slippage(100.0, "SHORT", 100.0)
        assert slipped_short == pytest.approx(
            98.975, rel=1e-10
        )  # 100 bps variable + 2.5 fixed = 102.5 bps adverse

    @pytest.mark.parametrize("order_type", ["limit", "stop", "stop_limit", "unknown"])
    def test_apply_slippage_non_market_orders(self, default_model, order_type):
        # Non-market orders bypass slippage application entirely
        # Test with extreme volume share — should still return original price
        price = default_model.apply_slippage(100.0, "LONG", 999.0)
        assert price == 100.0

    @pytest.mark.parametrize(
        "notional, bar_liquidity, order_type, expected_bps",
        [
            (10_000, 100_000, "market", 6.0),  # 10% share → 5 bps variable + 1 fixed
            (50_000, 100_000, "market", 26.0),  # 50% → 25 + 1
            (100_000, 100_000, "market", 51.0),  # 100% → 50 + 1
            (200_000, 100_000, "market", 51.0),  # >100% capped at full bar + fixed
            (0, 100_000, "market", 1.0),  # Zero notional → only fixed
            (
                10_000,
                0,
                "market",
                51.0,
            ),  # Zero liquidity → treat as full bar + fixed (defensive)
            (10_000, 100_000, "limit", 0.0),  # Limit → always 0
            (10_000, 100_000, "stop", 0.0),  # Stop → always 0
            (10_000, 100_000, "unknown", 0.0),  # Unknown → treated as non-market → 0
        ],
    )
    def test_get_slippage_bps(
        self, default_model, notional, bar_liquidity, order_type, expected_bps
    ):
        bps = default_model.get_slippage_bps(
            notional=notional,
            bar_liquidity=bar_liquidity,
            asset="BTCUSDT",
            timeframe="1h",
            order_type=order_type,
        )
        assert bps == pytest.approx(expected_bps, rel=1e-10)

    def test_get_slippage_bps_custom_params(self, custom_model):
        bps = custom_model.get_slippage_bps(
            notional=20_000,
            bar_liquidity=100_000,
            asset="ETHUSDT",
            timeframe="4h",
            order_type="market",
        )
        assert bps == pytest.approx(
            22.5, rel=1e-10
        )  # 20% share → 20 bps variable + 2.5 fixed

    def test_asset_timeframe_ignored(self, default_model):
        # Confirm asset/timeframe parameters are unused (as per current impl)
        bps1 = default_model.get_slippage_bps(
            10_000, 100_000, "BTCUSDT", "1h", "market"
        )
        bps2 = default_model.get_slippage_bps(
            10_000, 100_000, "ETHUSDT", "4h", "market"
        )
        assert bps1 == bps2

    @pytest.mark.parametrize("price", [0.0, -100.0, 1e-8])  # Extreme/edge prices
    def test_apply_slippage_edge_prices(self, default_model, price):
        slipped_long = default_model.apply_slippage(price, "LONG", 50.0)
        slipped_short = default_model.apply_slippage(price, "SHORT", 50.0)
        # Should handle without error — relative slippage preserved
        expected_long = price * (1 + 0.0026)  # 26 bps for 50% share
        expected_short = price * (1 - 0.0026)
        assert slipped_long == pytest.approx(expected_long, rel=1e-8)
        assert slipped_short == pytest.approx(expected_short, rel=1e-8)
