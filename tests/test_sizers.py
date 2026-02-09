"""Comprehensive tests for all PositionSizer implementations.

Tests cover:
- Core functionality (correct notional calculation)
- Trade size cap enforcement
- Edge cases (zero equity, missing SL/TP, low sample size for Kelly)
- Statistical correctness for Kelly (win rate, payoff ratio)
- Deterministic behavior with realistic inputs

Uses pytest fixtures for reusability and clarity.
All floating-point comparisons use pytest.approx for numerical stability.
"""

import pytest
import numpy as np
from typing import Dict

from src.backtester.sizers.sizers import (
    PositionSizer,
    FixedNotionalSizer,
    FixedFractionalSizer,
    FixedRiskSizer,
    KellyRiskSizer,
)


@pytest.fixture
def base_stats() -> Dict[str, float]:
    """Default trade statistics for Kelly sizer tests."""
    return {
        "total_trades": 100,
        "wins": 55,
        "total_win_pnl": 5500.0,  # avg_win = 100
        "total_loss_pnl": -3600.0,  # avg_loss_abs ≈ 80
    }


@pytest.fixture
def high_cap() -> float:
    return 100_000.0


@pytest.fixture
def low_cap() -> float:
    return 5_000.0


# =============================================================================
# FixedNotionalSizer
# =============================================================================


def test_fixed_notional_basic(high_cap):
    sizer = FixedNotionalSizer(notional=10_000.0, trade_size_cap=high_cap)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=90.0,
        tp_price=120.0,
        current_equity=50_000.0,
        stats={},
    )
    assert notional == pytest.approx(10_000.0)


def test_fixed_notional_cap_enforced(low_cap):
    sizer = FixedNotionalSizer(notional=10_000.0, trade_size_cap=low_cap)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        current_equity=1_000_000.0,
        stats={},
    )
    assert notional == pytest.approx(5_000.0)


def test_fixed_notional_no_cap(high_cap):
    sizer = FixedNotionalSizer(notional=10_000.0, trade_size_cap=None)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        current_equity=1_000_000.0,
        stats={},
    )
    assert notional == pytest.approx(10_000.0)


# =============================================================================
# FixedFractionalSizer
# =============================================================================


def test_fixed_fractional_basic(high_cap):
    sizer = FixedFractionalSizer(fraction=0.10, trade_size_cap=high_cap)  # 10%
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=90.0,
        tp_price=120.0,
        current_equity=50_000.0,
        stats={},
    )
    assert notional == pytest.approx(5_000.0)  # 10% of equity


def test_fixed_fractional_cap_enforced(low_cap):
    sizer = FixedFractionalSizer(fraction=0.20, trade_size_cap=low_cap)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        current_equity=100_000.0,
        stats={},
    )
    assert notional == pytest.approx(5_000.0)  # capped


def test_fixed_fractional_zero_equity():
    sizer = FixedFractionalSizer(fraction=0.10)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        current_equity=0.0,
        stats={},
    )
    assert notional == pytest.approx(0.0)


# =============================================================================
# FixedRiskSizer
# =============================================================================


def test_fixed_risk_basic(high_cap):
    sizer = FixedRiskSizer(risk_fraction=0.01, trade_size_cap=high_cap)  # 1% risk
    entry_price = 100.0
    sl_price = 95.0  # 5% below entry
    # expected ≈ 10_000 (see comment in original test)

    notional = sizer.get_notional(
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=None,
        current_equity=50_000.0,
        stats={},
    )
    assert notional == pytest.approx(10_000.0)


def test_fixed_risk_no_sl_returns_zero():
    sizer = FixedRiskSizer(risk_fraction=0.01)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        current_equity=50_000.0,
        stats={},
    )
    assert notional == pytest.approx(0.0)


def test_fixed_risk_invalid_sl_zero_risk():
    sizer = FixedRiskSizer(risk_fraction=0.01)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=100.0,  # zero distance
        tp_price=None,
        current_equity=50_000.0,
        stats={},
    )
    assert notional == pytest.approx(0.0)


def test_fixed_risk_cap_enforced(low_cap):
    sizer = FixedRiskSizer(risk_fraction=0.05, trade_size_cap=low_cap)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=99.0,  # 1% risk distance
        tp_price=None,
        current_equity=1_000_000.0,
        stats={},
    )
    assert notional == pytest.approx(5_000.0)


# =============================================================================
# KellyRiskSizer
# =============================================================================


def test_kelly_full_fallback_low_trades(base_stats):
    stats = base_stats.copy()
    stats["total_trades"] = 20  # < min_trades

    sizer = KellyRiskSizer(
        kelly_fraction=0.5,
        min_trades=30,
        fallback_notional=10_000.0,
    )
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=95.0,
        tp_price=None,
        current_equity=100_000.0,
        stats=stats,
    )
    assert notional == pytest.approx(10_000.0)


def test_kelly_no_sl_fallback():
    sizer = KellyRiskSizer(fallback_notional=8_000.0)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=None,
        tp_price=None,
        current_equity=100_000.0,
        stats={"total_trades": 100},
    )
    assert notional == pytest.approx(8_000.0)


def test_kelly_full_calculation(base_stats):
    sizer = KellyRiskSizer(
        kelly_fraction=0.5,
        min_trades=30,
        fallback_notional=10_000.0,
        trade_size_cap=None,  # Disable cap for pure math test
    )

    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=95.0,
        tp_price=None,
        current_equity=100_000.0,
        stats=base_stats,
    )
    assert notional == pytest.approx(190_000.0)


def test_kelly_default_cap_applied(base_stats):
    """Verify the built-in default trade_size_cap=100_000 limits exposure."""
    sizer = KellyRiskSizer(
        kelly_fraction=0.5,
        min_trades=30,
        fallback_notional=10_000.0,
        # No trade_size_cap → uses default 100_000.0
    )

    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=95.0,
        tp_price=None,
        current_equity=100_000.0,
        stats=base_stats,
    )
    assert notional == pytest.approx(100_000.0)


def test_kelly_cap_enforced(base_stats, low_cap):
    sizer = KellyRiskSizer(
        kelly_fraction=1.0,
        min_trades=30,
        fallback_notional=10_000.0,
        trade_size_cap=low_cap,
    )
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=90.0,
        tp_price=None,
        current_equity=1_000_000.0,
        stats=base_stats,
    )
    assert notional == pytest.approx(5_000.0)


def test_kelly_zero_or_negative_kelly_f(base_stats):
    stats = base_stats.copy()
    stats["wins"] = 30
    stats["total_win_pnl"] = 3000.0
    stats["total_loss_pnl"] = -7000.0

    sizer = KellyRiskSizer(kelly_fraction=0.5, fallback_notional=12_000.0)
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=95.0,
        tp_price=None,
        current_equity=100_000.0,
        stats=stats,
    )
    assert notional == pytest.approx(12_000.0)


def test_kelly_min_notional_enforced(base_stats):
    sizer = KellyRiskSizer(
        kelly_fraction=0.1,  # very conservative
        fallback_notional=10_000.0,
        trade_size_cap=None,
    )
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=99.9,
        tp_price=None,
        current_equity=100_000.0,
        stats=base_stats,
    )
    # Even with tiny risk distance (huge raw notional), low fraction keeps it above fallback
    assert notional > 10_000.0


def test_kelly_min_notional_enforced_low_calculation():
    """Strict test: poor expectancy → kelly_f = 0 → exact fallback."""
    poor_stats = {
        "total_trades": 100,
        "wins": 40,
        "total_win_pnl": 3200.0,  # avg_win = 80
        "total_loss_pnl": -4800.0,  # avg_loss_abs = 80 → payoff = 1.0 → kelly_f ≈ 0
    }
    sizer = KellyRiskSizer(
        kelly_fraction=0.5,
        fallback_notional=10_000.0,
        trade_size_cap=None,  # Disable cap interference
    )
    notional = sizer.get_notional(
        entry_price=100.0,
        sl_price=95.0,
        tp_price=None,
        current_equity=100_000.0,
        stats=poor_stats,
    )
    # kelly_f clamped to 0 → exact fallback
    assert notional == pytest.approx(10_000.0)
