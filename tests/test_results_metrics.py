"""Tests for metric calculations (using compute_trade_metrics and streaks)."""

import datetime

import pytest

from backtester.core.simulation import _compute_trade_metrics
from tests.conftest import sample_trade


def test_trade_metrics(sample_trade):
    metrics = _compute_trade_metrics([sample_trade])
    assert metrics["total_trades"] == 1
    assert metrics["win_rate"] == 100.0
    assert 400 < metrics["net_pnl"] < 500  # Realistic range after fees
    assert metrics["avg_duration_hours"] > 0  # Duration stats now work


def test_trade_metrics_no_trades():
    metrics = _compute_trade_metrics([])
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["avg_duration_hours"] == 0.0


def test_trade_metrics_multiple(sample_trade, sample_loss_trade):
    # Order: win → win → loss → correct max_win_streak = 2
    trades = [
        sample_trade,
        sample_trade,
        sample_loss_trade,
    ]  # two identical wins + one loss
    metrics = _compute_trade_metrics(trades)

    assert metrics["total_trades"] == 3
    assert metrics["win_rate"] == pytest.approx(66.66666666666667)
    assert metrics["max_win_streak"] == 2
    assert metrics["max_loss_streak"] == 1
