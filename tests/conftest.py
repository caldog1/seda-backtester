"""Shared pytest fixtures for core module tests."""

import pytest
import datetime as dt

from src.backtester.core.trade import Trade, TradeDirection, OrderType
from src.backtester.core.execution import DefaultFeeModel, HybridSlippageModel


@pytest.fixture
def sample_trade() -> Trade:
    """Fixture: a basic closed long trade with fees/slippage."""
    trade = Trade()
    trade.open_trade(
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=dt.datetime(2023, 1, 1),
        notional_value=10000.0,
        sl_pct=2.0,
        tp_pct=5.0,
        entry_order_type="market",
        entry_fee_rate=0.0004,
    )
    trade.close_trade(
        exit_price=105.0,
        exit_time=dt.datetime(2023, 1, 2),
        exit_order_type="market",
        exit_fee_rate=0.0004,
    )
    return trade


@pytest.fixture
def sample_loss_trade():
    trade = Trade()
    # Mirror sample_trade but make it a realistic losing LONG
    trade.open_trade(
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=dt.datetime(2024, 1, 10, 12, 0),
        quantity=10.0,  # 10 units → $1000 notional
        entry_order_type="market",
        entry_fee_rate=0.0004,
    )
    trade.close_trade(
        exit_price=90.0,  # -10% move
        exit_time=dt.datetime(2024, 1, 12, 12, 0),
        exit_order_type="market",
        exit_fee_rate=0.0004,
    )
    # Expected: gross PnL ≈ -100, fees ≈ $0.80 round-trip → net_pnl ≈ -100.80
    return trade


@pytest.fixture
def fee_model() -> DefaultFeeModel:
    return DefaultFeeModel()


@pytest.fixture
def slippage_model() -> HybridSlippageModel:
    return HybridSlippageModel()
