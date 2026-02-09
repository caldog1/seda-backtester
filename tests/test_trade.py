"""Comprehensive unit tests for the Trade class.

Covers:
- Full lifecycle (open → close)
- Quantity/notional validation
- PnL calculation (gross + fees)
- Slippage & fee recording
- SL/TP handling (pct + absolute, conflicts, long/short)
- Duration calculation
- Defensive errors (already closed, invalid params)
- Representation
"""

import datetime as dt

import pytest

from src.backtester.core.trade import Trade, TradeDirection, OrderType, TradeStatus


@pytest.fixture
def base_time():
    return dt.datetime(2023, 1, 1, 12, 0)


@pytest.fixture
def later_time(base_time):
    return base_time + dt.timedelta(days=2)


# =============================================================================
# Lifecycle & Basic PnL
# =============================================================================


def test_trade_lifecycle_long_with_fees_and_slippage(base_time, later_time):
    trade = Trade()

    trade.open_trade(
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=base_time,
        notional_value=10_000.0,
        entry_order_type="market",
        entry_fee_rate=0.0004,  # 4 bps
    )
    assert trade.quantity == pytest.approx(100.0)
    assert trade.trade_value == pytest.approx(10_000.0)
    assert trade.entry_fee_dollar == pytest.approx(4.0)
    assert trade.status is TradeStatus.OPEN

    trade.close_trade(
        exit_price=110.0,
        exit_time=later_time,
        exit_order_type="market",
        exit_fee_rate=0.0004,
    )
    # Gross: +1000, fees: 4 + 4 = 8 → net 992
    assert trade.pnl == pytest.approx(992.0)
    assert trade.fee == pytest.approx(8.0)
    assert trade.duration == dt.timedelta(days=2)
    assert trade.status is TradeStatus.CLOSED


def test_trade_lifecycle_short():
    trade = Trade()
    trade.open_trade(
        direction="SHORT",
        entry_price=100.0,
        entry_time=dt.datetime.now(),
        quantity=50.0,
    )
    trade.close_trade(exit_price=90.0, exit_time=dt.datetime.now())
    assert trade.pnl == pytest.approx(500.0)  # (100-90) * 50


def test_trade_close_already_closed():
    trade = Trade()
    trade.open_trade("LONG", 100.0, dt.datetime.now(), notional_value=1000.0)
    trade.close_trade(110.0, dt.datetime.now())

    with pytest.raises(ValueError, match="not open"):
        trade.close_trade(120.0, dt.datetime.now())


# =============================================================================
# Validation
# =============================================================================


@pytest.mark.parametrize("direction", ["LONG", TradeDirection.LONG, "long", "LoNg"])
def test_direction_case_insensitive(direction):
    trade = Trade()
    trade.open_trade(direction, 100.0, dt.datetime.now(), notional_value=1000.0)
    assert trade.direction is TradeDirection.LONG


def test_invalid_direction():
    trade = Trade()
    with pytest.raises(ValueError):
        trade.open_trade("DIAGONAL", 100.0, dt.datetime.now(), notional_value=1000.0)


@pytest.mark.parametrize("bad_price", [-100.0, 0.0])
def test_negative_or_zero_price(bad_price, base_time):
    trade = Trade()
    with pytest.raises(ValueError, match="entry_price"):
        trade.open_trade("LONG", bad_price, base_time, notional_value=1000.0)


def test_missing_quantity_and_notional(base_time):
    trade = Trade()
    with pytest.raises(ValueError, match="exactly one"):
        trade.open_trade("LONG", 100.0, base_time)


def test_both_quantity_and_notional(base_time):
    trade = Trade()
    with pytest.raises(ValueError, match="exactly one"):
        trade.open_trade("LONG", 100.0, base_time, quantity=10.0, notional_value=1000.0)


# =============================================================================
# SL/TP Handling
# =============================================================================


def test_sl_tp_percentage_long():
    trade = Trade()
    trade.open_trade(
        "LONG", 100.0, dt.datetime.now(), notional_value=10000.0, sl_pct=2.0, tp_pct=5.0
    )
    assert trade.stoploss == pytest.approx(98.0)
    assert trade.takeprofit == pytest.approx(105.0)


def test_sl_tp_percentage_short():
    trade = Trade()
    trade.open_trade(
        "SHORT",
        100.0,
        dt.datetime.now(),
        notional_value=10000.0,
        sl_pct=2.0,
        tp_pct=5.0,
    )
    assert trade.stoploss == pytest.approx(102.0)
    assert trade.takeprofit == pytest.approx(95.0)


def test_absolute_sl_tp():
    trade = Trade()
    trade.open_trade(
        "LONG", 100.0, dt.datetime.now(), notional_value=10000.0, sl=95.0, tp=110.0
    )
    assert trade.stoploss == 95.0
    assert trade.takeprofit == 110.0


def test_conflicting_sl_tp_params():
    trade = Trade()
    with pytest.raises(ValueError, match="both absolute sl and sl_pct"):
        trade.open_trade(
            "LONG",
            100.0,
            dt.datetime.now(),
            notional_value=10000.0,
            sl=95.0,
            sl_pct=2.0,
        )
    with pytest.raises(ValueError, match="both absolute tp and tp_pct"):
        trade.open_trade(
            "LONG",
            100.0,
            dt.datetime.now(),
            notional_value=10000.0,
            tp=110.0,
            tp_pct=5.0,
        )


# =============================================================================
# Slippage & Fee Recording (called by engine)
# =============================================================================


def test_record_exit_execution_long_adverse_slippage(base_time):
    trade = Trade()
    trade.open_trade("LONG", 100.0, base_time, quantity=100.0)

    # Intended 110, actual fill 109.5 → adverse slippage
    trade._record_exit_execution(
        trade=trade,
        intended_price=110.0,
        actual_price=109.5,
        fee_rate=0.0004,
    )
    assert trade.exit_slippage_bps == pytest.approx(
        45.45, rel=1e-3
    )  # (110-109.5)/110 * 10_000
    assert trade.exit_slippage_dollar == pytest.approx(50.0)
    assert trade.exit_fee_dollar == pytest.approx(4.0)  # based on original notional


# =============================================================================
# Representation & Edge Cases
# =============================================================================


def test_repr_closed_trade():
    trade = Trade()
    trade.open_trade("LONG", 100.0, dt.datetime.now(), notional_value=10000.0)
    trade.close_trade(105.0, dt.datetime.now())
    repr_str = repr(trade)
    assert "LONG" in repr_str
    assert "PnL=+500.00" in repr_str or "PnL=+499." in repr_str  # fees may vary
    assert "Closed" in repr_str


def test_breakeven():
    trade = Trade()
    trade.open_trade(
        "LONG", 100.0, dt.datetime.now(), notional_value=10000.0, entry_fee_rate=0.0004
    )
    trade.close_trade(100.0, dt.datetime.now(), exit_fee_rate=0.0004)
    assert trade.pnl == pytest.approx(-8.0)  # round-trip fees only
