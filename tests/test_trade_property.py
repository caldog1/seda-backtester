"""Property-based tests for Trade PnL calculation using hypothesis."""

from datetime import datetime, timedelta
from hypothesis import given, strategies as st
import pytest

from backtester.core.trade import Trade, TradeDirection


@given(
    direction=st.sampled_from(TradeDirection),
    entry_price=st.floats(min_value=0.01, max_value=1e6),
    exit_price=st.floats(min_value=0.01, max_value=1e6),
    quantity=st.floats(min_value=0.01, max_value=1000),
    entry_fee_rate=st.floats(min_value=0.0, max_value=0.01),
    exit_fee_rate=st.floats(min_value=0.0, max_value=0.01),
)
def test_trade_pnl_round_trip_correct(
    direction, entry_price, exit_price, quantity, entry_fee_rate, exit_fee_rate
):
    trade = Trade()

    trade.open_trade(
        direction=direction,
        entry_price=entry_price,
        entry_time=datetime.now(),
        quantity=quantity,
        entry_order_type="market",
        entry_fee_rate=entry_fee_rate,
    )

    trade.close_trade(
        exit_price=exit_price,
        exit_time=datetime.now() + timedelta(days=1),
        exit_order_type="market",
        exit_fee_rate=exit_fee_rate,
    )

    notional = entry_price * quantity
    expected_fee = notional * (entry_fee_rate + exit_fee_rate)

    if direction == TradeDirection.LONG:
        expected_gross = (exit_price - entry_price) * quantity
    else:
        expected_gross = (entry_price - exit_price) * quantity

    expected_pnl = expected_gross - expected_fee

    assert trade.pnl == pytest.approx(expected_pnl, rel=1e-6)
    assert trade.fee == pytest.approx(expected_fee, rel=1e-6)