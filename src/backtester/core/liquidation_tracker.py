"""
Tracks leverage limits, maintenance margin, and liquidation events.

Supports unlimited leverage or capped modes with configurable behavior:
- 'log_only': Record liquidation without action
- 'stop': Prevent new entries after liquidation
- 'force_close': Immediately close all positions with adverse slippage (future extension)
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from src.backtester.core.trade import TradeDirection


class LiquidationTracker:
    def __init__(
        self,
        leverage: Optional[float] = None,
        mmr_rate: float = 0.005,  # Maintenance margin rate (e.g., 0.5%)
        mode: str = "force_close",  # 'log_only', 'stop', 'force_close'
        slippage_pct: float = 0.01,  # Adverse fill on forced close
    ) -> None:
        self.leverage = leverage
        self.mmr_rate = mmr_rate
        self.mode = mode.lower()
        self.slippage_pct = slippage_pct

        self.liquidated = False
        self.liq_time: Optional[dt.datetime] = None

    def get_max_additional_notional(
        self,
        current_realized_equity: float,
        current_open_notional: float,
    ) -> float:
        """Maximum additional notional without breaching leverage cap."""
        if self.leverage is None:
            return float("inf")
        max_total = current_realized_equity * self.leverage
        return max(0.0, max_total - current_open_notional)

    def check_and_handle_liquidation(
        self,
        mtm_equity: float,
        open_notional: float,
        current_time: dt.datetime,
        current_equity: float,
        ongoing_trades: dict,
        verbose: bool = False,
    ) -> None:
        """Check for liquidation and handle according to mode."""
        if self.leverage is None or self.liquidated:
            return

        maintenance_margin = open_notional * self.mmr_rate
        if mtm_equity < maintenance_margin:
            self.liquidated = True
            self.liq_time = current_time

            if verbose:
                unreal_pnl = mtm_equity - current_equity
                effective_lev = (
                    open_notional / current_equity
                    if current_equity > 0
                    else float("inf")
                )
                total_positions = sum(len(trades) for trades in ongoing_trades.values())

                print("\n" + "!" * 100)
                print(f"ACCOUNT LIQUIDATED at {current_time}")
                print("!" * 100)
                print(f"Realized Equity: ${current_equity:,.2f}")
                print(f"Unrealized PnL: ${unreal_pnl:,.2f}")
                print(f"MTM Equity: ${mtm_equity:,.2f}")
                print(f"Maintenance Margin: ${maintenance_margin:,.2f}")
                print(f"Open Notional: ${open_notional:,.2f}")
                print(
                    f"Effective Leverage: {effective_lev:.2f}x (Cap: {self.leverage}x)"
                )
                print(f"Open Positions: {total_positions}")
                print("!" * 100)

            if self.mode == "stop":
                # Signal to Backtester — handled there
                pass
            elif self.mode == "force_close":
                # Future: force-close all trades with slippage
                pass

    def __repr__(self) -> str:
        return (
            f"<LiquidationTracker leverage={self.leverage} mode={self.mode} "
            f"liquidated={self.liquidated}>"
        )
