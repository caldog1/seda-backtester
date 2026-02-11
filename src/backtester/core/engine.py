"""Main orchestrator for synchronous event-driven backtesting.

Loads data via pluggable providers with flexible filepath support:
- Default quickstart: auto-discovers {asset}_{timeframe}.csv in sample_data/
- Advanced: explicit custom_filepaths dict for arbitrary filenames/locations
- Custom provider override for future extensions (e.g., Binance/CCXT)

Builds unified timeline, coordinates simulation, and returns comprehensive results.
Designed for multi-asset, multi-timeframe portfolio simulation in perpetual futures.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtester.core.liquidation_tracker import LiquidationTracker
from backtester.core.results import BacktestResults
from backtester.core.simulation import run_simulation
from backtester.core.execution import (
    DefaultFeeModel,
    HybridSlippageModel,
    OrderType,
)
from backtester.strategies.base import Strategy
from backtester.data.base_provider import DataProvider
from backtester.data.csv_provider import CSVDataProvider

# Quiet logging by default — examples can configure if needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Primary class for configuring and running backtests."""

    def __init__(
        self,
        timeframes: List[str],
        asset_list: List[str],
        strategies: List[Strategy],
        start_date: dt.datetime,
        end_date: Optional[dt.datetime] = None,
        initial_capital: float = 100_000.0,
        leverage: Optional[float] = None,
        mmr_rate: float = 0.005,
        liq_mode: str = "force_close",
        liq_slippage_pct: float = 0.01,
        data_provider: Optional[DataProvider] = None,
        data_root: Optional[str] = "../sample_data",  # For quick dev/testing when running from repo
        custom_filepaths: Optional[Dict[str | Tuple[str, str], str]] = None,  # Supports both formats
    ) -> None:
        self.timeframes = timeframes
        self.asset_list = asset_list
        self.strategies = strategies
        self.start_date = start_date
        self.end_date = end_date

        self.initial_capital = initial_capital
        self.current_equity = initial_capital

        self.liq_tracker = LiquidationTracker(
            leverage=leverage,
            mmr_rate=mmr_rate,
            mode=liq_mode,
            slippage_pct=liq_slippage_pct,
        )

        # Data provider setup
        if data_provider is None:
            data_provider = CSVDataProvider()
        self.data_provider = data_provider

        self.data_root = data_root if data_root is not None else None

        # Normalized custom_filepaths — always stored as Dict[Tuple[str, str], str]
        self.custom_filepaths: Dict[Tuple[str, str], str] = {}
        if custom_filepaths:
            for key, path in custom_filepaths.items():
                if isinstance(key, str):
                    if "_" not in key:
                        raise ValueError(
                            f"Invalid string key in custom_filepaths: '{key}'. "
                            "Expected format 'ASSET_TIMEFRAME' (e.g., 'BTCUSDT_1h') "
                            "or a tuple (e.g., ('BTCUSDT', '1h'))."
                        )
                    asset, timeframe = key.split("_", 1)  # Split on first underscore only
                    normalized_key = (asset, timeframe)
                elif isinstance(key, tuple) and len(key) == 2 and all(isinstance(x, str) for x in key):
                    normalized_key = key
                else:
                    raise ValueError(
                        f"Invalid key in custom_filepaths: {key}. "
                        "Expected str ('ASSET_TIMEFRAME') or tuple[str, str]."
                    )
                self.custom_filepaths[normalized_key] = path

        self.fee_model = DefaultFeeModel()
        self.slippage_model = HybridSlippageModel(
            fixed_market_bps=3.0,
            bps_per_full_bar=50.0,
        )

        self.arrays: Dict[Tuple[str, str], dict] = {}
        self.pointers: Dict[Tuple[str, str], int] = {}
        self.master_time_list: Optional[List[dt.datetime]] = None

    def load_data(self) -> None:
        """Load candle data using custom_filepaths, auto-discovery, or explicit provider."""
        logger.info("\n=== LOADING DATA ===")
        loaded = 0
        requested = len(self.asset_list) * len(self.timeframes)

        # Robust UTC normalization: handle both naive and aware input
        start_ts = pd.Timestamp(self.start_date)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

        end_ts: Optional[pd.Timestamp] = None
        if self.end_date is not None:
            end_ts = pd.Timestamp(self.end_date)
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")

        for asset in self.asset_list:
            for tf in self.timeframes:
                key = (asset, tf)

                # Priority 1: explicit custom filepath
                filepath = self.custom_filepaths.get(key)

                # Priority 2: auto-generate from data_root (quickstart)
                if filepath is None and self.data_root is not None:
                    filepath = os.path.join(self.data_root, f"{asset.upper()}_{tf}.csv")

                try:
                    arrays = self.data_provider.load_data(
                        asset=asset,
                        timeframe=tf,
                        start_date=start_ts,
                        end_date=end_ts,
                        filepath=filepath,  # Passed explicitly — required by CSVDataProvider
                    )

                    if len(arrays["Close"]) == 0:
                        logger.warning(f"No data returned for {asset} {tf}")
                        continue

                    self.arrays[key] = arrays
                    loaded += 1
                except FileNotFoundError:
                    logger.warning(
                        f"File not found for {asset} {tf}: {filepath or 'None'}"
                    )
                except Exception as e:
                    logger.warning(f"Error loading {asset} {tf}: {e}")

        logger.info(f"Loaded {loaded}/{requested} series")
        if loaded == 0:
            raise RuntimeError(
                "No data loaded for any asset/timeframe.\n\n"
                "SEDA requires OHLCV CSV files (columns: timestamp, open, high, low, close, volume).\n"
                "Options:\n"
                "1. Place files in a 'sample_data/' folder next to your script (naming: {asset}_{timeframe}.csv)\n"
                "2. Use the 'custom_filepaths' dict in Backtester (recommended for production)\n"
                "3. Clone the repo for included samples: git clone https://github.com/caldog1/seda-backtester.git\n"
                "   Samples: https://github.com/caldog1/seda-backtester/tree/main/sample_data"
            )

    def execute_order(
        self,
        intended_price: float,
        direction: str,  # "LONG" or "SHORT"
        notional: float,
        order_type: OrderType,
        arrays: dict,
        idx: int,
        asset: str,
        timeframe: str,
    ) -> tuple[float, float]:
        """
        Centralised realistic order execution.

        - Applies fee model
        - Applies slippage model (only for market orders)
        - Estimates bar liquidity in quote currency (Volume assumed to be base volume)

        Returns (actual_fill_price, fee_rate)
        """
        # Fee
        fee_rate = self.fee_model.get_fee_rate(order_type)

        # No slippage for limit orders (assumes perfect fill at intended price)
        if order_type == "limit":
            return intended_price, fee_rate

        # Market order → apply adverse slippage
        close_price = arrays["Close"][idx]
        base_volume = arrays["Volume"][idx]
        # Approximate quote liquidity (critical for USDT perpetuals)
        bar_liquidity_quote = base_volume * close_price if base_volume > 0 else 1.0

        slippage_bps = self.slippage_model.get_slippage_bps(
            notional=notional,
            bar_liquidity=bar_liquidity_quote,
            asset=asset,
            timeframe=timeframe,
            order_type=order_type,
        )

        actual_price = self.slippage_model.apply_slippage(
            price=intended_price,
            direction=direction.upper(),
            bps=slippage_bps,
        )

        return actual_price, fee_rate

    def build_master_timeline(self) -> None:
        """Build sorted unified timeline from all loaded series."""
        all_times = set()
        for arrays in self.arrays.values():
            all_times.update(arrays["Close Time"])
        self.master_time_list = sorted(all_times)

        if not self.master_time_list:
            raise RuntimeError("Master timeline empty — no data loaded")

    def prepare_backtest(self) -> None:
        """Load data, build timeline, and initialise pointers."""
        self.load_data()
        self.build_master_timeline()
        self.pointers = {key: 0 for key in self.arrays}

    def run(self) -> BacktestResults:
        """Public entry point — run the full backtest and return results."""
        self.prepare_backtest()
        return run_simulation(self)
