"""Simple CSV data provider for user-supplied OHLCV files.

Loads a single CSV file specified by full filepath.
Common column formats are auto-detected (timestamp + OHLCV).
No caching — fresh load every time for determinism and simplicity.

This design gives maximum flexibility: users can point to any filename/location/format.
For multi-asset quickstarts, Backtester auto-generates filepaths from data_root + standard naming.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.backtester.data.base_provider import DataProvider


class CSVDataProvider(DataProvider):
    """CSV data provider loading from explicit filepaths."""

    def load_data(
        self,
        asset: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: Optional[pd.Timestamp] = None,
        filepath: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Load a single CSV file and return engine-compatible arrays.

        Args:
            asset/timeframe: For error messages only (not used for path).
            filepath: Full path to CSV. Required — raises if None.
        """
        if filepath is None:
            raise ValueError("CSVDataProvider requires explicit filepath per series")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV not found: {filepath}")

        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError(f"Empty CSV: {filepath}")

        # Auto-detect columns
        cols = {c.lower(): c for c in df.columns}

        timestamp_col = None
        for candidate in [
            "timestamp",
            "date",
            "datetime",
            "time",
            "close time",
            "open time",
        ]:
            if candidate in cols:
                timestamp_col = cols[candidate]
                break
        if timestamp_col is None:
            raise ValueError(f"No timestamp column in {filepath}")

        mapping = {"Close Time": timestamp_col}
        for std, variants in {
            "Open": ["open"],
            "High": ["high"],
            "Low": ["low"],
            "Close": ["close"],
            "Volume": ["volume", "vol"],
        }.items():
            found = None
            for var in variants:
                if var in cols:
                    found = cols[var]
                    break
            if found is None:
                raise ValueError(f"Missing {std} column in {filepath}")
            mapping[std] = found

        # Parse & filter
        ts_col = mapping["Close Time"]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col])

        if start_date:
            df = df[df[ts_col] >= start_date]
        if end_date:
            df = df[df[ts_col] <= end_date]

        if df.empty:
            raise ValueError(f"No data in date range for {filepath}")

        df = df.sort_values(ts_col).reset_index(drop=True)

        arrays = {
            "Close Time": df[ts_col].values.astype("datetime64[ns]"),
            "Open": df[mapping["Open"]].values.astype(float),
            "High": df[mapping["High"]].values.astype(float),
            "Low": df[mapping["Low"]].values.astype(float),
            "Close": df[mapping["Close"]].values.astype(float),
            "Volume": df[mapping["Volume"]].values.astype(float),
        }

        return arrays
