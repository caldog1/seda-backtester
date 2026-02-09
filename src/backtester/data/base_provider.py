"""Abstract base class defining the DataProvider interface.

All concrete providers (CSV, Binance/CCXT, etc.) must implement load_data().
The method returns prepared candle data as a dictionary of NumPy arrays
compatible with the synchronous simulation loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DataProvider(ABC):
    """Abstract base class for all data providers."""

    @abstractmethod
    def load_data(
        self,
        asset: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, np.ndarray]:
        """Load and return candle data for a single asset/timeframe pair.

        Must return a dict with at least these keys (NumPy arrays):
            - 'Close Time': datetime64[ns] array of bar close times
            - 'Open': float array
            - 'High': float array
            - 'Low': float array
            - 'Close': float array
            - 'Volume': float array (base volume)

        Additional columns (e.g., 'Open Time') are allowed but ignored by the engine.
        Data must be sorted ascending by time and contain no duplicates.
        """
        raise NotImplementedError
