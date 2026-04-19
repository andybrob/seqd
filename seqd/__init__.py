"""seqd: Sequential time series decomposition.

Cleanly extracts weekly, holiday (with data-driven ramp detection), and annual
effects from a daily time series, leaving only trend and noise.

Quick start
-----------
>>> import pandas as pd
>>> from seqd import SeqdDecomposer
>>>
>>> holidays = ["2022-12-25", "2023-12-25", "2024-12-25"]
>>> decomp = SeqdDecomposer(holiday_dates=holidays)
>>> result = decomp.fit(y)          # y is a pd.Series with DatetimeIndex
>>> result.residual                  # trend + noise
>>> result.weekly_component()       # DOW effects
>>> result.holiday_component()      # holiday ramp effects
>>> result.annual_component()       # Fourier annual seasonality
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._annual import fit_annual
from ._holiday import fit_holidays
from ._structures import (
    AnnualEffect,
    DecompositionResult,
    HolidayEffect,
    WeeklyEffect,
)
from ._utils import all_holiday_dates_flat, normalize_holiday_input
from ._weekly import fit_weekly

__all__ = [
    "SeqdDecomposer",
    "DecompositionResult",
    "WeeklyEffect",
    "HolidayEffect",
    "AnnualEffect",
]

__version__ = "0.1.0"


class SeqdDecomposer:
    """Sequential time series decomposer.

    Applies a three-stage pipeline:
      1. Weekly day-of-week effects (robust, holiday-masked)
      2. Holiday effects with data-driven ramp detection (CUSUM)
      3. Annual seasonality via BIC-selected Fourier regression

    Parameters
    ----------
    holiday_dates : list or dict
        Holiday dates. Accepts:
        - ``["2023-12-25", "2024-12-25"]`` — flat list (same holiday each year)
        - ``{"Christmas": ["2023-12-25", "2024-12-25"], ...}`` — named holidays
        - Each element may be a str, ``datetime.date``, or ``pd.Timestamp``.
    multiplicative : bool or None
        Force additive (False) or multiplicative (True) mode. ``None`` = auto-detect.
        Auto-detection uses ``std(residuals) / mean(y) > 0.15``.
    holiday_window : int
        Days before/after each holiday to search for ramp effects. Default 14.
    reference_window : int
        Days used for baseline estimation pre/post the holiday gap. Default 60.
    """

    def __init__(
        self,
        holiday_dates: Union[
            List[Union[str, datetime.date, pd.Timestamp]],
            Dict[str, List[Union[str, datetime.date, pd.Timestamp]]],
        ],
        multiplicative: Optional[bool] = None,
        holiday_window: int = 14,
        reference_window: int = 60,
    ) -> None:
        self.holidays = normalize_holiday_input(holiday_dates)
        self.multiplicative = multiplicative
        self.holiday_window = holiday_window
        self.reference_window = reference_window

    def fit(self, y: pd.Series) -> DecompositionResult:
        """Fit the decomposition on the input series.

        Parameters
        ----------
        y : pd.Series
            Daily time series with a ``DatetimeIndex``. Must not contain
            duplicate dates.

        Returns
        -------
        DecompositionResult
        """
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("y must have a DatetimeIndex")
        if y.index.duplicated().any():
            raise ValueError("y contains duplicate dates in its index")

        y = y.sort_index().astype(float)
        original = y.copy()

        # Flatten all holiday dates for masking purposes
        flat_holidays = all_holiday_dates_flat(self.holidays)

        # Stage 1: Weekly
        weekly_effect, y_w, is_mult = fit_weekly(
            y=y,
            holiday_dates_flat=flat_holidays,
            multiplicative=self.multiplicative,
        )

        # Stage 2: Holiday
        holiday_effects, y_h = fit_holidays(
            y_w=y_w,
            holidays=self.holidays,
            holiday_window=self.holiday_window,
            reference_window=self.reference_window,
        )

        # Stage 3: Annual
        annual_effect, y_clean = fit_annual(y_h=y_h)

        # Compute R² by component
        var_original = float(np.var(original.values))
        r2 = {}
        if var_original > 0:
            # weekly variance
            dow = original.index.dayofweek
            weekly_comp = weekly_effect.coefficients[dow]
            if not is_mult:
                r2["weekly"] = float(np.var(weekly_comp)) / var_original
            else:
                removed = original.values * (1.0 - 1.0 / weekly_comp)
                r2["weekly"] = float(np.var(removed)) / var_original

            # holiday variance
            if holiday_effects:
                total_hol = np.zeros(len(original))
                for he in holiday_effects:
                    total_hol += he.effect_series.reindex(original.index, fill_value=0.0).values
                r2["holiday"] = float(np.var(total_hol)) / var_original
            else:
                r2["holiday"] = 0.0

            # annual variance
            r2["annual"] = float(np.var(annual_effect.component.values)) / var_original
        else:
            r2 = {"weekly": 0.0, "holiday": 0.0, "annual": 0.0}

        return DecompositionResult(
            series=original,
            weekly=weekly_effect,
            holidays=holiday_effects,
            annual=annual_effect,
            residual=y_clean,
            r2_by_component=r2,
        )
