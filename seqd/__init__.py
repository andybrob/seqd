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
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._annual import fit_annual
from ._holiday import fit_holidays
from ._seqdforecaster import SeqdForecaster, forecast_from_result
from ._structures import (
    AnnualEffect,
    DecompositionResult,
    ForecastResult,
    HolidayEffect,
    SegmentTrend,
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
    "SeqdForecaster",
    "ForecastResult",
    "SegmentTrend",
    "forecast_from_result",
]

__version__ = "0.2.2"


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
    max_holiday_merge_gap_days : int
        Two holiday occurrences are merged into a compound block only when the
        minimum gap between their actual calendar dates is ≤ this value (days).
        Default 7.  Thanksgiving + Black Friday (gap=1) and Black Friday + Cyber
        Monday (gap=3) will merge; Cyber Monday + Christmas (gap=25) will not.
        This criterion is independent of ``holiday_window`` size.
    max_harmonics : int
        Upper bound on the number of Fourier harmonics considered during BIC
        selection for annual seasonality.  BIC searches K ∈ {0, ..., max_harmonics}.
        Default 6.  Reducing this prevents over-fitting on short or sparse series.
        Must be >= 0.
    """

    def __init__(
        self,
        holiday_dates: Union[
            List[Union[str, datetime.date, pd.Timestamp]],
            Dict[str, List[Union[str, datetime.date, pd.Timestamp]]],
        ],
        multiplicative: Optional[bool] = None,
        holiday_window: int = 14,
        max_holiday_window: Optional[int] = None,
        reference_window: int = 60,
        max_holiday_merge_gap_days: int = 7,
        max_harmonics: int = 6,
    ) -> None:
        if holiday_window < 1:
            raise ValueError(
                f"holiday_window must be >= 1 (got {holiday_window})."
            )
        if max_holiday_window is not None and max_holiday_window < 1:
            raise ValueError(
                f"max_holiday_window must be >= 1 (got {max_holiday_window})."
            )
        if reference_window < 1:
            raise ValueError(
                f"reference_window must be >= 1 (got {reference_window})."
            )
        if max_holiday_merge_gap_days < 0:
            raise ValueError(
                f"max_holiday_merge_gap_days must be >= 0 "
                f"(got {max_holiday_merge_gap_days})."
            )
        if max_harmonics < 0:
            raise ValueError(
                f"max_harmonics must be >= 0 (got {max_harmonics})."
            )
        self.holidays = normalize_holiday_input(holiday_dates)
        self.multiplicative = multiplicative
        self.holiday_window = holiday_window
        self.max_holiday_window = max_holiday_window
        self.reference_window = reference_window
        self.max_holiday_merge_gap_days = max_holiday_merge_gap_days
        self.max_harmonics = max_harmonics

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
            raise ValueError("y must have a DatetimeIndex")
        if y.index.duplicated().any():
            raise ValueError("y contains duplicate dates in its index")

        y = y.sort_index().astype(float)

        # Validate: no NaN values
        if y.isna().any():
            n_nan = int(y.isna().sum())
            raise ValueError(
                f"y contains {n_nan} NaN value(s). "
                "Impute or drop missing values before decomposing."
            )

        # Validate: daily frequency (gaps must be exactly 1 day)
        if len(y) >= 2:
            diffs = y.index.to_series().diff().dropna()
            max_gap_days = diffs.max().days
            if max_gap_days > 1:
                raise ValueError(
                    f"y has gaps larger than 1 day (max gap = {max_gap_days} days). "
                    "seqd requires a contiguous daily series."
                )

        # Warn if series is too short for reliable baseline estimation
        if len(y) < 2 * self.reference_window:
            warnings.warn(
                f"Series length ({len(y)}) is less than 2 * reference_window "
                f"({2 * self.reference_window}). Holiday baseline estimation may be "
                "unreliable. Consider reducing reference_window or providing more data.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if series is too short for reliable annual seasonality estimation
        if len(y) < 365:
            warnings.warn(
                f"Series length ({len(y)}) is less than 365 days. "
                "Annual seasonality estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

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
        effective_window = self.max_holiday_window if self.max_holiday_window is not None else self.holiday_window
        holiday_effects, y_h = fit_holidays(
            y_w=y_w,
            holidays=self.holidays,
            holiday_window=effective_window,
            reference_window=self.reference_window,
            max_holiday_merge_gap_days=self.max_holiday_merge_gap_days,
        )

        # Stage 3: Annual
        annual_effect, y_clean = fit_annual(y_h=y_h, max_harmonics=self.max_harmonics)

        # Compute R² by component using marginal (sequential) contribution.
        #
        # Marginal contributions are computed from the sequential residuals already
        # produced by the pipeline:
        #   y       -> weekly removal  -> y_w   (residual after weekly)
        #   y_w     -> holiday removal -> y_h   (residual after holiday)
        #   y_h     -> annual removal  -> y_clean (final residual)
        #
        # r2_weekly  = 1 - var(y_w)      / var(original)
        # r2_holiday = [var(y_w) - var(y_h)]  / var(original)
        # r2_annual  = [var(y_h) - var(y_clean)] / var(original)
        #
        # Each value is the marginal variance fraction removed at that stage.
        # Values are clipped to [0, 1] to guard against floating-point noise when
        # components are near zero.
        var_original = float(np.var(original.values))
        r2 = {}
        if var_original > 0:
            var_after_weekly = float(np.var(y_w.values))
            var_after_holiday = float(np.var(y_h.values))
            var_final = float(np.var(y_clean.values))

            r2_weekly_raw = 1.0 - (var_after_weekly / var_original)
            r2_holiday_raw = (var_after_weekly - var_after_holiday) / var_original
            r2_annual_raw = (var_after_holiday - var_final) / var_original

            _component_stage = {"weekly": 1, "holiday": 2, "annual": 3}
            for _cname, _raw in [("weekly", r2_weekly_raw), ("holiday", r2_holiday_raw), ("annual", r2_annual_raw)]:
                if _raw < -0.01:  # tolerance for float noise
                    warnings.warn(
                        f"R² for '{_cname}' component is {_raw:.4f} (negative). "
                        f"Stage {_component_stage[_cname]} increased residual variance. "
                        "Decomposition may be unreliable. Check holiday_window and reference_window.",
                        UserWarning,
                        stacklevel=2,
                    )

            r2 = {
                "weekly": float(np.clip(r2_weekly_raw, 0.0, 1.0)),
                "holiday": float(np.clip(r2_holiday_raw, 0.0, 1.0)),
                "annual": float(np.clip(r2_annual_raw, 0.0, 1.0)),
            }
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
