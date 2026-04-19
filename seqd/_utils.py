"""Shared utility functions for seqd."""

from __future__ import annotations

import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def trimmed_mean(x: np.ndarray, proportiontocut: float = 0.10) -> float:
    """Compute trimmed mean, cutting ``proportiontocut`` from each tail."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    n = len(x)
    sorted_x = np.sort(x)
    cut = int(np.floor(proportiontocut * n))
    if cut == 0:
        return float(np.mean(sorted_x))
    trimmed = sorted_x[cut : n - cut]
    if len(trimmed) == 0:
        return float(np.mean(sorted_x))
    return float(np.mean(trimmed))


def mad_sigma(x: np.ndarray) -> float:
    """Robust sigma estimate via median absolute deviation.

    sigma = median(|x - median(x)|) / 0.6745
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.6745)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS slope of y on x (simple linear regression)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 0.0
    n = len(x)
    xm = x.mean()
    ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS via normal equations. Returns coefficients."""
    # Add numerical stability via least-squares
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

HolidayInput = Union[
    List[Union[str, datetime.date, pd.Timestamp]],
    Dict[str, List[Union[str, datetime.date, pd.Timestamp]]],
]


def normalize_holiday_input(
    holiday_dates: HolidayInput,
) -> Dict[str, List[datetime.date]]:
    """Normalize holiday input to Dict[name -> List[date]].

    Accepts:
    - list of date strings / date objects / Timestamps  -> {"holiday_0": [...]}
    - dict mapping name -> list of dates
    """
    if isinstance(holiday_dates, dict):
        result: Dict[str, List[datetime.date]] = {}
        for name, dates in holiday_dates.items():
            result[name] = [_to_date(d) for d in dates]
        return result
    else:
        # flat list — group by "calendar date" pattern (month-day)
        # Each unique (month, day) gets its own name
        grouped: Dict[str, List[datetime.date]] = {}
        for i, d in enumerate(holiday_dates):
            date = _to_date(d)
            key = f"holiday_{i}"
            grouped[key] = [date]
        # Merge same (month, day) across years
        merged: Dict[str, List[datetime.date]] = {}
        md_to_name: Dict[tuple, str] = {}
        counter = 0
        for d in [_to_date(x) for x in holiday_dates]:
            md = (d.month, d.day)
            if md not in md_to_name:
                name = f"holiday_{counter}"
                counter += 1
                md_to_name[md] = name
                merged[name] = []
            merged[md_to_name[md]].append(d)
        return merged


def _to_date(d: Union[str, datetime.date, pd.Timestamp]) -> datetime.date:
    """Convert various date formats to datetime.date."""
    if isinstance(d, datetime.date) and not isinstance(d, datetime.datetime):
        return d
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, str):
        return pd.Timestamp(d).date()
    raise TypeError(f"Cannot convert {type(d)} to date")


def dates_to_index_mask(
    index: pd.DatetimeIndex, dates: List[datetime.date]
) -> np.ndarray:
    """Return boolean mask of index positions matching any date in dates."""
    date_set = set(dates)
    return np.array([d.date() in date_set for d in index])


def all_holiday_dates_flat(
    holidays: Dict[str, List[datetime.date]]
) -> List[datetime.date]:
    """Flatten all holiday dates into a single list."""
    result = []
    for dates in holidays.values():
        result.extend(dates)
    return result
