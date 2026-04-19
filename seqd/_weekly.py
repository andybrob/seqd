"""Stage 1: Weekly day-of-week effect extraction."""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._structures import WeeklyEffect
from ._utils import trimmed_mean, ols_slope, all_holiday_dates_flat


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_weekly(
    y: pd.Series,
    holiday_dates_flat: List[datetime.date],
    multiplicative: Optional[bool] = None,
) -> tuple[WeeklyEffect, pd.Series, bool]:
    """Fit and remove weekly day-of-week effects.

    Parameters
    ----------
    y : pd.Series
        Daily series with DatetimeIndex.
    holiday_dates_flat : list of datetime.date
        All holiday dates (used for masking during DOW estimation).
    multiplicative : bool or None
        If None, auto-detect from data.

    Returns
    -------
    weekly_effect : WeeklyEffect
    y_w : pd.Series
        Series with weekly effect removed.
    is_multiplicative : bool
    """
    y = y.copy().astype(float)
    idx = y.index

    # Step 1: rolling median
    T_local = _rolling_median(y, window=7, min_periods=4)

    # Step 2: de-trended series
    # Detect multiplicative/additive
    if multiplicative is None:
        is_mult = _detect_multiplicative(y, T_local)
    else:
        is_mult = multiplicative

    if is_mult:
        # y_dt = y / T_local; guard against zeros/negatives in T_local
        T_safe = T_local.copy()
        T_safe[T_safe <= 0] = np.nan
        y_dt = y / T_safe
    else:
        y_dt = y - T_local

    # Step 3: estimate DOW coefficients (holiday-masked)
    coefficients = _estimate_dow_coefficients(y_dt, idx, holiday_dates_flat, is_mult)

    # Step 4: apply
    y_w = _apply_weekly(y, coefficients, idx, is_mult)

    # Step 5: recency analysis
    recency, drift = _recency_analysis(
        y_dt, idx, holiday_dates_flat, is_mult, coefficients
    )

    weekly = WeeklyEffect(
        coefficients=coefficients,
        is_multiplicative=is_mult,
        recency=recency,
        drift=drift,
    )
    return weekly, y_w, is_mult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rolling_median(y: pd.Series, window: int = 7, min_periods: int = 4) -> pd.Series:
    """7-day centered rolling median with NaN filling."""
    rm = y.rolling(window=window, center=True, min_periods=min_periods).median()
    # Fill NaNs at edges with series mean
    series_mean = float(y.mean())
    rm = rm.fillna(series_mean)
    return rm


def _detect_multiplicative(y: pd.Series, T_local: pd.Series) -> bool:
    """Auto-detect multiplicative vs additive mode.

    We check whether the variance of the detrended residual scales with the
    local level (indicative of multiplicative structure).  Specifically:
    - Compute additive residual y - T_local
    - Compute multiplicative residual y / T_local
    - Check if std(y - T_local) / mean(y) > 0.15, OR
      if the std of the additive residual correlates with T_local (variance
      proportional to level).
    """
    y_vals = y.values.astype(float)
    T_vals = T_local.values.astype(float)
    mean_y = float(np.nanmean(y_vals))
    if mean_y == 0:
        return False

    y_dt = y_vals - T_vals
    std_dt = float(np.nanstd(y_dt))

    # Primary check: overall CV of detrended residual
    if (std_dt / abs(mean_y)) > 0.15:
        return True

    # Secondary check: does the local variance scale with the level?
    # Split into chunks and check if std correlates with mean level
    n = len(y_vals)
    chunk_size = max(30, n // 10)
    n_chunks = n // chunk_size
    if n_chunks >= 4:
        chunk_stds = []
        chunk_means = []
        for i in range(n_chunks):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            r = y_dt[sl]
            t = T_vals[sl]
            chunk_stds.append(float(np.nanstd(r)))
            chunk_means.append(float(np.nanmean(t)))
        chunk_stds_arr = np.array(chunk_stds)
        chunk_means_arr = np.array(chunk_means)
        # Check correlation between local std and local mean
        if np.std(chunk_means_arr) > 0:
            corr = float(np.corrcoef(chunk_means_arr, chunk_stds_arr)[0, 1])
            if corr > 0.7:
                return True

    return False


def _estimate_dow_coefficients(
    y_dt: pd.Series,
    idx: pd.DatetimeIndex,
    holiday_dates_flat: List[datetime.date],
    is_mult: bool,
) -> np.ndarray:
    """Compute 10% trimmed mean per DOW, excluding ±4 days around holidays."""
    holiday_set = set(holiday_dates_flat)
    # Build mask: True = near holiday (exclude)
    exclude_mask = _holiday_proximity_mask(idx, holiday_set, radius=4)

    dow_array = idx.dayofweek  # Monday=0
    values = y_dt.values.astype(float)

    raw_coeff = np.zeros(7)
    for d in range(7):
        mask = (dow_array == d) & (~exclude_mask) & np.isfinite(values)
        vals = values[mask]
        if len(vals) == 0:
            # Fallback: include holiday days
            mask2 = (dow_array == d) & np.isfinite(values)
            vals = values[mask2]
        raw_coeff[d] = trimmed_mean(vals, proportiontocut=0.10)

    # Normalize
    if is_mult:
        mean_coeff = float(np.mean(raw_coeff))
        if mean_coeff == 0:
            mean_coeff = 1.0
        coefficients = raw_coeff / mean_coeff
    else:
        mean_coeff = float(np.mean(raw_coeff))
        coefficients = raw_coeff - mean_coeff

    return coefficients


def _holiday_proximity_mask(
    idx: pd.DatetimeIndex, holiday_set: set, radius: int = 4
) -> np.ndarray:
    """Boolean mask: True if index date is within `radius` days of any holiday."""
    dates = np.array([d.date() for d in idx])
    mask = np.zeros(len(dates), dtype=bool)
    for h in holiday_set:
        for delta in range(-radius, radius + 1):
            target = h + datetime.timedelta(days=delta)
            mask |= dates == target
    return mask


def _apply_weekly(
    y: pd.Series,
    coefficients: np.ndarray,
    idx: pd.DatetimeIndex,
    is_mult: bool,
) -> pd.Series:
    """Remove weekly effect from y."""
    dow = idx.dayofweek
    effect = coefficients[dow]
    if is_mult:
        # Avoid division by zero
        safe_effect = np.where(np.abs(effect) < 1e-10, 1.0, effect)
        y_w = y.values / safe_effect
    else:
        y_w = y.values - effect
    return pd.Series(y_w, index=idx, name=y.name)


def _recency_analysis(
    y_dt: pd.Series,
    idx: pd.DatetimeIndex,
    holiday_dates_flat: List[datetime.date],
    is_mult: bool,
    full_coeff: np.ndarray,
) -> tuple[dict, dict]:
    """Compute recency trailing-window DOW coefficients and drift."""
    windows = [60, 90, 365]
    recency: Dict[int, pd.DataFrame] = {}
    # For drift: collect coefficients over time for each window
    # We need multiple window endpoints to estimate drift
    # Step every 7 days

    holiday_set = set(holiday_dates_flat)
    n = len(idx)
    values = y_dt.values.astype(float)
    dow_array = idx.dayofweek

    # Collect trailing window endpoints
    endpoints = list(range(n - 1, -1, -7))
    endpoints = sorted(endpoints)  # ascending

    for w in windows:
        # Skip windows larger than the series length — not enough data
        if w > n:
            continue
        rows = []
        for end in endpoints:
            start = max(0, end - w + 1)
            sub_idx = idx[start : end + 1]
            sub_vals = values[start : end + 1]
            sub_dow = dow_array[start : end + 1]
            exclude_mask = _holiday_proximity_mask(sub_idx, holiday_set, radius=4)

            coeff_row = {"date": idx[end]}
            for d in range(7):
                mask = (
                    (sub_dow == d)
                    & (~exclude_mask)
                    & np.isfinite(sub_vals)
                )
                vals = sub_vals[mask]
                if len(vals) < 3:
                    mask2 = (sub_dow == d) & np.isfinite(sub_vals)
                    vals = sub_vals[mask2]
                tm = trimmed_mean(vals, proportiontocut=0.10) if len(vals) > 0 else np.nan
                coeff_row[f"dow_{d}"] = tm

            # Normalize within window
            dow_vals = np.array([coeff_row.get(f"dow_{d}", np.nan) for d in range(7)])
            if is_mult:
                mean_v = float(np.nanmean(dow_vals))
                if mean_v == 0 or np.isnan(mean_v):
                    mean_v = 1.0
                dow_vals_norm = dow_vals / mean_v
            else:
                mean_v = float(np.nanmean(dow_vals))
                dow_vals_norm = dow_vals - mean_v
            for d in range(7):
                coeff_row[f"dow_{d}"] = float(dow_vals_norm[d]) if np.isfinite(dow_vals_norm[d]) else float(full_coeff[d])

            rows.append(coeff_row)

        if rows:
            df = pd.DataFrame(rows)
            recency[w] = df

    # Drift: use 365-day window if available, else largest
    ref_window = 365 if 365 in recency else max(windows)
    drift: Dict[int, Dict] = {}
    if ref_window in recency:
        df_ref = recency[ref_window]
        dates_num = np.arange(len(df_ref), dtype=float)
        for d in range(7):
            col = f"dow_{d}"
            if col in df_ref.columns:
                slope = ols_slope(dates_num, df_ref[col].values.astype(float))
                fc = full_coeff[d]
                threshold = 0.01 * abs(fc) if fc != 0 else 1e-6
                classification = "drifting" if abs(slope) > threshold else "stable"
                drift[d] = {"slope": slope, "classification": classification}
            else:
                drift[d] = {"slope": 0.0, "classification": "stable"}
    else:
        for d in range(7):
            drift[d] = {"slope": 0.0, "classification": "stable"}

    return recency, drift
