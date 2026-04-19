"""Stage 2: Holiday ramp detection and effect extraction."""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._structures import HolidayEffect
from ._utils import mad_sigma, ols_fit, ols_slope


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_holidays(
    y_w: pd.Series,
    holidays: Dict[str, List[datetime.date]],
    holiday_window: int = 14,
    reference_window: int = 60,
    reference_gap: int = 14,
) -> Tuple[List[HolidayEffect], pd.Series]:
    """Fit and remove holiday effects from the weekly-adjusted series.

    Parameters
    ----------
    y_w : pd.Series
        Weekly-adjusted series with DatetimeIndex.
    holidays : dict
        Mapping from holiday name to list of datetime.date instances.
    holiday_window : int
        Number of days before/after holiday to search for ramp (±holiday_window).
    reference_window : int
        Days before/after the gap used as reference baseline.
    reference_gap : int
        Days before/after holiday excluded from baseline estimation.

    Returns
    -------
    holiday_effects : list of HolidayEffect
    y_h : pd.Series
        Series with all holiday effects removed.
    """
    y_w = y_w.copy().astype(float)
    idx = y_w.index
    idx_dates = np.array([d.date() for d in idx])

    # Collect all holiday dates for cross-masking
    all_holiday_dates_flat = []
    for dates in holidays.values():
        all_holiday_dates_flat.extend(dates)
    all_holiday_set = set(all_holiday_dates_flat)

    # Process each holiday occurrence
    # First pass: compute individual occurrence effects
    # We'll track: name -> list of (date, ramp_start, ramp_end, magnitude, effect_array)
    occurrence_data: Dict[str, List[dict]] = {}

    for name, dates in holidays.items():
        occurrence_data[name] = []
        for h_date in sorted(dates):
            result = _process_one_holiday(
                y_w=y_w,
                idx=idx,
                idx_dates=idx_dates,
                h_date=h_date,
                all_holiday_set=all_holiday_set,
                holiday_window=holiday_window,
                reference_window=reference_window,
                reference_gap=reference_gap,
            )
            if result is not None:
                occurrence_data[name].append(result)

    # Handle overlapping ramp windows
    # Collect all (ramp_start, ramp_end, effect_array, name, h_date) across all occurrences
    all_occurrences = []
    for name, occ_list in occurrence_data.items():
        for occ in occ_list:
            all_occurrences.append((name, occ))

    # Merge overlapping windows
    merged_effects = _merge_overlapping(all_occurrences, idx, idx_dates)

    # Build HolidayEffect objects grouped by holiday name
    holiday_effects = _build_holiday_effects(
        occurrence_data, merged_effects, idx, idx_dates, y_w
    )

    # Sum all effects and remove from y_w
    total_effect = np.zeros(len(y_w))
    for he in holiday_effects:
        aligned = he.effect_series.reindex(idx, fill_value=0.0).values
        total_effect += aligned

    y_h = pd.Series(y_w.values - total_effect, index=idx, name=y_w.name)
    return holiday_effects, y_h


# ---------------------------------------------------------------------------
# Per-holiday processing
# ---------------------------------------------------------------------------


def _process_one_holiday(
    y_w: pd.Series,
    idx: pd.DatetimeIndex,
    idx_dates: np.ndarray,
    h_date: datetime.date,
    all_holiday_set: set,
    holiday_window: int,
    reference_window: int,
    reference_gap: int,
) -> Optional[dict]:
    """Process one holiday occurrence. Returns dict or None if insufficient data."""
    # Reference window: [h-reference_window, h-reference_gap] U [h+reference_gap, h+reference_window]
    ref_start_pre = h_date - datetime.timedelta(days=reference_window)
    ref_end_pre = h_date - datetime.timedelta(days=reference_gap)
    ref_start_post = h_date + datetime.timedelta(days=reference_gap)
    ref_end_post = h_date + datetime.timedelta(days=reference_window)

    # Build reference mask (exclude other holidays)
    ref_mask = (
        (
            ((idx_dates >= ref_start_pre) & (idx_dates <= ref_end_pre))
            | ((idx_dates >= ref_start_post) & (idx_dates <= ref_end_post))
        )
    )

    # Exclude dates near other holidays (but not h_date itself — it's in search span anyway)
    other_holidays = all_holiday_set - {h_date}
    for oh in other_holidays:
        near_oh = np.array([
            abs((d - oh).days) <= 3 for d in idx_dates
        ])
        ref_mask &= ~near_oh

    ref_indices = np.where(ref_mask)[0]
    if len(ref_indices) < 10:
        return None

    # Fit OLS linear trend on reference window
    t_ref = np.array([
        (idx_dates[i] - h_date).days for i in ref_indices
    ], dtype=float)
    y_ref = y_w.iloc[ref_indices].values.astype(float)

    # Design matrix: [1, t]
    X_ref = np.column_stack([np.ones_like(t_ref), t_ref])
    try:
        coef = ols_fit(X_ref, y_ref)
    except Exception:
        coef = np.array([float(np.nanmean(y_ref)), 0.0])

    # Search span: [h - holiday_window, h + holiday_window]
    search_start = h_date - datetime.timedelta(days=holiday_window)
    search_end = h_date + datetime.timedelta(days=holiday_window)
    search_mask = (idx_dates >= search_start) & (idx_dates <= search_end)
    search_indices = np.where(search_mask)[0]

    if len(search_indices) == 0:
        return None

    # Baseline and residuals over search span
    t_search = np.array([
        (idx_dates[i] - h_date).days for i in search_indices
    ], dtype=float)
    B_search = coef[0] + coef[1] * t_search
    y_search = y_w.iloc[search_indices].values.astype(float)
    r_search = y_search - B_search

    # sigma_ref: MAD of residuals on reference window
    r_ref = y_ref - (coef[0] + coef[1] * t_ref)
    sigma_ref = mad_sigma(r_ref)
    if sigma_ref == 0 or np.isnan(sigma_ref):
        sigma_ref = float(np.nanstd(r_ref)) or 1.0

    # Map from day-offset to residual
    day_to_residual = {}
    for k, si in enumerate(search_indices):
        day_offset = int(round((idx_dates[si] - h_date).days))
        day_to_residual[day_offset] = r_search[k]

    # Ramp-start detection: backward CUSUM from h
    ramp_start = _detect_ramp_start(
        h_date, day_to_residual, sigma_ref, holiday_window
    )

    # Ramp-end detection: forward scan from h+1
    ramp_end = _detect_ramp_end(
        h_date, day_to_residual, sigma_ref, holiday_window
    )

    # Magnitude: mean residual over [ramp_start, ramp_end]
    ramp_days = []
    cur = ramp_start
    while cur <= ramp_end:
        offset = (cur - h_date).days
        if offset in day_to_residual:
            ramp_days.append(day_to_residual[offset])
        cur += datetime.timedelta(days=1)

    if len(ramp_days) == 0:
        # Use holiday day itself
        offset = 0
        if offset in day_to_residual:
            ramp_days = [day_to_residual[offset]]
        else:
            return None

    magnitude = float(np.mean(ramp_days))

    # Build effect_series: r(t) within ramp window, 0 elsewhere
    effect_array = np.zeros(len(idx))
    for k, si in enumerate(search_indices):
        d = idx_dates[si]
        if ramp_start <= d <= ramp_end:
            effect_array[si] = r_search[k]

    effect_series = pd.Series(effect_array, index=idx)

    return {
        "date": h_date,
        "ramp_start": ramp_start,
        "ramp_end": ramp_end,
        "magnitude": magnitude,
        "effect_series": effect_series,
    }


def _detect_ramp_start(
    h_date: datetime.date,
    day_to_residual: dict,
    sigma_ref: float,
    holiday_window: int,
) -> datetime.date:
    """Backward CUSUM from h to detect ramp start.

    Strategy: scan backward from the holiday date.  We accumulate excess signal
    (|r(t)| - sigma_ref) via a standard CUSUM with reset.  The threshold is
    3 * sigma_ref (tight enough to avoid false detections from pure noise over
    14 steps).  The ramp_start is the *latest* (closest to h) date where the
    CUSUM resets — i.e., the beginning of the contiguous elevated-signal run
    that includes the holiday.  If the CUSUM never exceeds the threshold, we
    return h_date (spike only).
    """
    threshold = 3.0 * sigma_ref
    S = 0.0
    ramp_start = h_date  # default: spike only
    last_reset = h_date  # track where CUSUM last reset to zero

    for delta in range(0, holiday_window + 1):
        t = h_date - datetime.timedelta(days=delta)
        offset = -delta
        r = day_to_residual.get(offset, 0.0)
        increment = abs(r) - sigma_ref
        new_S = S + increment
        if new_S <= 0:
            S = 0.0
            last_reset = t - datetime.timedelta(days=1)  # reset point just before this
        else:
            S = new_S
            if S > threshold:
                # The ramp starts at last_reset + 1 day (first day after the reset)
                ramp_start = last_reset + datetime.timedelta(days=1)

    return ramp_start


def _detect_ramp_end(
    h_date: datetime.date,
    day_to_residual: dict,
    sigma_ref: float,
    holiday_window: int,
) -> datetime.date:
    """Forward scan from h+1 to find baseline return."""
    threshold_return = 1.5 * sigma_ref
    consecutive = 0
    ramp_end = h_date + datetime.timedelta(days=7)  # default

    for delta in range(1, holiday_window + 1):
        t = h_date + datetime.timedelta(days=delta)
        offset = delta
        r = day_to_residual.get(offset, 0.0)
        if abs(r) < threshold_return:
            consecutive += 1
            if consecutive >= 2:
                # ramp_end is 2 days back (first of the two consecutive small days)
                ramp_end = t - datetime.timedelta(days=1)
                return ramp_end
        else:
            consecutive = 0

    return ramp_end


# ---------------------------------------------------------------------------
# Overlap handling
# ---------------------------------------------------------------------------


def _merge_overlapping(
    all_occurrences: list,
    idx: pd.DatetimeIndex,
    idx_dates: np.ndarray,
) -> List[dict]:
    """Merge overlapping ramp windows into compound blocks."""
    if not all_occurrences:
        return []

    # Sort by ramp_start
    events = []
    for name, occ in all_occurrences:
        events.append({
            "name": name,
            "date": occ["date"],
            "ramp_start": occ["ramp_start"],
            "ramp_end": occ["ramp_end"],
            "magnitude": occ["magnitude"],
            "effect_series": occ["effect_series"],
        })
    events.sort(key=lambda e: e["ramp_start"])

    merged = []
    current = events[0].copy()
    current["merged_from"] = [events[0]]

    for ev in events[1:]:
        if ev["ramp_start"] <= current["ramp_end"]:
            # Overlapping — merge
            current["ramp_end"] = max(current["ramp_end"], ev["ramp_end"])
            current["ramp_start"] = min(current["ramp_start"], ev["ramp_start"])
            current["merged_from"].append(ev)
            # Recompute magnitude as mean over merged window
            # Effect series: sum both
            current["effect_series"] = (
                current["effect_series"].reindex(idx, fill_value=0.0)
                + ev["effect_series"].reindex(idx, fill_value=0.0)
            )
        else:
            merged.append(current)
            current = ev.copy()
            current["merged_from"] = [ev]

    merged.append(current)

    # For merged blocks with multiple events, recompute magnitude
    for block in merged:
        if len(block["merged_from"]) > 1:
            # Recompute magnitude as mean over merged window from combined effect
            rs = block["ramp_start"]
            re = block["ramp_end"]
            mask = (idx_dates >= rs) & (idx_dates <= re)
            vals = block["effect_series"].values[mask]
            block["magnitude"] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return merged


# ---------------------------------------------------------------------------
# Build HolidayEffect objects
# ---------------------------------------------------------------------------


def _build_holiday_effects(
    occurrence_data: Dict[str, List[dict]],
    merged_effects: List[dict],
    idx: pd.DatetimeIndex,
    idx_dates: np.ndarray,
    y_w: pd.Series,
) -> List[HolidayEffect]:
    """Build final HolidayEffect objects with recency drift."""
    result = []

    # Map each original occurrence date to its merged block
    date_to_merged: Dict[datetime.date, dict] = {}
    for block in merged_effects:
        for src in block["merged_from"]:
            date_to_merged[src["date"]] = block

    # For each holiday name, build a single HolidayEffect using the *most recent* occurrence
    # but aggregate recency from all occurrences
    for name, occ_list in occurrence_data.items():
        if not occ_list:
            continue

        # Sort by date
        occ_list = sorted(occ_list, key=lambda o: o["date"])

        # Year magnitudes from each occurrence
        year_magnitudes = [occ["magnitude"] for occ in occ_list]

        # Magnitude drift: OLS slope over year indices
        if len(year_magnitudes) >= 2:
            x = np.arange(len(year_magnitudes), dtype=float)
            slope = ols_slope(x, np.array(year_magnitudes, dtype=float))
        else:
            slope = 0.0

        # Use most recent occurrence for the primary HolidayEffect
        # but combine effect_series from all occurrences
        combined_effect = pd.Series(0.0, index=idx)
        seen_blocks = set()
        for occ in occ_list:
            h_date = occ["date"]
            block = date_to_merged.get(h_date)
            if block is None:
                continue
            block_id = id(block)
            if block_id not in seen_blocks:
                seen_blocks.add(block_id)
                aligned = block["effect_series"].reindex(idx, fill_value=0.0)
                combined_effect = combined_effect + aligned

        # Primary occurrence (most recent)
        primary = occ_list[-1]
        primary_block = date_to_merged.get(primary["date"])
        if primary_block is None:
            rs = primary["ramp_start"]
            re = primary["ramp_end"]
            mag = primary["magnitude"]
        else:
            rs = primary_block["ramp_start"]
            re = primary_block["ramp_end"]
            mag = primary_block["magnitude"]

        he = HolidayEffect(
            date=primary["date"],
            name=name,
            ramp_start=rs,
            ramp_end=re,
            magnitude=mag,
            effect_series=combined_effect,
            year_magnitudes=year_magnitudes,
            magnitude_drift=slope,
        )
        result.append(he)

    return result
