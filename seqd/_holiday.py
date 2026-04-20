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
    max_holiday_merge_gap_days: int = 7,
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
    max_holiday_merge_gap_days : int
        Two HolidayEffect occurrences are merged into a compound block only when
        the minimum gap between any holiday date in one group and any holiday date
        in the other group is ≤ this value (in days).  Default 7.  This criterion
        is based on calendar proximity of the actual holiday dates, not on whether
        their ramp windows overlap — so it is stable with respect to holiday_window
        size.  Set to 3 to merge only back-to-back holidays; set to 30 to merge
        all of BFCM plus Christmas into one block.

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
    merged_effects = _merge_overlapping(
        all_occurrences, idx, idx_dates,
        max_holiday_merge_gap_days=max_holiday_merge_gap_days,
    )

    # Build HolidayEffect objects grouped by holiday name
    holiday_effects = _build_holiday_effects(
        occurrence_data, merged_effects, idx, idx_dates, y_w,
        holiday_window=holiday_window,
        series_start=y_w.index[0].date(),
        series_end=y_w.index[-1].date(),
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

    # Cap ramp_end at the last date in the series.  _detect_ramp_end uses
    # day_to_residual.get(offset, 0.0), which returns 0 for offsets beyond the
    # data boundary and can trigger a spurious "baseline return" one day past the
    # last observation.  Clamp here to avoid metadata overrun.
    series_last_date = idx_dates[-1]
    if ramp_end > series_last_date:
        ramp_end = series_last_date

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
    """Backward CUSUM from h to detect ramp start using dual criteria.

    Criterion 1 (CUSUM): cumulative excess of |r(t)| over sigma_ref exceeds
    2 * sigma_ref.  This is sensitive to sharp, concentrated ramps.

    Criterion 2 (run): 4 or more consecutive days where |r(t)| > 0.5 * sigma_ref.
    This is sensitive to gradual, low-amplitude multi-week ramps.

    ramp_start is the *earliest* date (farthest from h) where either criterion
    first fires while scanning backwards.  If neither fires, returns h_date
    (spike only).
    """
    cusum_threshold = 2.0 * sigma_ref
    run_threshold = 0.5 * sigma_ref
    run_required = 4

    # CUSUM criterion: scan backwards accumulating excess
    S = 0.0
    cusum_ramp_start = h_date
    last_reset = h_date

    for delta in range(0, holiday_window + 1):
        t = h_date - datetime.timedelta(days=delta)
        offset = -delta
        r = day_to_residual.get(offset, 0.0)
        increment = abs(r) - sigma_ref
        new_S = S + increment
        if new_S <= 0:
            S = 0.0
            last_reset = t - datetime.timedelta(days=1)
        else:
            S = new_S
            if S > cusum_threshold:
                cusum_ramp_start = last_reset + datetime.timedelta(days=1)

    # Run criterion: find earliest point in any run of 4+ consecutive elevated days.
    # Scanning backward from h_date, each step to a larger delta is an earlier date.
    # current_run_start is updated on every elevated step so it always holds the
    # farthest-back (earliest) point reached in the current run.
    run_ramp_start = h_date  # default: no useful detection
    best_run_start = None
    current_run_start = None
    current_run_length = 0

    for delta in range(0, holiday_window + 1):
        t = h_date - datetime.timedelta(days=delta)
        offset = -delta
        if offset not in day_to_residual:
            # Break any current run
            if current_run_length >= run_required:
                if best_run_start is None or current_run_start < best_run_start:
                    best_run_start = current_run_start
            current_run_length = 0
            current_run_start = None
            continue

        r = day_to_residual[offset]
        if abs(r) > run_threshold:
            current_run_length += 1
            # Each step backward is an earlier date — always update run start
            current_run_start = t
        else:
            # End of run
            if current_run_length >= run_required:
                if best_run_start is None or current_run_start < best_run_start:
                    best_run_start = current_run_start
            current_run_length = 0
            current_run_start = None

    # Handle run still active at end of search window
    if current_run_length >= run_required:
        if best_run_start is None or current_run_start < best_run_start:
            best_run_start = current_run_start

    if best_run_start is not None:
        run_ramp_start = best_run_start

    # Use whichever criterion gives the earlier (farther back) ramp_start
    if run_ramp_start < cusum_ramp_start:
        return run_ramp_start
    return cusum_ramp_start


def _detect_ramp_end(
    h_date: datetime.date,
    day_to_residual: dict,
    sigma_ref: float,
    holiday_window: int,
) -> datetime.date:
    """Forward scan from h+1 to find baseline return.

    Applies a 3-point median smoothing to the residuals before checking the
    consecutive-pair threshold criterion. This reduces false early exits caused
    by isolated noisy days within an otherwise active ramp.
    """
    threshold_return = 1.5 * sigma_ref
    consecutive = 0
    # Default: full search window (not a hardcoded 7-day fallback).
    # If no recovery is detected within holiday_window days the effect is
    # considered to persist through the entire search span.
    ramp_end = h_date + datetime.timedelta(days=holiday_window)

    for delta in range(1, holiday_window + 1):
        t = h_date + datetime.timedelta(days=delta)
        offset = delta
        # 3-point median smoothing: median of [r(t-1), r(t), r(t+1)]
        # Boundary values use the existing .get(offset, 0.0) convention
        r_prev = day_to_residual.get(offset - 1, 0.0)
        r_curr = day_to_residual.get(offset, 0.0)
        r_next = day_to_residual.get(offset + 1, 0.0)
        r_smooth = float(np.median([r_prev, r_curr, r_next]))
        if abs(r_smooth) < threshold_return:
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
    max_holiday_merge_gap_days: int = 7,
) -> List[dict]:
    """Merge holiday occurrences into compound blocks based on calendar proximity.

    Two groups are merged if and only if the minimum gap between any holiday date
    in one group and any holiday date in the other group is ≤
    ``max_holiday_merge_gap_days``.  This criterion is stable with respect to
    ``holiday_window`` size — it depends only on the actual calendar dates of the
    holidays, not on how wide their ramp windows are.

    Examples (default max_holiday_merge_gap_days=7):
    - Thanksgiving (Nov 26) + Black Friday (Nov 27): gap=1 → MERGE
    - Black Friday (Nov 27) + Cyber Monday (Nov 30): gap=3 → MERGE
    - Cyber Monday (Nov 30) + Christmas (Dec 25): gap=25 → DO NOT MERGE
    - Christmas (Dec 25) + New Year's Eve (Dec 31): gap=6 → MERGE
    """
    if not all_occurrences:
        return []

    # Build event list sorted by holiday date
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
    events.sort(key=lambda e: e["date"])

    # Group events: each group accumulates a set of holiday dates.
    # A new event joins the current group iff its holiday date is within
    # max_holiday_merge_gap_days of the *nearest* date already in the group.
    merged = []
    current = events[0].copy()
    current["merged_from"] = [events[0]]
    current["_group_dates"] = {events[0]["date"]}

    for ev in events[1:]:
        ev_date = ev["date"]
        # Minimum gap between ev_date and all dates already in the current group
        min_gap = min(
            abs((ev_date - d).days) for d in current["_group_dates"]
        )
        if min_gap <= max_holiday_merge_gap_days:
            # Close enough — merge into current group
            current["ramp_end"] = max(current["ramp_end"], ev["ramp_end"])
            current["ramp_start"] = min(current["ramp_start"], ev["ramp_start"])
            current["merged_from"].append(ev)
            current["_group_dates"].add(ev_date)
            # Accumulate sum into a running total; divide by n_members at end.
            current["_effect_sum"] = (
                current.get("_effect_sum",
                            current["effect_series"].reindex(idx, fill_value=0.0))
                + ev["effect_series"].reindex(idx, fill_value=0.0)
            )
        else:
            # Too far apart — close the current group and start a new one
            merged.append(current)
            current = ev.copy()
            current["merged_from"] = [ev]
            current["_group_dates"] = {ev_date}

    merged.append(current)

    # Clean up the temporary _group_dates field
    for block in merged:
        block.pop("_group_dates", None)

    # For merged blocks with multiple events, finalise effect_series as the MEAN
    # of the constituent individual effects (not the sum), then recompute magnitude.
    # Using the mean means block["effect_series"] is the canonical per-block effect,
    # and only the PRIMARY member will carry it — no division needed downstream.
    for block in merged:
        if len(block["merged_from"]) > 1:
            n_members = len(block["merged_from"])
            # "_effect_sum" was accumulated in the merge loop above
            effect_sum = block.pop("_effect_sum")
            block["effect_series"] = effect_sum / n_members
            # Recompute magnitude as mean over merged window from the canonical effect
            rs = block["ramp_start"]
            re = block["ramp_end"]
            mask = (idx_dates >= rs) & (idx_dates <= re)
            vals = block["effect_series"].values[mask]
            block["magnitude"] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return merged


# ---------------------------------------------------------------------------
# Build HolidayEffect objects
# ---------------------------------------------------------------------------


def _precompute_name_stats(
    occurrence_data: Dict[str, List[dict]],
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Compute per-holiday-name year_magnitudes lists and OLS drift slopes.

    Parameters
    ----------
    occurrence_data : dict
        Mapping from holiday name to list of occurrence dicts (each with a
        ``"magnitude"`` key), as produced by the per-holiday fitting stage.

    Returns
    -------
    name_year_magnitudes : dict
        Mapping from holiday name to sorted list of magnitudes (one per year).
    name_drift : dict
        Mapping from holiday name to OLS slope of magnitudes over year index.
        Zero when fewer than 2 occurrences exist.
    """
    name_year_magnitudes: Dict[str, List[float]] = {}
    name_drift: Dict[str, float] = {}
    for name, occ_list in occurrence_data.items():
        if not occ_list:
            continue
        sorted_occs = sorted(occ_list, key=lambda o: o["date"])
        mags = [occ["magnitude"] for occ in sorted_occs]
        name_year_magnitudes[name] = mags
        if len(mags) >= 2:
            x = np.arange(len(mags), dtype=float)
            slope = ols_slope(x, np.array(mags, dtype=float))
        else:
            slope = 0.0
        name_drift[name] = slope
    return name_year_magnitudes, name_drift


def _assign_compound_roles(
    merged_effects: List[dict],
) -> Tuple[Dict[datetime.date, dict], Dict[int, datetime.date]]:
    """Map occurrence dates to their merged blocks and identify primary members.

    A compound block's PRIMARY member is the occurrence with the earliest
    calendar date.  Only the primary member carries the canonical block
    ``effect_series`` (mean of all constituents); non-primary members receive
    a zero series to prevent double-counting in ``holiday_component()``.

    Parameters
    ----------
    merged_effects : list of dict
        Merged block dicts from ``_merge_overlapping``.

    Returns
    -------
    date_to_merged : dict
        Mapping from each original occurrence date to its containing merged block.
    block_primary_date : dict
        Mapping from ``id(block)`` to the primary (earliest) occurrence date for
        compound blocks (blocks with ``len(merged_from) > 1``).  Non-compound
        blocks are absent from this dict.
    """
    date_to_merged: Dict[datetime.date, dict] = {}
    for block in merged_effects:
        for src in block["merged_from"]:
            date_to_merged[src["date"]] = block

    block_primary_date: Dict[int, datetime.date] = {}
    for block in merged_effects:
        if len(block["merged_from"]) > 1:
            earliest = min(src["date"] for src in block["merged_from"])
            block_primary_date[id(block)] = earliest

    return date_to_merged, block_primary_date


def _build_holiday_effects(
    occurrence_data: Dict[str, List[dict]],
    merged_effects: List[dict],
    idx: pd.DatetimeIndex,
    idx_dates: np.ndarray,
    y_w: pd.Series,
    holiday_window: int = 14,
    series_start: Optional[datetime.date] = None,
    series_end: Optional[datetime.date] = None,
) -> List[HolidayEffect]:
    """Build final HolidayEffect objects — one per occurrence (year), with recency drift.

    Each occurrence of a holiday (i.e. each year) gets its own HolidayEffect so
    that holiday_component() has non-zero values across *all* years, not just the
    most recent one.  year_magnitudes on each HolidayEffect lists the magnitudes
    of all occurrences of the same holiday name (for drift tracking).

    Delegates to:
    - :func:`_precompute_name_stats` — per-holiday year_magnitudes and drift slopes.
    - :func:`_assign_compound_roles` — maps occurrence dates to merged blocks and
      identifies primary compound-block members.
    """
    result = []

    # Build date→block map and compound primary-date map
    date_to_merged, block_primary_date = _assign_compound_roles(merged_effects)

    # Also build a block -> compound_block_id mapping (assigned on first encounter)
    block_to_id: Dict[int, str] = {}
    block_counter = [0]

    # Precompute per-name year_magnitudes and drift slopes
    name_year_magnitudes, name_drift = _precompute_name_stats(occurrence_data)

    # Build one HolidayEffect per occurrence
    for name, occ_list in occurrence_data.items():
        if not occ_list:
            continue

        year_magnitudes = name_year_magnitudes[name]
        slope = name_drift[name]

        for occ in sorted(occ_list, key=lambda o: o["date"]):
            h_date = occ["date"]
            block = date_to_merged.get(h_date)

            # Compound block detection: block has multiple sources
            is_compound = block is not None and len(block["merged_from"]) > 1

            # Assign a stable compound_block_id per block object
            compound_block_id: Optional[str] = None
            if is_compound and block is not None:
                bid = id(block)
                if bid not in block_to_id:
                    # Use the year from the first date in this block
                    first_year = block["merged_from"][0]["date"].year
                    block_counter[0] += 1
                    block_to_id[bid] = f"compound_block_{first_year}_{block_counter[0]}"
                compound_block_id = block_to_id[id(block)]

            # Ramp bounds: use block-level if merged, otherwise occurrence-level.
            if block is not None:
                rs = block["ramp_start"]
                re = block["ramp_end"]
                mag = block["magnitude"]
            else:
                rs = occ["ramp_start"]
                re = occ["ramp_end"]
                mag = occ["magnitude"]

            # effect_series assignment:
            # - Non-compound occurrence: use the individual occurrence effect.
            # - Compound block PRIMARY member (earliest date): carry the
            #   canonical block effect_series (mean of all constituent effects).
            # - Compound block NON-PRIMARY member: zero series — so that
            #   holiday_component() sums exactly ONE copy of the canonical
            #   effect per compound block with no double-counting.
            if is_compound and block is not None:
                primary_date = block_primary_date[id(block)]
                if h_date == primary_date:
                    effect_series = block["effect_series"].reindex(idx, fill_value=0.0)
                else:
                    effect_series = pd.Series(0.0, index=idx)
            else:
                effect_series = occ["effect_series"].reindex(idx, fill_value=0.0)

            # individual_peak_magnitude: mean residual in ±3 days around h_date
            # (uses occurrence-level effect_series before merge)
            h_minus3 = h_date - datetime.timedelta(days=3)
            h_plus3 = h_date + datetime.timedelta(days=3)
            local_mask = (idx_dates >= h_minus3) & (idx_dates <= h_plus3)
            local_vals = occ["effect_series"].values[local_mask]
            indiv_peak = float(np.mean(local_vals)) if len(local_vals) > 0 else None

            # individual_peak_magnitude_reliable: False if ±3 day window exceeds series
            # bounds, OR if the occurrence's ramp_end reaches the series end (meaning no
            # post-holiday reference window exists to anchor the baseline, so the residual
            # within the ±3 window is not reliably separated from the trend).
            if series_start is not None and series_end is not None:
                ipm_reliable = (
                    (h_minus3 >= series_start)
                    and (h_plus3 <= series_end)
                    and (occ["ramp_end"] < series_end)
                )
            else:
                ipm_reliable = True

            # ramp_start_ceiling_hit: True if ramp_start reached the search boundary
            ceiling_date = occ["date"] - datetime.timedelta(days=holiday_window)
            occ_ramp_start = occ["ramp_start"]
            ceiling_hit = (occ_ramp_start <= ceiling_date + datetime.timedelta(days=1))

            he = HolidayEffect(
                date=h_date,
                name=name,
                ramp_start=rs,
                ramp_end=re,
                magnitude=mag,
                effect_series=effect_series,
                year_magnitudes=year_magnitudes,
                magnitude_drift=slope,
                compound=is_compound,
                compound_block_id=compound_block_id,
                individual_peak_magnitude=indiv_peak,
                ramp_start_ceiling_hit=ceiling_hit,
                individual_peak_magnitude_reliable=ipm_reliable,
            )
            result.append(he)

    return result
