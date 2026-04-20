"""Stage 6: Out-of-sample forecasting from fitted V1 decomposition."""

from __future__ import annotations

import datetime as _dt
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._structures import DecompositionResult, ForecastResult, SegmentTrend
from ._trend import evaluate_segment

PERIOD = 365.25  # annual period in days (matches _annual.py)


# ---------------------------------------------------------------------------
# Trend projection
# ---------------------------------------------------------------------------


def _slope_at_t1_per_day(segment: SegmentTrend) -> float:
    """Return the slope of ``segment`` at t=1, in units of value/day.

    The SegmentTrend parameters are in normalised t-space [0, 1].
    Converting to per-day units requires dividing by (n_obs - 1).

    Model-specific derivatives at t=1:

    - linear:     ``beta / (n_obs - 1)``
    - log:        ``beta * T_days / (1 + 1.0 * T_days) / (n_obs - 1)``
    - exp:        ``alpha * beta * exp(beta) / (n_obs - 1)``
    - quadratic:  ``(beta + 2 * gamma) / (n_obs - 1)``
    - constant:   0.0

    Parameters
    ----------
    segment : SegmentTrend
        A fitted segment.

    Returns
    -------
    float
        Slope in value-per-day units.
    """
    n_j = segment.n_obs
    if n_j <= 1:
        return 0.0

    denom = float(n_j - 1)
    model = segment.model_type

    if model == "linear":
        return segment.beta / denom

    if model == "log":
        T = float(segment.T_days)
        # d/dt [beta * ln(1 + t * T)] at t=1 = beta * T / (1 + T)
        # in normalised t-space, then divide by (n_j - 1) for per-day units
        slope_norm = segment.beta * T / (1.0 + T)
        return slope_norm / denom

    if model == "exp":
        # d/dt [alpha * exp(beta * t)] at t=1 = alpha * beta * exp(beta)
        slope_norm = segment.alpha * segment.beta * np.exp(segment.beta)
        return slope_norm / denom

    if model == "quadratic":
        gamma = segment.gamma if segment.gamma is not None else 0.0
        # d/dt [alpha + beta*t + gamma*t^2] at t=1 = beta + 2*gamma
        slope_norm = segment.beta + 2.0 * gamma
        return slope_norm / denom

    # constant or unknown
    return 0.0


def _project_trend(
    last_segment: SegmentTrend,
    horizon: int,
    max_extrapolation_days: int,
    penultimate_segment: Optional["SegmentTrend"] = None,
    slope_blend_alpha: float = 0.0,
) -> tuple[np.ndarray, List[str]]:
    """Project the last segment's trend forward for ``horizon`` days.

    **Blended linear extrapolation** (H1 fix):

    When ``penultimate_segment`` is provided and ``slope_blend_alpha > 0``,
    the extrapolation slope is blended between the penultimate and final
    segment slopes (both at t=1 in their own coordinate systems):

        blended_slope = alpha * slope_penultimate + (1 - alpha) * slope_final

    The trend is then projected as a linear continuation from the final
    segment's end value using the blended slope:

        trend[i] = baseline + blended_slope * (i + 1)   for i = 0, 1, ..., horizon-1

    where ``baseline = segment.predict(t=1)``.  This replaces the previous
    model-based quadratic/exp clamping for extrapolation.

    Parameters
    ----------
    last_segment : SegmentTrend
        The final fitted segment (Stage 5 output).
    horizon : int
        Number of forecast steps.
    max_extrapolation_days : int
        Warn if horizon exceeds this value.
    penultimate_segment : SegmentTrend or None
        The second-to-last segment.  When provided and
        ``slope_blend_alpha > 0``, enables blended linear extrapolation.
    slope_blend_alpha : float
        Weight on penultimate slope (0.0 = use final slope only;
        1.0 = use penultimate slope only).  Default 0.0 (no blending,
        backward-compatible with old clamping behaviour).

    Returns
    -------
    trend_array : np.ndarray
        Shape ``(horizon,)``.
    emitted_warnings : list of str
        Warning messages emitted during this call (also raised via
        ``warnings.warn``).
    """
    n_j = last_segment.n_obs
    emitted_warnings: List[str] = []

    if horizon > max_extrapolation_days:
        msg = (
            f"Forecast horizon ({horizon} days) exceeds "
            f"max_extrapolation_days ({max_extrapolation_days})."
        )
        warnings.warn(msg, UserWarning, stacklevel=4)
        emitted_warnings.append(msg)

    # Edge case: single-observation last segment → constant extrapolation
    if n_j <= 1:
        return np.full(horizon, last_segment.alpha), emitted_warnings

    # -------------------------------------------------------------------
    # H1: Blended linear extrapolation
    # -------------------------------------------------------------------
    # When slope blending is active (alpha > 0 and penultimate segment
    # exists), replace the model-based extrapolation entirely with a
    # linear continuation using the blended slope.  This prevents the
    # Q4-contaminated final segment slope from driving catastrophic
    # overforecasts in BFCM cutoffs.
    if slope_blend_alpha > 0.0 and penultimate_segment is not None:
        slope_final = _slope_at_t1_per_day(last_segment)
        slope_penult = _slope_at_t1_per_day(penultimate_segment)
        blended_slope = slope_blend_alpha * slope_penult + (1.0 - slope_blend_alpha) * slope_final

        # Baseline: value of last segment at t=1 (the segment end)
        baseline = evaluate_segment(last_segment, 1.0)

        # Linear extrapolation: day 1, 2, ..., horizon beyond series end
        trend_array = np.array(
            [baseline + blended_slope * (i + 1) for i in range(horizon)],
            dtype=float,
        )
        return trend_array, emitted_warnings

    # -------------------------------------------------------------------
    # Legacy path: model-based extrapolation with clamping (alpha == 0)
    # -------------------------------------------------------------------
    if last_segment.model_type == "exp" and last_segment.beta > 0 and horizon > n_j:
        msg = (
            "Last segment model is 'exp' with positive beta and horizon "
            f"({horizon}) > segment length ({n_j}). "
            "Exponential extrapolation clamped to linear continuation at t=1."
        )
        warnings.warn(msg, UserWarning, stacklevel=4)
        emitted_warnings.append(msg)

    if last_segment.model_type == "quadratic":
        # t_H = (n_j - 1 + horizon) / (n_j - 1)
        if n_j > 1:
            t_H = (n_j - 1 + horizon) / (n_j - 1)
        else:
            t_H = float(horizon)
        if t_H > 2.0:
            msg = (
                "Last segment model is 'quadratic' and forecast extends "
                f"t={t_H:.2f} > 2 (horizon={horizon} > segment length={n_j}). "
                "Quadratic extrapolation clamped to linear continuation at t=1."
            )
            warnings.warn(msg, UserWarning, stacklevel=4)
            emitted_warnings.append(msg)

    # t_h = ((n_j - 1) + h) / (n_j - 1) for h = 1..horizon
    denom = float(n_j - 1)
    t_values = np.array([(denom + h) / denom for h in range(1, horizon + 1)])

    model = last_segment.model_type

    if model == "quadratic":
        # For t > 1 (extrapolation beyond segment end), switch to linear
        # continuation using the slope at t=1. This prevents the quadratic
        # term from dominating and producing explosive growth (e.g. t=4.92
        # in a 93-day Q4 segment → catastrophic overforecast).
        alpha = last_segment.alpha
        beta = last_segment.beta
        gamma = last_segment.gamma if last_segment.gamma is not None else 0.0
        f_at_end = alpha + beta + gamma          # f(t=1)
        slope_at_end = beta + 2.0 * gamma        # df/dt at t=1
        result = np.where(
            t_values <= 1.0,
            alpha + beta * t_values + gamma * t_values ** 2,
            f_at_end + slope_at_end * (t_values - 1.0),
        )
        return result, emitted_warnings

    if model == "exp":
        # For t > 1 with positive beta, exponential grows without bound.
        # Switch to linear continuation from t=1 using the instantaneous
        # slope there: df/dt = alpha * beta * exp(beta * t).
        alpha = last_segment.alpha
        beta = last_segment.beta
        f_at_end = alpha * np.exp(beta)                  # f(t=1)
        slope_at_end = alpha * beta * np.exp(beta)        # df/dt at t=1
        result = np.where(
            t_values <= 1.0,
            alpha * np.exp(beta * t_values),
            f_at_end + slope_at_end * (t_values - 1.0),
        )
        return result, emitted_warnings

    return np.array([evaluate_segment(last_segment, t) for t in t_values]), emitted_warnings


# ---------------------------------------------------------------------------
# Weekly projection
# ---------------------------------------------------------------------------


def _project_weekly(
    result: DecompositionResult,
    forecast_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Project weekly DOW coefficients onto forecast dates.

    Uses the 60-day recency window (last row).  Falls back to 90-day, then
    365-day, then the full-sample ``result.weekly.coefficients``.

    Parameters
    ----------
    result : DecompositionResult
        V1 decomposition result.
    forecast_dates : pd.DatetimeIndex
        Dates for which to return DOW coefficients.

    Returns
    -------
    np.ndarray
        Shape ``(len(forecast_dates),)`` — raw coefficient (factor or offset)
        for each forecast date.
    """
    recency = result.weekly.recency
    dow_coeffs: Optional[np.ndarray] = None

    for window in [60, 90, 365]:
        if window in recency and recency[window] is not None:
            df = recency[window]
            if len(df) > 0:
                # Columns: date, dow_0, ..., dow_6
                last_row = df.iloc[-1]
                dow_cols = [f"dow_{d}" for d in range(7)]
                # Check all columns exist
                if all(c in last_row.index for c in dow_cols):
                    dow_coeffs = np.array([last_row[c] for c in dow_cols], dtype=float)
                    break

    if dow_coeffs is None:
        # None of the recency windows (60, 90, 365) had valid data — fall back
        # to the full-sample weekly coefficients.  This can happen when the
        # V1 series is shorter than 60 days or when recency tracking was
        # unavailable.  The full-sample coefficients may be stale if DOW
        # effects have drifted recently.
        warnings.warn(
            "No recency window (60, 90, or 365 days) returned valid weekly "
            "coefficients. Falling back to full-sample coefficients. "
            "Weekly forecast may not reflect recent DOW pattern shifts.",
            UserWarning,
            stacklevel=3,
        )
        dow_coeffs = result.weekly.coefficients.copy()

    # Map forecast dates to DOW (Monday=0)
    dow_indices = forecast_dates.dayofweek
    return dow_coeffs[dow_indices]


# ---------------------------------------------------------------------------
# Annual projection
# ---------------------------------------------------------------------------


def _project_annual(
    result: DecompositionResult,
    forecast_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Project Fourier annual component onto forecast dates.

    Uses the same calendar-anchored time index as V1 ``_annual.py``:
    ``t = (date - Jan 1 of result.series.index[0].year).days``.
    The intercept ``a0`` is excluded (absorbed into trend level).

    Parameters
    ----------
    result : DecompositionResult
        V1 decomposition result.
    forecast_dates : pd.DatetimeIndex
        Forecast dates.

    Returns
    -------
    np.ndarray
        Shape ``(len(forecast_dates),)``.
    """
    K = result.annual.n_harmonics
    if K == 0:
        return np.zeros(len(forecast_dates))

    # Calendar anchor: Jan 1 of the year the series starts
    t0_date = pd.Timestamp(result.series.index[0].year, 1, 1)
    t_future = np.array(
        [(d - t0_date).days for d in forecast_dates], dtype=float
    )

    # Coefficients: [a0, a1, b1, a2, b2, ..., aK, bK]
    coef = result.annual.coefficients
    annual = np.zeros(len(forecast_dates))
    for k in range(1, K + 1):
        angle = 2.0 * np.pi * k * t_future / PERIOD
        a_k = coef[2 * k - 1]  # a1 at index 1, a2 at index 3, ...
        b_k = coef[2 * k]      # b1 at index 2, b2 at index 4, ...
        annual += a_k * np.cos(angle) + b_k * np.sin(angle)

    return annual


# ---------------------------------------------------------------------------
# Holiday projection
# ---------------------------------------------------------------------------


def _ols_project_ipm(
    ipm_pairs: List[tuple],
    max_years: int = 4,
    exponential_decay_halflife_years: float = 0.0,
) -> float:
    """Project next IPM from a list of (year, magnitude) pairs using WLS.

    Uses only the most recent ``max_years`` data points (recency weighting by
    truncation).  This prevents over-projection on concave growth series where
    early high-growth years inflate the OLS slope.

    Optionally applies exponential down-weighting of older observations via
    ``exponential_decay_halflife_years``.  Weights are
    ``w_i = 2^(-(distance_from_most_recent / halflife))``, so the most-recent
    observation always has weight 1.0.  When ``halflife == 0`` (default) all
    within-window observations receive equal weight (standard OLS).

    If the projected IPM implies >40% YoY growth from the most recent known
    IPM, the projection is capped at 40% above the most recent value.

    Parameters
    ----------
    ipm_pairs : list of (year_index, magnitude) tuples, sorted by year.
    max_years : int
        Maximum number of most-recent years to use.  Default 4.
    exponential_decay_halflife_years : float
        Exponential decay half-life in years.  0.0 = uniform weights
        (backward-compatible).  Values in (0, 2] provide meaningful
        down-weighting of older points.

    Returns
    -------
    float
        Projected magnitude for the next year (year index = len(ipm_pairs)).
    """
    n = len(ipm_pairs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ipm_pairs[0][1])

    # Apply recency window: use only the last max_years pairs
    years_to_use = min(max_years, n)
    recent_pairs = ipm_pairs[-years_to_use:]

    x_arr = np.array([p[0] for p in recent_pairs], dtype=float)
    y_arr = np.array([p[1] for p in recent_pairs], dtype=float)

    if len(recent_pairs) == 1:
        projected = float(y_arr[0])
    else:
        # Build observation weights
        if exponential_decay_halflife_years > 0.0:
            # Distance from most-recent observation (index 0 = oldest in window)
            distances = x_arr[-1] - x_arr  # 0 at most-recent
            w_arr = np.power(2.0, -distances / exponential_decay_halflife_years)
        else:
            w_arr = np.ones(len(x_arr))

        # Weighted OLS: slope = sum(w*(x-xw)*(y-yw)) / sum(w*(x-xw)^2)
        w_sum = w_arr.sum()
        xw = float(np.dot(w_arr, x_arr) / w_sum)
        yw = float(np.dot(w_arr, y_arr) / w_sum)
        wx_dev = w_arr * (x_arr - xw)
        denom = float(np.dot(wx_dev, x_arr - xw))
        if denom == 0.0:
            slope = 0.0
        else:
            slope = float(np.dot(wx_dev, y_arr - yw) / denom)
        # Project to the next year index (last x + 1)
        projected = float(y_arr[-1]) + slope

    # Growth cap: if projected > most_recent * 1.4, cap it
    most_recent_ipm = float(ipm_pairs[-1][1])
    if most_recent_ipm > 0 and projected > most_recent_ipm * 1.4:
        projected = most_recent_ipm * 1.4

    return projected


def _trend_linked_ipm(
    result: "DecompositionResult",
    projected_ipm_ols: float,
    last_known_ipm: float,
    trend_yoy_blend: float = 0.5,
) -> float:
    """Blend OLS IPM projection with a trend-growth-implied projection.

    The holiday-season lift is economically a multiplier on the underlying
    business level, not an independent quantity.  If the business grew 5% YoY
    in recent months, BFCM magnitude should also grow ~5% — not the (often
    larger) linear extrapolation from historical IPM growth.

    The de-seasonalized YoY growth rate is estimated from the V1 residual
    **excluding all compound block windows**.  Using the raw trailing 90-day
    window would straddle the BFCM period for series ending in Q4, making the
    YoY ratio ≈ 1.0 (holiday vs. holiday) and defeating the purpose of the
    blending.  By excluding compound-block dates and using a wider 180-day
    non-compound window, the ratio accurately captures structural business
    growth rather than holiday seasonality noise.

    Parameters
    ----------
    result : DecompositionResult
        V1 decomposition result (provides residual series and holiday effects).
    projected_ipm_ols : float
        OLS-based IPM projection (from ``_ols_project_ipm``).
    last_known_ipm : float
        Most-recent reliable IPM value.
    trend_yoy_blend : float
        Weight on the trend-implied projection (0 = pure OLS,
        1 = pure trend-implied).  Default 0.5.

    Returns
    -------
    float
        Blended IPM projection (>= 0).
    """
    if trend_yoy_blend <= 0.0:
        return projected_ipm_ols

    residual = result.residual

    # Build compound-block exclusion set: all dates inside any compound
    # holiday ramp window across all years.  These dates contain the
    # holiday lift in the residual and would bias the YoY ratio if included.
    compound_dates: set = set()
    for eff in result.holidays:
        if eff.compound:
            ramp_start = pd.Timestamp(eff.ramp_start).date()
            ramp_end = pd.Timestamp(eff.ramp_end).date()
            d = ramp_start
            while d <= ramp_end:
                compound_dates.add(d)
                d += _dt.timedelta(days=1)

    # Filter residual to non-compound dates only
    non_compound_residual = residual[
        ~residual.index.map(lambda ts: ts.date() in compound_dates)
    ]

    if len(non_compound_residual) < 90:
        return projected_ipm_ols  # not enough non-compound data

    n_nc = len(non_compound_residual)

    # Recent 180-day (non-compound) window and same window one year prior.
    # 180-day window is used (vs. the former 90-day) because excluding
    # compound-block dates reduces the effective sample; 180 days gives
    # ~130-150 non-compound observations, safely above the 30-obs threshold.
    window_size = 180
    recent_nc = non_compound_residual.iloc[max(0, n_nc - window_size):]

    # One-year-prior window: same calendar span shifted back ~365 days.
    # We identify the prior window by date range rather than positional
    # offset, because compound-date exclusion makes the positional mapping
    # non-trivial.
    if len(recent_nc) == 0:
        return projected_ipm_ols

    recent_start = recent_nc.index[0]
    prior_end = recent_start - pd.Timedelta(days=365 - window_size)
    prior_start = prior_end - pd.Timedelta(days=window_size)

    prior_nc = non_compound_residual[
        (non_compound_residual.index >= prior_start)
        & (non_compound_residual.index <= prior_end)
    ]

    if len(recent_nc) < 30 or len(prior_nc) < 30:
        return projected_ipm_ols  # not enough data for trend estimate

    recent_mean = recent_nc.mean()
    prior_mean = prior_nc.mean()

    if prior_mean <= 0:
        return projected_ipm_ols

    # De-seasonalized YoY growth from recent (non-compound) trend
    trend_yoy_ratio = recent_mean / prior_mean
    trend_implied_ipm = last_known_ipm * trend_yoy_ratio

    # Blend: trend_yoy_blend on trend-implied, (1-blend) on OLS
    blended = trend_yoy_blend * trend_implied_ipm + (1.0 - trend_yoy_blend) * projected_ipm_ols
    return max(blended, 0.0)


def _compute_ipm_projection(
    h_name: str,
    matching,
    result: Optional["DecompositionResult"] = None,
    trend_yoy_blend: float = 0.0,
    ipm_max_years: int = 4,
    ipm_decay_halflife: float = 0.0,
) -> float:
    """Compute projected IPM for a compound holiday from its historical occurrences.

    Parameters
    ----------
    h_name : str
        Holiday name (for warnings).
    matching : list of HolidayEffect
        All HolidayEffect objects for this holiday name.
    result : DecompositionResult or None
        V1 decomposition result.  Required when ``trend_yoy_blend > 0``
        to compute the trend-linked IPM projection.  Ignored when
        ``trend_yoy_blend == 0``.
    trend_yoy_blend : float
        Weight on the trend-growth-implied IPM projection (see
        ``_trend_linked_ipm``).  Default 0.0 (pure OLS, backward-compatible).
    ipm_max_years : int
        Maximum number of most-recent historical years to include in the
        OLS fit.  Passed directly to ``_ols_project_ipm``.  Default 4.
    ipm_decay_halflife : float
        Exponential decay half-life in years for down-weighting older IPM
        observations.  0.0 = uniform weights.  Passed to
        ``_ols_project_ipm``.  Default 0.0.

    Returns
    -------
    float
        Projected IPM for the next occurrence.
    """
    sorted_occ = sorted(matching, key=lambda he: he.date)
    ipm_pairs = []
    for idx, occ in enumerate(sorted_occ):
        if (
            occ.individual_peak_magnitude is not None
            and occ.individual_peak_magnitude_reliable
        ):
            ipm_pairs.append((idx, occ.individual_peak_magnitude))

    n_reliable = len(ipm_pairs)
    if n_reliable >= 2:
        projected_ipm = _ols_project_ipm(
            ipm_pairs,
            max_years=ipm_max_years,
            exponential_decay_halflife_years=ipm_decay_halflife,
        )
    elif n_reliable == 1:
        projected_ipm = float(ipm_pairs[0][1])
    else:
        all_ipms = [
            occ.individual_peak_magnitude
            for occ in sorted_occ
            if occ.individual_peak_magnitude is not None
        ]
        if all_ipms:
            projected_ipm = float(np.mean(all_ipms))
            warnings.warn(
                f"Holiday '{h_name}': no reliable IPM values available. "
                "Using mean of all occurrences as projection.",
                UserWarning,
                stacklevel=5,
            )
        else:
            most_recent = sorted_occ[-1]
            projected_ipm = most_recent.magnitude
            warnings.warn(
                f"Holiday '{h_name}': no IPM values available at all. "
                "Falling back to block magnitude.",
                UserWarning,
                stacklevel=5,
            )

    # Guard: if all historical IPMs were positive and projection < 0,
    # clamp to half the minimum observed IPM.
    positive_ipms = [v for _, v in ipm_pairs if v >= 0] if ipm_pairs else []
    if positive_ipms and projected_ipm < 0:
        projected_ipm = min(positive_ipms) / 2.0

    # Trend-linked blending: dampen OLS over-extrapolation with de-seasonalized
    # YoY business growth rate from the V1 residual.
    if trend_yoy_blend > 0.0 and result is not None and ipm_pairs:
        last_known_ipm = float(ipm_pairs[-1][1])
        projected_ipm = _trend_linked_ipm(
            result=result,
            projected_ipm_ols=projected_ipm,
            last_known_ipm=last_known_ipm,
            trend_yoy_blend=trend_yoy_blend,
        )

    return projected_ipm


def _triangular_ramp_contribution(
    h_date: "_dt.date",
    projected_ipm: float,
    forecast_date_set: Dict["_dt.date", int],
    min_fc_date: "_dt.date",
    max_fc_date: "_dt.date",
    ramp_half_width: int = 3,
) -> Dict["_dt.date", float]:
    """Compute triangular ramp contributions for a single holiday occurrence.

    Returns a dict mapping forecast date → contribution value for all days
    within ``±ramp_half_width`` days of h_date that fall in the forecast
    window.  The triangular weight is:

        weight = max(0, 1 - |day_delta| / (ramp_half_width + 1))

    so that the peak weight at day_delta=0 is 1.0 and the weight at
    ±ramp_half_width is 1 / (ramp_half_width + 1) > 0.

    Parameters
    ----------
    h_date : datetime.date
        The holiday date.
    projected_ipm : float
        Projected individual peak magnitude for this occurrence.
    forecast_date_set : dict
        Mapping from date → forecast array index.
    min_fc_date, max_fc_date : datetime.date
        Forecast window boundaries (inclusive).
    ramp_half_width : int
        Half-width of the triangular ramp in days.  Default 3.
    """
    contribs: Dict["_dt.date", float] = {}
    divisor = float(ramp_half_width + 1)
    for day_delta in range(-ramp_half_width, ramp_half_width + 1):
        eff_date = h_date + _dt.timedelta(days=day_delta)
        if eff_date < min_fc_date or eff_date > max_fc_date:
            continue
        if eff_date not in forecast_date_set:
            continue
        weight = max(0.0, 1.0 - abs(day_delta) / divisor)
        contribs[eff_date] = projected_ipm * weight
    return contribs


def _group_future_holidays_by_proximity(
    future_holidays: Dict[str, List[pd.Timestamp]],
    max_holiday_merge_gap_days: int,
) -> List[List[tuple]]:
    """Group future (holiday_name, holiday_date) pairs into compound groups.

    Two holidays belong in the same group if any member of the group is within
    ``max_holiday_merge_gap_days`` days of the candidate.

    Holidays NOT in the compound projection path (i.e., non-compound per
    result.holidays) are excluded before calling this function — this function
    works purely on (name, date) pairs.

    Parameters
    ----------
    future_holidays : dict
        Mapping from holiday name to list of future dates.
    max_holiday_merge_gap_days : int
        Maximum gap in days between holiday dates for them to be grouped.

    Returns
    -------
    list of groups
        Each group is a list of (h_name, h_date as _dt.date) tuples.
    """
    # Flatten to (name, date) pairs, sorted by date
    all_pairs: List[tuple] = []
    for h_name, dates in future_holidays.items():
        for d in dates:
            all_pairs.append((h_name, pd.Timestamp(d).date()))
    all_pairs.sort(key=lambda x: x[1])

    if not all_pairs:
        return []

    groups: List[List[tuple]] = []
    current_group: List[tuple] = [all_pairs[0]]

    for pair in all_pairs[1:]:
        _, d_new = pair
        # Check if d_new is within gap of any member in current group
        in_group = any(
            abs((d_new - d_existing).days) <= max_holiday_merge_gap_days
            for _, d_existing in current_group
        )
        if in_group:
            current_group.append(pair)
        else:
            groups.append(current_group)
            current_group = [pair]

    groups.append(current_group)
    return groups


def _project_holidays(
    result: DecompositionResult,
    forecast_dates: pd.DatetimeIndex,
    future_holidays: Dict[str, List[pd.Timestamp]],
    max_holiday_merge_gap_days: int = 35,
    trend_yoy_blend: float = 0.0,
    ramp_half_width: int = 3,
    ipm_max_years: int = 4,
    ipm_decay_halflife: float = 0.0,
) -> np.ndarray:
    """Project holiday effects onto forecast dates.

    For **compound block members** the projection uses
    ``individual_peak_magnitude`` (IPM) and a triangular ramp of
    ``±ramp_half_width`` days centred on the holiday date.

    **Compound ramp max-pooling** (Fix A): when multiple future holidays fall
    within ``max_holiday_merge_gap_days`` days of each other, their triangular
    ramp contributions are max-pooled rather than summed.  This prevents
    double-counting when Thanksgiving (Nov 27), Black Friday (Nov 28), and
    Cyber Monday (Dec 1) each add a ramp to the same forecast days.

    For **non-compound holidays** the legacy shape-scaling path is used
    (project ``year_magnitudes`` via OLS, normalise the reference
    ``effect_series``, scale by projected magnitude).

    Parameters
    ----------
    result : DecompositionResult
        V1 decomposition result.
    forecast_dates : pd.DatetimeIndex
        Forecast dates.
    future_holidays : dict
        Mapping from holiday name (str) to list of future occurrence dates.
    max_holiday_merge_gap_days : int
        Maximum date gap in days for grouping future compound holidays into
        a single max-pooled group.  Default 35.
    trend_yoy_blend : float
        Weight on trend-growth-implied IPM projection (see
        ``_trend_linked_ipm``).  Default 0.0 (pure OLS, backward-compatible).
    ramp_half_width : int
        Half-width of the triangular ramp used for compound holiday
        projection (days).  Default 3.
    ipm_max_years : int
        Maximum number of most-recent historical years used for OLS IPM
        projection.  Default 4.  Passed to ``_compute_ipm_projection``.
    ipm_decay_halflife : float
        Exponential decay half-life in years for down-weighting older IPM
        observations in the OLS fit.  0.0 = uniform weights (default).

    Returns
    -------
    np.ndarray
        Shape ``(len(forecast_dates),)``, combined holiday effects.
    """
    holiday_array = np.zeros(len(forecast_dates))
    forecast_date_set = {d.date(): i for i, d in enumerate(forecast_dates)}
    min_fc_date = forecast_dates[0].date()
    max_fc_date = forecast_dates[-1].date()

    # Separate future holidays into compound and non-compound sets
    compound_future: Dict[str, List[pd.Timestamp]] = {}
    noncompound_future: Dict[str, List[pd.Timestamp]] = {}

    for h_name, future_dates in future_holidays.items():
        matching = [he for he in result.holidays if he.name == h_name]

        if not matching:
            warnings.warn(
                f"Holiday '{h_name}' not found in result.holidays. "
                "Holiday component will be zero.",
                UserWarning,
                stacklevel=4,
            )
            continue

        is_compound = any(he.compound for he in matching)
        if is_compound:
            compound_future[h_name] = future_dates
        else:
            noncompound_future[h_name] = future_dates

    # ------------------------------------------------------------------
    # Compound holidays: max-pooling within proximity groups
    # ------------------------------------------------------------------
    if compound_future:
        # Pre-compute projected IPM for each compound holiday
        projected_ipms: Dict[str, float] = {}
        for h_name in compound_future:
            matching = [he for he in result.holidays if he.name == h_name]
            projected_ipms[h_name] = _compute_ipm_projection(
                h_name, matching, result=result, trend_yoy_blend=trend_yoy_blend,
                ipm_max_years=ipm_max_years, ipm_decay_halflife=ipm_decay_halflife,
            )

        # Group future holidays by date proximity
        groups = _group_future_holidays_by_proximity(
            compound_future, max_holiday_merge_gap_days
        )

        for group in groups:
            if len(group) == 1:
                # Single holiday in group — add directly (no overlap issue)
                h_name, h_date = group[0]
                ipm = projected_ipms[h_name]
                for eff_date, contrib in _triangular_ramp_contribution(
                    h_date, ipm, forecast_date_set, min_fc_date, max_fc_date,
                    ramp_half_width=ramp_half_width,
                ).items():
                    holiday_array[forecast_date_set[eff_date]] += contrib
            else:
                # Multiple holidays in group — collect all ramp contributions
                # per forecast date, then take MAX within the group
                # Build: date → list of contributions from all group members
                date_contributions: Dict["_dt.date", List[float]] = {}
                for h_name, h_date in group:
                    ipm = projected_ipms[h_name]
                    for eff_date, contrib in _triangular_ramp_contribution(
                        h_date, ipm, forecast_date_set, min_fc_date, max_fc_date,
                        ramp_half_width=ramp_half_width,
                    ).items():
                        if eff_date not in date_contributions:
                            date_contributions[eff_date] = []
                        date_contributions[eff_date].append(contrib)

                # Max-pool within group, then add to holiday_array
                for eff_date, contribs in date_contributions.items():
                    holiday_array[forecast_date_set[eff_date]] += max(contribs)

    # ------------------------------------------------------------------
    # Non-compound holidays: legacy shape-scaling path
    # ------------------------------------------------------------------
    for h_name, future_dates in noncompound_future.items():
        matching = [he for he in result.holidays if he.name == h_name]

        year_mags = matching[0].year_magnitudes

        # Project next magnitude via OLS
        Y = len(year_mags)
        if Y == 0:
            most_recent_he = max(matching, key=lambda he: he.date)
            projected_mag = most_recent_he.magnitude
        elif Y == 1:
            projected_mag = float(year_mags[0])
        else:
            y_arr = np.array(year_mags, dtype=float)
            x_arr = np.arange(Y, dtype=float)
            xm = x_arr.mean()
            ym = y_arr.mean()
            denom = np.sum((x_arr - xm) ** 2)
            if denom == 0.0:
                slope = 0.0
            else:
                slope = float(np.sum((x_arr - xm) * (y_arr - ym)) / denom)
            projected_mag = float(y_arr[-1]) + slope

        # Negative projection guard
        if Y > 0 and all(m >= 0 for m in year_mags) and projected_mag < 0:
            projected_mag = min(float(m) for m in year_mags) / 2.0

        # Find best reference ramp shape
        sorted_occ = sorted(matching, key=lambda he: he.date, reverse=True)

        ref_he = None
        for occ in sorted_occ:
            if not occ.ramp_start_ceiling_hit and occ.individual_peak_magnitude_reliable:
                ref_he = occ
                break

        if ref_he is None:
            ref_he = sorted_occ[0]
            warnings.warn(
                f"Holiday '{h_name}': all historical occurrences have unreliable "
                "ramp shapes (ceiling hit or boundary truncation). "
                "Using most recent; shape may be truncated.",
                UserWarning,
                stacklevel=4,
            )

        # Restrict effect series to [ramp_start, ramp_end]
        ref_effect = ref_he.effect_series.copy()
        ramp_start_ts = pd.Timestamp(ref_he.ramp_start)
        ramp_end_ts = pd.Timestamp(ref_he.ramp_end)
        ref_effect = ref_effect.loc[ramp_start_ts:ramp_end_ts]

        ref_mag = ref_he.magnitude
        if abs(ref_mag) < 1e-10:
            continue

        unit_shape = ref_effect / ref_mag
        ref_h_date = pd.Timestamp(ref_he.date)
        offset_days = (ref_h_date - ramp_start_ts).days

        for future_date in future_dates:
            future_ts = pd.Timestamp(future_date)
            future_ramp_start = future_ts - pd.Timedelta(days=offset_days)
            projected_effect = unit_shape * projected_mag

            for eff_date_ts, eff_val in projected_effect.items():
                day_offset = (eff_date_ts - ramp_start_ts).days
                future_eff_date = (
                    future_ramp_start + pd.Timedelta(days=day_offset)
                ).date()

                if future_eff_date < min_fc_date or future_eff_date > max_fc_date:
                    continue

                if future_eff_date in forecast_date_set:
                    fc_idx = forecast_date_set[future_eff_date]
                    holiday_array[fc_idx] += eff_val

    return holiday_array


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def _combine_components(
    trend: np.ndarray,
    weekly: np.ndarray,
    annual: np.ndarray,
    holiday: np.ndarray,
    is_multiplicative: bool,
) -> np.ndarray:
    """Combine forecast components into total forecast.

    CRITICAL: In multiplicative mode the weekly factor multiplies the SUM of
    trend, annual, and holiday — not each term independently.  Holiday and
    annual effects were estimated on the weekly-adjusted series (y / w_d) and
    are additive offsets to that adjusted level.  Multiplying each component
    by w_d independently would incorrectly double-apply the DOW scaling.

    Parameters
    ----------
    trend : np.ndarray
        Piecewise trend projection.
    weekly : np.ndarray
        DOW coefficients (unit-mean factors in multiplicative mode; zero-sum
        offsets in additive mode).
    annual : np.ndarray
        Fourier annual projection.
    holiday : np.ndarray
        Holiday effect projection.
    is_multiplicative : bool
        If ``True``, use multiplicative combination; otherwise additive.

    Returns
    -------
    np.ndarray
        Total forecast.
    """
    if is_multiplicative:
        # ŷ(t) = w_d × (trend + annual + holiday)
        return weekly * (trend + annual + holiday)
    else:
        # ŷ(t) = trend + w_d + annual + holiday
        return trend + weekly + annual + holiday
