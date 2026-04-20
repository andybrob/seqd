"""Stage 6: Out-of-sample forecasting from fitted V1 decomposition."""

from __future__ import annotations

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


def _project_holidays(
    result: DecompositionResult,
    forecast_dates: pd.DatetimeIndex,
    future_holidays: Dict[str, List[pd.Timestamp]],
) -> np.ndarray:
    """Project holiday effects onto forecast dates.

    For each future holiday occurrence provided in ``future_holidays``:

    1. Retrieve historical ``year_magnitudes`` from ``result.holidays``.
    2. Project next magnitude via OLS (linear extrapolation).
    3. Use the most recent reliable ramp shape as template (falls back to
       most recent if all shapes are unreliable).
    4. Scale template by projected magnitude and map to forecast dates.

    Overlapping ramp windows are summed.

    Parameters
    ----------
    result : DecompositionResult
        V1 decomposition result.
    forecast_dates : pd.DatetimeIndex
        Forecast dates.
    future_holidays : dict
        Mapping from holiday name (str) to list of future occurrence dates.

    Returns
    -------
    np.ndarray
        Shape ``(len(forecast_dates),)``, summed holiday effects.
    """
    holiday_array = np.zeros(len(forecast_dates))
    forecast_date_set = {d.date(): i for i, d in enumerate(forecast_dates)}
    min_fc_date = forecast_dates[0].date()
    max_fc_date = forecast_dates[-1].date()

    for h_name, future_dates in future_holidays.items():
        # Find all HolidayEffect objects matching this holiday name
        matching = [he for he in result.holidays if he.name == h_name]

        if not matching:
            warnings.warn(
                f"Holiday '{h_name}' not found in result.holidays. "
                "Holiday component will be zero.",
                UserWarning,
                stacklevel=4,
            )
            continue

        # year_magnitudes is the same across all occurrences for the same name
        # (stored per-occurrence but represents the full historical series)
        # Use the first matching occurrence's year_magnitudes
        year_mags = matching[0].year_magnitudes

        # --- Step 2: Project next magnitude ---
        Y = len(year_mags)
        if Y == 0:
            # Fall back to the magnitude attribute of the most recent occurrence
            most_recent_he = max(matching, key=lambda he: he.date)
            projected_mag = most_recent_he.magnitude
        elif Y == 1:
            projected_mag = float(year_mags[0])
        else:
            # OLS slope on year indices 0..Y-1 (works for Y=2 and Y>=3)
            y_arr = np.array(year_mags, dtype=float)
            x_arr = np.arange(Y, dtype=float)
            n_y = Y
            xm = x_arr.mean()
            ym = y_arr.mean()
            denom = np.sum((x_arr - xm) ** 2)
            if denom == 0.0:
                slope = 0.0
            else:
                slope = float(np.sum((x_arr - xm) * (y_arr - ym)) / denom)
            projected_mag = float(y_arr[-1]) + slope

        # Negative projection guard: if all historical positive and projected < 0
        if Y > 0 and all(m >= 0 for m in year_mags) and projected_mag < 0:
            projected_mag = min(float(m) for m in year_mags) / 2.0

        # --- Step 3: Find best reference ramp shape ---
        # Sort matching occurrences by date descending (most recent first)
        sorted_occ = sorted(matching, key=lambda he: he.date, reverse=True)

        ref_he = None
        for occ in sorted_occ:
            if not occ.ramp_start_ceiling_hit and occ.individual_peak_magnitude_reliable:
                ref_he = occ
                break

        if ref_he is None:
            # All occurrences unreliable: use most recent with warning
            ref_he = sorted_occ[0]
            warnings.warn(
                f"Holiday '{h_name}': all historical occurrences have unreliable "
                "ramp shapes (ceiling hit or boundary truncation). "
                "Using most recent; shape may be truncated.",
                UserWarning,
                stacklevel=4,
            )

        # Handle compound block: if ref_he is non-primary (effect_series is zero),
        # trace to primary member (earliest date in the same compound block)
        if ref_he.compound and ref_he.compound_block_id is not None:
            block_members = [
                he for he in result.holidays
                if he.compound_block_id == ref_he.compound_block_id
            ]
            if block_members:
                primary = min(block_members, key=lambda he: he.date)
                # Check if ref_he effect_series is all zero (non-primary)
                if np.allclose(ref_he.effect_series.values, 0.0):
                    ref_he = primary

        # Restrict effect series to [ramp_start, ramp_end]
        ref_effect = ref_he.effect_series.copy()
        ramp_start_ts = pd.Timestamp(ref_he.ramp_start)
        ramp_end_ts = pd.Timestamp(ref_he.ramp_end)
        ref_effect = ref_effect.loc[ramp_start_ts:ramp_end_ts]

        # Magnitude of reference occurrence for normalisation
        ref_mag = ref_he.magnitude
        if abs(ref_mag) < 1e-10:
            # Skip: cannot normalise
            continue

        # Normalise to unit magnitude
        unit_shape = ref_effect / ref_mag

        # Reference holiday date (calendar date of the holiday itself)
        ref_h_date = pd.Timestamp(ref_he.date)

        # Offset from ramp_start to the holiday date
        offset_days = (ref_h_date - ramp_start_ts).days

        # --- Step 4: Map to each future occurrence ---
        for future_date in future_dates:
            future_ts = pd.Timestamp(future_date)

            # Future ramp start: future holiday date minus offset
            future_ramp_start = future_ts - pd.Timedelta(days=offset_days)

            # Build projected shape with scaled magnitude
            projected_effect = unit_shape * projected_mag

            for rel_idx, (eff_date, eff_val) in enumerate(projected_effect.items()):
                # Map relative position to future ramp date
                day_offset = (eff_date - ramp_start_ts).days
                future_eff_date = (future_ramp_start + pd.Timedelta(days=day_offset)).date()

                if future_eff_date < min_fc_date or future_eff_date > max_fc_date:
                    continue  # outside horizon — clip silently

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
