"""Stage 5: Piecewise trend fitting on changepoint segments."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._structures import SegmentTrend
from ._utils import ols_fit


# ---------------------------------------------------------------------------
# Internal fitting helpers
# ---------------------------------------------------------------------------


def _fit_linear(
    r_segment: np.ndarray,
    t_norm: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """Fit linear model f(t) = α + β·t via OLS.

    Parameters
    ----------
    r_segment : np.ndarray
        Observed values in segment.
    t_norm : np.ndarray
        Normalised time [0, 1].

    Returns
    -------
    (alpha, beta, None, rss)
    """
    X = np.column_stack([np.ones_like(t_norm), t_norm])
    coef = ols_fit(X, r_segment)
    if not np.all(np.isfinite(coef)):
        return None, None, None, np.inf
    fitted = coef[0] + coef[1] * t_norm
    rss = float(np.sum((r_segment - fitted) ** 2))
    return float(coef[0]), float(coef[1]), None, rss


def _fit_log(
    r_segment: np.ndarray,
    t_norm: np.ndarray,
    T_days: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """Fit logarithmic model f(t) = α + β·ln(1 + t·T_days) via OLS.

    Parameters
    ----------
    r_segment : np.ndarray
        Observed values.
    t_norm : np.ndarray
        Normalised time [0, 1].
    T_days : int
        Segment length in calendar days (used to anchor log curvature).

    Returns
    -------
    (alpha, beta, None, rss)
    """
    log_arg = np.log1p(t_norm * T_days)  # ln(1 + t * T_days)
    X = np.column_stack([np.ones_like(t_norm), log_arg])
    coef = ols_fit(X, r_segment)
    if not np.all(np.isfinite(coef)):
        return None, None, None, np.inf
    fitted = coef[0] + coef[1] * log_arg
    rss = float(np.sum((r_segment - fitted) ** 2))
    return float(coef[0]), float(coef[1]), None, rss


def _fit_exp(
    r_segment: np.ndarray,
    t_norm: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """Fit exponential model f(t) = α·exp(β·t) via log-linearisation.

    Only applicable when all values are strictly positive.

    Parameters
    ----------
    r_segment : np.ndarray
        Observed values.  Must all be > 0 to call this function.
    t_norm : np.ndarray
        Normalised time [0, 1].

    Returns
    -------
    (alpha, beta, None, rss) or (None, None, None, inf) on failure.
    """
    if np.any(r_segment <= 0.0):
        return None, None, None, np.inf

    z = np.log(r_segment)
    if not np.all(np.isfinite(z)):
        return None, None, None, np.inf

    X = np.column_stack([np.ones_like(t_norm), t_norm])
    coef = ols_fit(X, z)
    if not np.all(np.isfinite(coef)):
        return None, None, None, np.inf

    alpha = float(np.exp(coef[0]))
    beta = float(coef[1])
    fitted = alpha * np.exp(beta * t_norm)
    rss = float(np.sum((r_segment - fitted) ** 2))
    return alpha, beta, None, rss


def _fit_quadratic(
    r_segment: np.ndarray,
    t_norm: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """Fit quadratic model f(t) = α + β·t + γ·t² via OLS.

    Parameters
    ----------
    r_segment : np.ndarray
        Observed values.
    t_norm : np.ndarray
        Normalised time [0, 1].

    Returns
    -------
    (alpha, beta, gamma, rss)
    """
    X = np.column_stack([np.ones_like(t_norm), t_norm, t_norm ** 2])
    coef = ols_fit(X, r_segment)
    if not np.all(np.isfinite(coef)):
        return None, None, None, np.inf
    fitted = coef[0] + coef[1] * t_norm + coef[2] * t_norm ** 2
    rss = float(np.sum((r_segment - fitted) ** 2))
    return float(coef[0]), float(coef[1]), float(coef[2]), rss


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_aic(rss: float, n: int, k: int) -> float:
    """Compute AIC = n·ln(RSS/n) + 2k.

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of observations.
    k : int
        Number of parameters.

    Returns
    -------
    float
        AIC value, or ``-np.inf`` when ``rss <= 0``, or ``np.inf`` when the
        model is fully saturated (``n <= k``).
    """
    if n <= k:
        return np.inf
    if rss <= 0.0:
        return -np.inf
    return n * np.log(rss / n) + 2.0 * k


def evaluate_segment(segment: SegmentTrend, t: float) -> float:
    """Evaluate the fitted trend function at normalised time ``t``.

    Valid for any ``t``, including ``t > 1`` (extrapolation beyond segment
    end) and ``t < 0`` (rarely needed).

    Parameters
    ----------
    segment : SegmentTrend
        Fitted segment record.
    t : float
        Normalised time coordinate.  ``t = 0`` corresponds to
        ``segment.start_date``, ``t = 1`` to ``segment.end_date``.

    Returns
    -------
    float
        Trend value at ``t``.
    """
    model = segment.model_type
    alpha = segment.alpha
    beta = segment.beta

    if model == "constant":
        return alpha
    if model == "linear":
        return alpha + beta * t
    if model == "log":
        log_arg = np.log1p(t * segment.T_days)
        return alpha + beta * log_arg
    if model == "exp":
        return alpha * np.exp(beta * t)
    if model == "quadratic":
        gamma = segment.gamma if segment.gamma is not None else 0.0
        return alpha + beta * t + gamma * t ** 2
    raise ValueError(f"Unknown model_type: '{model}'")


# ---------------------------------------------------------------------------
# Segment fitter
# ---------------------------------------------------------------------------


def fit_segment(
    r_segment: np.ndarray,
    model_type: str,
    t_normalized: np.ndarray,
    T_days: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """Fit a single trend model to a segment.

    Parameters
    ----------
    r_segment : np.ndarray
        Observed values in the segment.
    model_type : str
        One of ``"linear"``, ``"log"``, ``"exp"``, ``"quadratic"``.
    t_normalized : np.ndarray
        Normalised time array, same length as ``r_segment``.
    T_days : int
        Segment length in calendar days (used in log model).

    Returns
    -------
    (alpha, beta, gamma, rss)
        ``(None, None, None, np.inf)`` when model is inapplicable or OLS
        fails.
    """
    n = len(r_segment)
    if model_type == "linear":
        return _fit_linear(r_segment, t_normalized)
    if model_type == "log":
        return _fit_log(r_segment, t_normalized, T_days)
    if model_type == "exp":
        if np.any(r_segment <= 0.0):
            return None, None, None, np.inf
        return _fit_exp(r_segment, t_normalized)
    if model_type == "quadratic":
        if n < 5:
            return None, None, None, np.inf
        return _fit_quadratic(r_segment, t_normalized)
    raise ValueError(f"Unknown model_type: '{model_type}'")


# ---------------------------------------------------------------------------
# Piecewise trend fitting
# ---------------------------------------------------------------------------


def fit_piecewise_trend(
    residual: pd.Series,
    changepoint_dates: List[pd.Timestamp],
    aic_linear_delta: float = 2.0,
) -> List[SegmentTrend]:
    """Fit piecewise parametric trends on changepoint-defined segments.

    For each segment, candidate models (linear, log, exp, quadratic) are fit
    and the best is selected by AIC with a linear parsimony bonus.

    Parameters
    ----------
    residual : pd.Series
        V1 residual series with daily ``DatetimeIndex``.
    changepoint_dates : list of pd.Timestamp
        Changepoint dates from Stage 4 (each is the first date of a new
        segment).  May be empty (single-segment case).
    aic_linear_delta : float
        AIC threshold δ.  Linear is preferred over a better-fitting model
        when ``AIC_linear - AIC_best < δ``.  Default 2.0.

    Returns
    -------
    list of SegmentTrend
        One record per segment, ordered by ``segment_index`` (1-based).
    """
    if aic_linear_delta < 0:
        raise ValueError(
            f"aic_linear_delta must be >= 0 (got {aic_linear_delta})."
        )

    r_full = residual.dropna().copy().astype(float)
    idx = r_full.index
    n = len(r_full)

    # Build segment boundary indices (0-based)
    # changepoint_dates are the first date of each new segment
    cp_indices: List[int] = []
    for cp_date in sorted(changepoint_dates):
        pos = idx.get_loc(cp_date)
        cp_indices.append(int(pos))

    # Boundaries: [0, cp1, cp2, ..., n]
    boundaries = [0] + cp_indices + [n]
    segments: List[SegmentTrend] = []

    for seg_num, (start_b, end_b) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), start=1
    ):
        seg_slice = slice(start_b, end_b)
        r_seg = r_full.values[seg_slice]
        seg_idx = idx[seg_slice]
        n_j = len(r_seg)

        start_date = seg_idx[0]
        end_date = seg_idx[-1]
        T_days = int(n_j)  # daily series: T_days == n_obs

        # Handle single-observation segment: constant model
        if n_j == 1:
            segments.append(
                SegmentTrend(
                    segment_index=seg_num,
                    start_date=start_date,
                    end_date=end_date,
                    n_obs=n_j,
                    model_type="constant",
                    alpha=float(r_seg[0]),
                    beta=0.0,
                    gamma=None,
                    T_days=T_days,
                    aic=np.nan,
                    aic_linear=np.nan,
                    rss=0.0,
                    selected_reason="only candidate",
                    t_anchor_date=start_date,
                )
            )
            continue

        # Normalised time: t_i = (i-1) / (n_j - 1) for i=1..n_j
        t_norm = np.arange(n_j, dtype=float) / (n_j - 1)

        # All-constant segment: RSS_L == 0 → fit constant
        all_same = np.allclose(r_seg, r_seg[0])
        if all_same:
            segments.append(
                SegmentTrend(
                    segment_index=seg_num,
                    start_date=start_date,
                    end_date=end_date,
                    n_obs=n_j,
                    model_type="constant",
                    alpha=float(r_seg[0]),
                    beta=0.0,
                    gamma=None,
                    T_days=T_days,
                    aic=np.nan,
                    aic_linear=np.nan,
                    rss=0.0,
                    selected_reason="only candidate",
                    t_anchor_date=start_date,
                )
            )
            continue

        # Fit all applicable candidate models
        candidates: dict = {}

        # Linear (always applicable)
        a_l, b_l, _, rss_l = _fit_linear(r_seg, t_norm)
        if a_l is not None:
            candidates["linear"] = (a_l, b_l, None, rss_l, compute_aic(rss_l, n_j, 2))

        # Log (always applicable)
        a_g, b_g, _, rss_g = _fit_log(r_seg, t_norm, T_days)
        if a_g is not None:
            candidates["log"] = (a_g, b_g, None, rss_g, compute_aic(rss_g, n_j, 2))

        # Exponential: only when all values strictly positive
        if np.all(r_seg > 0.0):
            a_e, b_e, _, rss_e = _fit_exp(r_seg, t_norm)
            if a_e is not None:
                candidates["exp"] = (a_e, b_e, None, rss_e, compute_aic(rss_e, n_j, 2))

        # Quadratic: only when n_j >= 5
        if n_j >= 5:
            a_q, b_q, g_q, rss_q = _fit_quadratic(r_seg, t_norm)
            if a_q is not None:
                candidates["quadratic"] = (
                    a_q, b_q, g_q, rss_q, compute_aic(rss_q, n_j, 3)
                )

        if not candidates:
            # Fallback: constant (should not normally happen)
            segments.append(
                SegmentTrend(
                    segment_index=seg_num,
                    start_date=start_date,
                    end_date=end_date,
                    n_obs=n_j,
                    model_type="constant",
                    alpha=float(np.mean(r_seg)),
                    beta=0.0,
                    gamma=None,
                    T_days=T_days,
                    aic=np.nan,
                    aic_linear=np.nan,
                    rss=float(np.sum((r_seg - np.mean(r_seg)) ** 2)),
                    selected_reason="only candidate",
                    t_anchor_date=start_date,
                )
            )
            continue

        # AIC of linear fit (always computed for reference)
        aic_linear_val = candidates["linear"][4] if "linear" in candidates else np.inf

        # Find best model by AIC
        best_model = min(candidates, key=lambda m: candidates[m][4])
        best_aic = candidates[best_model][4]

        # Single candidate: record reason and skip parsimony comparison
        if len(candidates) == 1:
            selected = best_model
            reason = "only candidate"
        else:
            # Linear parsimony rule
            delta = aic_linear_val - best_aic
            if best_model == "linear":
                selected = "linear"
                reason = "linear only"
            elif delta < aic_linear_delta:
                # AIC difference small: prefer linear
                selected = "linear"
                reason = f"linear preference, ΔAIC={delta:.2f}"
            else:
                selected = best_model
                reason = "lowest AIC"

        alpha_s, beta_s, gamma_s, rss_s, aic_s = candidates[selected]

        segments.append(
            SegmentTrend(
                segment_index=seg_num,
                start_date=start_date,
                end_date=end_date,
                n_obs=n_j,
                model_type=selected,
                alpha=float(alpha_s),
                beta=float(beta_s) if beta_s is not None else 0.0,
                gamma=float(gamma_s) if gamma_s is not None else None,
                T_days=T_days,
                aic=float(aic_s),
                aic_linear=float(aic_linear_val),
                rss=float(rss_s),
                selected_reason=reason,
                t_anchor_date=start_date,
            )
        )

    return segments
