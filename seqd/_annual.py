"""Stage 3: Annual Fourier seasonality extraction."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ._structures import AnnualEffect
from ._utils import ols_fit


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

PERIOD = 365.25


def fit_annual(y_h: pd.Series) -> Tuple[AnnualEffect, pd.Series]:
    """Fit and remove annual Fourier seasonality.

    Parameters
    ----------
    y_h : pd.Series
        Holiday-adjusted series with DatetimeIndex.

    Returns
    -------
    annual_effect : AnnualEffect
    y_clean : pd.Series
        Series with annual effect removed.
    """
    y_h = y_h.copy().astype(float)
    idx = y_h.index
    n = len(y_h)
    # Anchor t=0 to January 1 of the series' first year for calendar-aligned phase
    t0_date = pd.Timestamp(idx[0].year, 1, 1)
    t = np.array([(d - t0_date).days for d in idx], dtype=float)

    # Detrend y_h with a linear OLS before BIC selection so that a strong
    # upward (or downward) trend does not compete with the Fourier harmonics
    # for variance.  We fit and remove the trend here for BIC selection only;
    # the actual component fitting below uses the original y_h so that the
    # returned annual_component and y_clean remain on the original scale.
    y_h_detrended = _linear_detrend(y_h.values, t)

    # Select number of harmonics by BIC.
    # K=0 (intercept only) is included so that a series with no annual
    # seasonality is not forced to absorb a spurious Fourier harmonic.
    best_K, best_bic = 0, np.inf
    for K in range(0, 7):
        if K == 0:
            X = np.ones((n, 1))
            n_params = 1
        else:
            X = _fourier_design(t, K)
            n_params = 2 * K + 1  # K cosine + K sine + intercept
        coef = ols_fit(X, y_h_detrended)
        # Suppress spurious overflow/divide-by-zero warnings from numpy 2.0+ matmul
        # when the result is finite (known numpy issue with float64 matmul)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            fitted = X @ coef
        if not np.all(np.isfinite(fitted)):
            continue  # skip this K if result is genuinely non-finite
        rss = float(np.sum((y_h_detrended - fitted) ** 2))
        bic = n * np.log(max(rss / n, 1e-30)) + n_params * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_K = K

    # Fit with selected K
    if best_K == 0:
        # Intercept-only model: no annual seasonality detected.
        # annual_component is identically zero (nothing to remove).
        annual_component = np.zeros(n)
    else:
        # Fit Fourier on the *linearly-detrended* y_h rather than raw y_h.
        #
        # Rationale: fitting on raw y_h embeds the mean level in the intercept
        # and computes seasonal amplitudes relative to that global mean.  For a
        # growing series (e.g. revenue with a strong upward trend), this means
        # the annual component has a *fixed* amplitude equal to the average
        # seasonal swing across the whole sample.  In early years the component
        # over-subtracts the seasonal uplift (leaving a negative residual in
        # those months); in later years it under-subtracts (leaving a positive
        # residual).  The net effect is residual autocorrelation at lag 364 and
        # a distorted seasonal pattern.
        #
        # Fitting on the linearly-detrended series isolates the pure cyclical
        # component from the trend, so the Fourier coefficients represent
        # deviations from the local trend rather than from the global mean.
        # The annual_component (intercept excluded) is then on the correct scale
        # to subtract from y_h.
        X = _fourier_design(t, best_K)
        coef = ols_fit(X, y_h_detrended)

        # Annual component excludes intercept (coef[0])
        X_no_intercept = X.copy()
        X_no_intercept[:, 0] = 0.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            annual_component = X_no_intercept @ coef

    y_clean = pd.Series(
        y_h.values - annual_component,
        index=idx,
        name=y_h.name,
    )

    # Recency amplitudes
    recency_amplitudes = _recency_amplitudes(y_h, t)

    annual = AnnualEffect(
        n_harmonics=best_K,
        coefficients=coef,
        component=pd.Series(annual_component, index=idx, name="annual_component"),
        recency_amplitudes=recency_amplitudes,
    )
    return annual, y_clean


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _linear_detrend(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove a linear OLS trend from y using t as the time axis.

    Returns y minus the fitted linear trend, preserving the cyclical component.
    """
    X = np.column_stack([np.ones_like(t), t])
    coef = ols_fit(X, y)
    trend = coef[0] + coef[1] * t
    return y - trend


def _fourier_design(t: np.ndarray, K: int) -> np.ndarray:
    """Build Fourier design matrix with intercept.

    Columns: [1, cos(2pi*1*t/P), sin(2pi*1*t/P), ..., cos(2pi*K*t/P), sin(2pi*K*t/P)]
    """
    cols = [np.ones_like(t)]
    for k in range(1, K + 1):
        angle = 2.0 * np.pi * k * t / PERIOD
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    return np.column_stack(cols)


def _recency_amplitudes(y_h: pd.Series, t: np.ndarray) -> Dict[int, float]:
    """Compute K=1 Fourier amplitude on trailing [1, 2, 3]-year windows.

    Each window is linearly detrended before fitting so that a strong trend
    does not inflate or deflate the estimated cyclical amplitude.
    """
    n = len(y_h)
    result = {}
    for years in [1, 2, 3]:
        window_days = int(years * PERIOD)
        if window_days >= n:
            start_idx = 0
        else:
            start_idx = n - window_days

        t_win = t[start_idx:] - t[start_idx]  # reset t to 0
        y_win = y_h.values[start_idx:].astype(float)

        if len(t_win) < 30:
            result[years] = np.nan
            continue

        # Detrend the window before Fourier fitting
        y_win_detrended = _linear_detrend(y_win, t_win)

        X = _fourier_design(t_win, K=1)
        try:
            coef = ols_fit(X, y_win_detrended)
            # coef = [a0, a1, b1]
            amplitude = float(np.sqrt(coef[1] ** 2 + coef[2] ** 2))
        except Exception:
            amplitude = np.nan
        result[years] = amplitude

    return result
