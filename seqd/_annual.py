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
    t = np.arange(n, dtype=float)  # days since first observation

    # Select number of harmonics by BIC
    best_K, best_bic = 1, np.inf
    for K in range(1, 5):
        X = _fourier_design(t, K)
        coef = ols_fit(X, y_h.values)
        # Suppress spurious overflow/divide-by-zero warnings from numpy 2.0+ matmul
        # when the result is finite (known numpy issue with float64 matmul)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            fitted = X @ coef
        if not np.all(np.isfinite(fitted)):
            continue  # skip this K if result is genuinely non-finite
        rss = float(np.sum((y_h.values - fitted) ** 2))
        n_params = 2 * K + 1  # K cosine + K sine + intercept
        bic = n * np.log(max(rss / n, 1e-30)) + n_params * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_K = K

    # Fit with selected K
    X = _fourier_design(t, best_K)
    coef = ols_fit(X, y_h.values)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        fitted_full = X @ coef

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
    """Compute K=1 Fourier amplitude on trailing [1, 2, 3]-year windows."""
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

        X = _fourier_design(t_win, K=1)
        try:
            coef = ols_fit(X, y_win)
            # coef = [a0, a1, b1]
            amplitude = float(np.sqrt(coef[1] ** 2 + coef[2] ** 2))
        except Exception:
            amplitude = np.nan
        result[years] = amplitude

    return result
