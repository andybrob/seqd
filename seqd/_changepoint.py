"""Stage 4: Changepoint detection via PELT with BIC-inspired penalty."""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt


def detect_changepoints(
    residual: pd.Series,
    penalty_beta: float = 3.0,
    min_size: int = 90,
) -> Tuple[List[pd.Timestamp], List[int]]:
    """Detect changepoints in the V1 residual series using PELT.

    Uses the ``"l2"`` (least-squares, mean-shift) cost function with a
    BIC-inspired penalty scaled by robust variance and series length.

    Parameters
    ----------
    residual : pd.Series
        V1 residual series (trend + noise) with a daily ``DatetimeIndex``.
    penalty_beta : float
        Multiplier β for the penalty formula λ = σ̂² × β × ln(n).
        Default 3.0.  Reasonable range [1.0, 6.0].
    min_size : int
        Minimum segment size in days.  Default 90 (≈ one quarter).
        Must satisfy ``n >= 2 * min_size`` for any changepoints to be
        detectable.

    Returns
    -------
    changepoint_dates : list of pd.Timestamp
        Detected changepoint dates sorted ascending.  May be empty.
        Convention: a changepoint date ``τ`` means a new segment begins on
        that date.
    changepoint_indices : list of int
        Corresponding 0-based positions in ``residual``.

    Raises
    ------
    RuntimeError
        If the number of changepoints found violates the minimum-segment
        constraint (defensive guard; should never occur under normal
        ``ruptures`` operation).

    Warnings
    --------
    UserWarning
        Emitted when the series is too short for changepoints, when variance
        is near-zero, or when the series allows very few segments.
    """
    if penalty_beta <= 0:
        raise ValueError(
            f"penalty_beta must be > 0 (got {penalty_beta})."
        )
    if min_size < 2:
        raise ValueError(
            f"min_size must be >= 2 (got {min_size})."
        )

    r_t = residual.dropna().copy().astype(float)
    n = len(r_t)

    # Guard: series too short for any changepoints
    if n < 2 * min_size:
        warnings.warn(
            f"Series length ({n}) < 2 * min_size ({2 * min_size}). "
            "No changepoints possible.",
            UserWarning,
            stacklevel=2,
        )
        return [], []

    # Soft warning: very few segments possible
    if n < 4 * min_size:
        warnings.warn(
            f"Series length ({n}) allows at most {n // min_size} segments "
            f"with min_size={min_size}. "
            "Consider reducing min_size for short series.",
            UserWarning,
            stacklevel=2,
        )

    # Robust variance estimate: σ̂² = (MAD / 0.6745)²
    r_values = r_t.values
    median_r = np.median(r_values)
    mad = np.median(np.abs(r_values - median_r))
    sigma2_hat = (mad / 0.6745) ** 2

    # Near-zero variance guard
    r_bar = float(np.mean(r_values))
    near_zero = (abs(r_bar) > 1e-10 and sigma2_hat < 1e-10 * abs(r_bar)) or (
        abs(r_bar) <= 1e-10 and sigma2_hat < 1.0
    )
    if near_zero:
        warnings.warn(
            "Residual has near-zero variance. "
            "Setting penalty floor to prevent over-segmentation.",
            UserWarning,
            stacklevel=2,
        )
        sigma2_hat = 1.0

    # Penalty: λ = σ̂² × β × ln(n)
    penalty = sigma2_hat * penalty_beta * np.log(n)

    # PELT with l2 cost
    signal = r_values.reshape(-1, 1)
    model = rpt.Pelt(model="l2", min_size=min_size, jump=1)
    model.fit(signal)
    breakpoints = model.predict(pen=penalty)

    # breakpoints[-1] is always n (sentinel end) — drop it.
    # ruptures returns breakpoints as 0-indexed start positions of new segments:
    # e.g. for a shift at index k, ruptures returns k (not k+1).
    # These values ARE the 0-based first indices of the new segments, so no
    # subtraction is needed.  The variable name below is kept for clarity.
    changepoint_indices_0based = list(breakpoints[:-1])
    changepoint_dates = [r_t.index[i] for i in changepoint_indices_0based]

    # Defensive guard
    max_allowed = n // min_size - 1
    if len(changepoint_indices_0based) > max_allowed:
        raise RuntimeError(
            f"ruptures returned {len(changepoint_indices_0based)} changepoints, "
            f"exceeding the maximum allowed {max_allowed} for n={n}, "
            f"min_size={min_size}. This should not occur."
        )

    return changepoint_dates, changepoint_indices_0based
