"""Tests for Stage 4: Changepoint detection."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from seqd._changepoint import detect_changepoints


def make_series(values, start="2020-01-01"):
    """Build a pd.Series with a daily DatetimeIndex."""
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=dates, dtype=float)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------


def test_constant_series_no_changepoints():
    """A flat series should produce no changepoints."""
    y = make_series(np.ones(400))
    dates, indices = detect_changepoints(y, penalty_beta=3.0, min_size=90)
    assert dates == []
    assert indices == []


def test_level_shift_one_changepoint():
    """A clear level shift at the midpoint should produce one changepoint at exactly index 200."""
    n = 400
    values = np.zeros(n)
    values[200:] = 100.0
    y = make_series(values)

    dates, indices = detect_changepoints(y, penalty_beta=3.0, min_size=90)
    assert len(dates) == 1
    # Changepoint index is the 0-based first index of the new segment (index 200).
    assert indices[0] == 200
    # The changepoint date corresponds to the first day of the elevated segment.
    assert dates[0] == y.index[200]


def test_series_too_short_returns_empty():
    """Series with n < 2 * min_size should return empty and emit UserWarning."""
    y = make_series(np.random.default_rng(0).normal(100, 5, 100))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dates, indices = detect_changepoints(y, min_size=90)
    assert dates == []
    assert indices == []
    assert any("No changepoints possible" in str(warning.message) for warning in w)


def test_near_zero_variance_warning():
    """Near-zero variance series should emit a floor-penalty warning."""
    # All identical values → variance = 0
    y = make_series(np.full(400, 50.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detect_changepoints(y, penalty_beta=3.0, min_size=90)
    # Should warn about near-zero variance
    messages = [str(warning.message) for warning in w]
    assert any("near-zero variance" in m for m in messages)


def test_few_segments_warning():
    """Series with n < 4 * min_size should emit a soft warning."""
    # n=300, min_size=90 → 300 < 4*90=360
    y = make_series(np.random.default_rng(42).normal(0, 5, 300))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detect_changepoints(y, min_size=90)
    messages = [str(warning.message) for warning in w]
    assert any("allows at most" in m for m in messages)


def test_higher_penalty_fewer_changepoints():
    """Higher penalty_beta should produce fewer changepoints."""
    rng = np.random.default_rng(7)
    n = 800
    # Three segments with distinct levels
    values = np.concatenate([
        rng.normal(0, 5, 200),
        rng.normal(50, 5, 400),
        rng.normal(100, 5, 200),
    ])
    y = make_series(values)

    dates_low, _ = detect_changepoints(y, penalty_beta=0.5, min_size=90)
    dates_high, _ = detect_changepoints(y, penalty_beta=10.0, min_size=90)
    assert len(dates_low) >= len(dates_high)


def test_return_types():
    """Return types must be list[pd.Timestamp] and list[int]."""
    y = make_series(np.random.default_rng(0).normal(0, 5, 400))
    dates, indices = detect_changepoints(y, min_size=90)
    assert isinstance(dates, list)
    assert isinstance(indices, list)
    for d in dates:
        assert isinstance(d, pd.Timestamp)
    for i in indices:
        assert isinstance(i, int)
