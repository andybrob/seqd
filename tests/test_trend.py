"""Tests for Stage 5: Piecewise trend fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqd._trend import (
    compute_aic,
    evaluate_segment,
    fit_piecewise_trend,
    fit_segment,
)
from seqd._structures import SegmentTrend
from seqd._forecast import _project_trend


def make_segment_trend(
    model_type="linear",
    alpha=1.0,
    beta=2.0,
    gamma=None,
    T_days=100,
    n_obs=100,
):
    """Helper to construct a SegmentTrend for evaluate_segment tests."""
    start = pd.Timestamp("2021-01-01")
    end = start + pd.Timedelta(days=n_obs - 1)
    return SegmentTrend(
        segment_index=1,
        start_date=start,
        end_date=end,
        n_obs=n_obs,
        model_type=model_type,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        T_days=T_days,
        aic=0.0,
        aic_linear=0.0,
        rss=0.0,
        selected_reason="test",
        t_anchor_date=start,
    )


def make_series(values, start="2021-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=dates, dtype=float)


# ---------------------------------------------------------------------------
# compute_aic
# ---------------------------------------------------------------------------


def test_compute_aic_basic():
    """AIC = n*ln(rss/n) + 2k."""
    n, k, rss = 100, 2, 10.0
    expected = n * np.log(rss / n) + 2 * k
    assert abs(compute_aic(rss, n, k) - expected) < 1e-10


def test_compute_aic_rss_zero():
    assert compute_aic(0.0, 100, 2) == -np.inf


def test_compute_aic_saturated():
    assert compute_aic(1.0, 2, 2) == np.inf


# ---------------------------------------------------------------------------
# fit_segment
# ---------------------------------------------------------------------------


def test_fit_segment_linear_recovery():
    """Linear fit should recover known α=1, β=3."""
    n = 50
    t = np.linspace(0, 1, n)
    r = 1.0 + 3.0 * t
    a, b, g, rss = fit_segment(r, "linear", t, T_days=n)
    assert g is None
    assert abs(a - 1.0) < 1e-8
    assert abs(b - 3.0) < 1e-8
    assert rss < 1e-20


def test_fit_segment_log_recovery():
    """Log fit should recover known coefficients."""
    n = 100
    t = np.linspace(0, 1, n)
    T_days = n
    true_alpha, true_beta = 5.0, 2.0
    r = true_alpha + true_beta * np.log1p(t * T_days)
    a, b, g, rss = fit_segment(r, "log", t, T_days=T_days)
    assert g is None
    assert abs(a - true_alpha) < 1e-6
    assert abs(b - true_beta) < 1e-6
    assert rss < 1e-15


def test_fit_segment_exp_recovery():
    """Exponential fit should recover known α=5, β=0.5."""
    n = 100
    t = np.linspace(0, 1, n)
    true_alpha, true_beta = 5.0, 0.5
    r = true_alpha * np.exp(true_beta * t)
    a, b, g, rss = fit_segment(r, "exp", t, T_days=n)
    assert g is None
    assert abs(a - true_alpha) < 1e-6
    assert abs(b - true_beta) < 1e-6
    assert rss < 1e-10


def test_fit_segment_exp_with_zero_returns_inf():
    """Exponential fit should return (None, None, None, inf) when values <= 0."""
    n = 50
    t = np.linspace(0, 1, n)
    r = np.ones(n)
    r[10] = 0.0  # include zero
    a, b, g, rss = fit_segment(r, "exp", t, T_days=n)
    assert a is None
    assert b is None
    assert g is None
    assert rss == np.inf


def test_fit_segment_exp_negative_returns_inf():
    """Exponential fit returns inf when values include negatives."""
    n = 50
    t = np.linspace(0, 1, n)
    r = np.full(n, -1.0)
    a, b, g, rss = fit_segment(r, "exp", t, T_days=n)
    assert rss == np.inf


def test_fit_segment_quadratic_recovery():
    """Quadratic fit should recover known α=1, β=2, γ=3."""
    n = 50
    t = np.linspace(0, 1, n)
    r = 1.0 + 2.0 * t + 3.0 * t ** 2
    a, b, g, rss = fit_segment(r, "quadratic", t, T_days=n)
    assert abs(a - 1.0) < 1e-8
    assert abs(b - 2.0) < 1e-8
    assert abs(g - 3.0) < 1e-8
    assert rss < 1e-18


def test_fit_segment_quadratic_short_returns_inf():
    """Quadratic fit should return inf for n < 5."""
    n = 4
    t = np.linspace(0, 1, n)
    r = np.ones(n) * 5.0
    a, b, g, rss = fit_segment(r, "quadratic", t, T_days=n)
    assert rss == np.inf


# ---------------------------------------------------------------------------
# evaluate_segment
# ---------------------------------------------------------------------------


def test_evaluate_linear():
    seg = make_segment_trend("linear", alpha=2.0, beta=4.0)
    assert abs(evaluate_segment(seg, 0.0) - 2.0) < 1e-10
    assert abs(evaluate_segment(seg, 1.0) - 6.0) < 1e-10
    assert abs(evaluate_segment(seg, 2.0) - 10.0) < 1e-10  # extrapolation


def test_evaluate_log():
    T_days = 100
    seg = make_segment_trend("log", alpha=1.0, beta=2.0, T_days=T_days)
    # t=0: log(1 + 0) = 0 → value = alpha = 1
    assert abs(evaluate_segment(seg, 0.0) - 1.0) < 1e-10
    # t=1: log(1 + T_days)
    expected = 1.0 + 2.0 * np.log1p(T_days)
    assert abs(evaluate_segment(seg, 1.0) - expected) < 1e-10
    # t=2 (extrapolation)
    expected_extrap = 1.0 + 2.0 * np.log1p(2 * T_days)
    assert abs(evaluate_segment(seg, 2.0) - expected_extrap) < 1e-10


def test_evaluate_exp():
    seg = make_segment_trend("exp", alpha=3.0, beta=0.5)
    assert abs(evaluate_segment(seg, 0.0) - 3.0) < 1e-10
    assert abs(evaluate_segment(seg, 1.0) - 3.0 * np.exp(0.5)) < 1e-10


def test_evaluate_quadratic():
    seg = make_segment_trend("quadratic", alpha=1.0, beta=2.0, gamma=3.0)
    assert abs(evaluate_segment(seg, 0.0) - 1.0) < 1e-10
    assert abs(evaluate_segment(seg, 1.0) - 6.0) < 1e-10
    # t=2: 1 + 4 + 12 = 17
    assert abs(evaluate_segment(seg, 2.0) - 17.0) < 1e-10


def test_evaluate_constant():
    seg = make_segment_trend("constant", alpha=7.0, beta=0.0)
    assert evaluate_segment(seg, 0.0) == 7.0
    assert evaluate_segment(seg, 10.0) == 7.0


# ---------------------------------------------------------------------------
# fit_piecewise_trend
# ---------------------------------------------------------------------------


def test_fit_piecewise_no_changepoints():
    """Single-segment case: all data in one segment."""
    n = 200
    t = np.linspace(0, 1, n)
    r = 5.0 + 3.0 * t + np.random.default_rng(0).normal(0, 0.01, n)
    y = make_series(r)
    segs = fit_piecewise_trend(y, changepoint_dates=[], aic_linear_delta=2.0)
    assert len(segs) == 1
    assert segs[0].segment_index == 1
    assert segs[0].n_obs == n


def test_fit_piecewise_one_changepoint():
    """Two-segment case with one changepoint."""
    n1, n2 = 150, 150
    t1 = np.linspace(0, 1, n1)
    t2 = np.linspace(0, 1, n2)
    seg1 = 1.0 + 2.0 * t1
    seg2 = 10.0 + 5.0 * t2
    r = np.concatenate([seg1, seg2])
    y = make_series(r)
    cp_date = y.index[n1]
    segs = fit_piecewise_trend(y, changepoint_dates=[cp_date], aic_linear_delta=2.0)
    assert len(segs) == 2
    assert segs[0].n_obs == n1
    assert segs[1].n_obs == n2
    assert segs[1].segment_index == 2


def test_fit_piecewise_three_changepoints():
    """Four-segment case."""
    rng = np.random.default_rng(42)
    chunks = [rng.normal(lvl, 1.0, 120) for lvl in [0, 10, 20, 30]]
    r = np.concatenate(chunks)
    y = make_series(r)
    cps = [y.index[120], y.index[240], y.index[360]]
    segs = fit_piecewise_trend(y, changepoint_dates=cps, aic_linear_delta=2.0)
    assert len(segs) == 4
    for i, seg in enumerate(segs, 1):
        assert seg.segment_index == i


def test_fit_piecewise_all_negative_no_exp():
    """Negative segment should not select exponential model."""
    n = 100
    t = np.linspace(0, 1, n)
    r = -10.0 - 2.0 * t  # all negative
    y = make_series(r)
    segs = fit_piecewise_trend(y, changepoint_dates=[], aic_linear_delta=2.0)
    assert segs[0].model_type != "exp"


def test_fit_piecewise_short_segment_no_quadratic():
    """A segment with n=3 should not select quadratic."""
    n1, n2 = 150, 3
    r = np.concatenate([np.ones(n1), np.array([1.0, 2.0, 3.0])])
    y = make_series(r)
    cp = y.index[n1]
    segs = fit_piecewise_trend(y, changepoint_dates=[cp])
    short_seg = segs[1]
    assert short_seg.n_obs == n2
    assert short_seg.model_type != "quadratic"


def test_fit_piecewise_constant_segment():
    """All-constant segment should produce model_type='constant'."""
    n = 150
    r = np.full(n, 7.0)
    y = make_series(r)
    segs = fit_piecewise_trend(y, changepoint_dates=[], aic_linear_delta=2.0)
    assert segs[0].model_type == "constant"
    assert segs[0].alpha == pytest.approx(7.0)


def test_fit_piecewise_linear_parsimony():
    """When best non-linear model is only marginally better (ΔAIC < 2), prefer linear."""
    rng = np.random.default_rng(5)
    n = 200
    t = np.linspace(0, 1, n)
    # Slightly curved but essentially linear
    r = 1.0 + 2.0 * t + 0.0001 * t ** 2 + rng.normal(0, 0.001, n)
    y = make_series(r)
    segs = fit_piecewise_trend(y, changepoint_dates=[], aic_linear_delta=2.0)
    assert segs[0].model_type == "linear"


# ---------------------------------------------------------------------------
# Linear continuation for quadratic extrapolation (Fix 2)
# ---------------------------------------------------------------------------


def test_quadratic_extrapolation_linear_continuation():
    """_project_trend with quadratic model should use linear continuation for t > 1.

    The continuation must be C0-smooth (no jump at t=1) and must NOT
    reproduce the pure quadratic value at t > 1.
    """
    # Build a quadratic segment: alpha=10, beta=5, gamma=2
    # f(t) = 10 + 5t + 2t^2
    # f(1) = 17, slope at t=1 = 5 + 4 = 9
    seg = make_segment_trend("quadratic", alpha=10.0, beta=5.0, gamma=2.0, n_obs=100)

    # Project with a long horizon so t >> 1
    horizon = 200  # t_max = (99 + 200) / 99 ≈ 3.02
    arr, warns = _project_trend(seg, horizon=horizon, max_extrapolation_days=365)
    assert len(arr) == horizon

    # At t=1 (h=1): still within segment end, should match quadratic
    # t at h=1 is (99 + 1)/99 = 100/99 ≈ 1.0101 — just past the boundary,
    # linear continuation applies: f(1) + 9*(t-1)
    f_at_end = 10.0 + 5.0 + 2.0  # = 17
    slope_at_end = 5.0 + 4.0     # = 9

    for i, h in enumerate(range(1, horizon + 1)):
        t_i = (99 + h) / 99.0
        if t_i > 1.0:
            expected = f_at_end + slope_at_end * (t_i - 1.0)
            assert abs(arr[i] - expected) < 1e-8, (
                f"h={h}, t={t_i:.4f}: expected linear {expected:.4f}, got {arr[i]:.4f}"
            )
        else:
            expected_quad = 10.0 + 5.0 * t_i + 2.0 * t_i ** 2
            assert abs(arr[i] - expected_quad) < 1e-8


def test_quadratic_extrapolation_does_not_explode():
    """Quadratic extrapolation capped at t=1 should produce bounded, reasonable values."""
    # Use parameters that would explode quadratically: gamma=100 (large)
    seg = make_segment_trend("quadratic", alpha=100.0, beta=5.0, gamma=100.0, n_obs=93)
    # t at end of 365-day forecast: (92 + 365) / 92 ≈ 4.97
    arr, _ = _project_trend(seg, horizon=365, max_extrapolation_days=365)

    # Without the fix, arr[-1] would be ~100 + 5*4.97 + 100*4.97^2 ≈ 2594.
    # With the fix (linear continuation from t=1): f(1) = 205, slope = 205
    # so arr[-1] ≈ 205 + 205 * (4.97 - 1) ≈ 205 + 814 ≈ 1019.
    # The key assertion: it must be less than the uncapped quadratic would give.
    f_at_end = 100.0 + 5.0 + 100.0   # = 205
    slope_at_end = 5.0 + 200.0        # = 205
    t_last = (92 + 365) / 92.0
    expected_last = f_at_end + slope_at_end * (t_last - 1.0)
    uncapped_last = 100.0 + 5.0 * t_last + 100.0 * t_last ** 2

    assert abs(arr[-1] - expected_last) < 1e-6
    assert arr[-1] < uncapped_last, "Capped value should be less than uncapped quadratic"


def test_exp_extrapolation_linear_continuation():
    """_project_trend with exp model caps to linear continuation for t > 1."""
    # alpha=10, beta=0.5 → f(1) = 10*exp(0.5), slope at 1 = 10*0.5*exp(0.5)
    seg = make_segment_trend("exp", alpha=10.0, beta=0.5, n_obs=100)

    horizon = 200
    arr, _ = _project_trend(seg, horizon=horizon, max_extrapolation_days=365)

    f_at_end = 10.0 * np.exp(0.5)
    slope_at_end = 10.0 * 0.5 * np.exp(0.5)

    for i, h in enumerate(range(1, horizon + 1)):
        t_i = (99 + h) / 99.0
        if t_i > 1.0:
            expected = f_at_end + slope_at_end * (t_i - 1.0)
            assert abs(arr[i] - expected) < 1e-8, (
                f"h={h}, t={t_i:.4f}: expected {expected:.4f}, got {arr[i]:.4f}"
            )


def test_project_trend_returns_warnings_list():
    """_project_trend must return a (array, list) tuple."""
    seg = make_segment_trend("linear", alpha=1.0, beta=0.5, n_obs=100)
    result = _project_trend(seg, horizon=30, max_extrapolation_days=365)
    assert isinstance(result, tuple)
    arr, warns = result
    assert isinstance(arr, np.ndarray)
    assert isinstance(warns, list)


def test_project_trend_quadratic_warning_collected():
    """Quadratic extrapolation warning should appear in returned warnings list."""
    import warnings as _warnings
    seg = make_segment_trend("quadratic", alpha=10.0, beta=5.0, gamma=2.0, n_obs=93)
    with _warnings.catch_warnings(record=True):
        _warnings.simplefilter("always")
        _, warns = _project_trend(seg, horizon=365, max_extrapolation_days=365)
    assert any("quadratic" in w.lower() for w in warns), (
        f"Expected quadratic warning in {warns}"
    )
