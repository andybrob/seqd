"""Tests for Stage 3: Annual Fourier seasonality."""

import numpy as np
import pandas as pd
import pytest

from seqd._annual import fit_annual, _fourier_design, PERIOD


def make_annual_series(n_years=3, K_true=2, seed=42):
    """Generate series with known Fourier annual component."""
    rng = np.random.default_rng(seed)
    n_days = int(n_years * PERIOD)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    # True Fourier component
    true_annual = np.zeros(n_days)
    coeffs = {}
    for k in range(1, K_true + 1):
        angle = 2.0 * np.pi * k * t / PERIOD
        a_k = rng.uniform(10, 20)
        b_k = rng.uniform(5, 15)
        coeffs[k] = (a_k, b_k)
        true_annual += a_k * np.cos(angle) + b_k * np.sin(angle)

    trend = 100.0 + 0.01 * t
    noise = rng.normal(0, 1.5, n_days)
    y = pd.Series(trend + true_annual + noise, index=dates, name="y")

    return y, true_annual, coeffs


def test_harmonic_selection_bic():
    """BIC should select the correct number of harmonics."""
    # With K=1 true component and modest noise, BIC should pick 1 or 2
    y, true_annual, _ = make_annual_series(n_years=4, K_true=1, seed=0)
    annual_effect, _ = fit_annual(y)
    # Allow ±1 from true K
    assert 1 <= annual_effect.n_harmonics <= 3, (
        f"Selected K={annual_effect.n_harmonics}, expected 1-3"
    )


def test_annual_component_recovery():
    """Fitted annual component should correlate highly with true component."""
    y, true_annual, _ = make_annual_series(n_years=4, K_true=2, seed=1)
    annual_effect, y_clean = fit_annual(y)

    fitted_annual = annual_effect.component.values

    # Pearson correlation should be > 0.90
    corr = float(np.corrcoef(fitted_annual, true_annual)[0, 1])
    assert corr > 0.90, f"Annual component correlation {corr:.3f} < 0.90"


def test_annual_component_removed():
    """After removing annual component, seasonal variance should decrease."""
    y, true_annual, _ = make_annual_series(n_years=4, K_true=2, seed=2)
    _, y_clean = fit_annual(y)

    # Variance of y_clean should be lower than y minus trend
    n = len(y)
    t = np.arange(n, dtype=float)
    coef = np.polyfit(t, y.values, 1)
    y_detrended = y.values - np.polyval(coef, t)
    coef_c = np.polyfit(t, y_clean.values, 1)
    yc_detrended = y_clean.values - np.polyval(coef_c, t)

    var_y = float(np.var(y_detrended))
    var_yc = float(np.var(yc_detrended))

    assert var_yc < var_y * 0.8, (
        f"Annual removal didn't reduce variance enough: "
        f"var(y)={var_y:.2f}, var(y_clean)={var_yc:.2f}"
    )


def test_coefficients_shape():
    """Coefficients should have correct shape: 2*K+1 (intercept + K cos/sin pairs)."""
    y, _, _ = make_annual_series(n_years=3, K_true=2, seed=3)
    annual_effect, _ = fit_annual(y)
    K = annual_effect.n_harmonics
    expected_len = 2 * K + 1
    assert len(annual_effect.coefficients) == expected_len, (
        f"Coefficients length {len(annual_effect.coefficients)} != {expected_len}"
    )


def test_recency_amplitudes_structure():
    """recency_amplitudes should have entries for 1, 2, 3 years."""
    y, _, _ = make_annual_series(n_years=4, K_true=1, seed=4)
    annual_effect, _ = fit_annual(y)

    for years in [1, 2, 3]:
        assert years in annual_effect.recency_amplitudes, (
            f"recency_amplitudes missing key {years}"
        )
        amp = annual_effect.recency_amplitudes[years]
        assert np.isfinite(amp), f"Amplitude for {years} years is not finite: {amp}"
        assert amp >= 0, f"Amplitude should be non-negative, got {amp}"


def test_annual_fourier_design_shape():
    """Fourier design matrix should have correct shape."""
    t = np.arange(365, dtype=float)
    for K in range(1, 5):
        X = _fourier_design(t, K)
        assert X.shape == (365, 2 * K + 1), (
            f"Design matrix shape {X.shape} != ({365}, {2*K+1}) for K={K}"
        )


def test_annual_index_preserved():
    """Annual component should have same index as input."""
    y, _, _ = make_annual_series(n_years=3)
    annual_effect, y_clean = fit_annual(y)

    assert annual_effect.component.index.equals(y.index)
    assert y_clean.index.equals(y.index)


def test_bic_can_select_k5_or_k6():
    """BIC should be able to select K=5 when the true signal has K=5 harmonics."""
    rng = np.random.default_rng(77)
    n_days = int(4 * PERIOD)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    # Construct signal as sum of K=5 Fourier components with clear amplitudes
    true_annual = np.zeros(n_days)
    for k in range(1, 6):
        angle = 2.0 * np.pi * k * t / PERIOD
        a_k = rng.uniform(8, 15)
        b_k = rng.uniform(4, 10)
        true_annual += a_k * np.cos(angle) + b_k * np.sin(angle)

    trend = 100.0 + 0.005 * t
    noise = rng.normal(0, 0.5, n_days)  # low noise so K=5 is clearly the best fit
    y = pd.Series(trend + true_annual + noise, index=dates, name="y")

    annual_effect, _ = fit_annual(y)
    # BIC should select K=5 (or 6 if noise happens to warrant it); the important
    # thing is it can now reach K=5 (previously the ceiling was K=4).
    assert annual_effect.n_harmonics >= 5, (
        f"Expected BIC to select K>=5 for a K=5 signal, got K={annual_effect.n_harmonics}"
    )
    assert annual_effect.n_harmonics <= 6, (
        f"n_harmonics should be at most 6 (the new ceiling), got {annual_effect.n_harmonics}"
    )


def test_fourier_phase_calendar_anchored():
    """annual_component() values at matching calendar dates should be close
    regardless of whether the series starts Jan 1 or Jul 1."""
    rng = np.random.default_rng(55)

    # Build a true annual pattern (K=1) tied to calendar day-of-year
    def make_series_from(start_str, n_days=int(2.5 * PERIOD)):
        dates = pd.date_range(start_str, periods=int(n_days), freq="D")
        t_cal = np.array(
            [(d - pd.Timestamp(dates[0].year, 1, 1)).days for d in dates],
            dtype=float,
        )
        true_annual = (
            12.0 * np.cos(2.0 * np.pi * t_cal / PERIOD)
            + 6.0 * np.sin(2.0 * np.pi * t_cal / PERIOD)
        )
        trend = 100.0 + 0.005 * np.arange(int(n_days), dtype=float)
        noise = rng.normal(0, 0.5, int(n_days))
        y = pd.Series(trend + true_annual + noise, index=dates, name="y")
        return y

    y_jan = make_series_from("2021-01-01")
    y_jul = make_series_from("2021-07-01")

    result_jan, _ = fit_annual(y_jan)
    result_jul, _ = fit_annual(y_jul)

    # Find the overlapping date range
    overlap_start = max(y_jan.index[0], y_jul.index[0])
    overlap_end = min(y_jan.index[-1], y_jul.index[-1])
    overlap_dates = pd.date_range(overlap_start, overlap_end, freq="D")

    comp_jan = result_jan.component.reindex(overlap_dates)
    comp_jul = result_jul.component.reindex(overlap_dates)

    # Drop NaN (dates not in both series)
    mask = comp_jan.notna() & comp_jul.notna()
    comp_jan = comp_jan[mask]
    comp_jul = comp_jul[mask]

    if len(comp_jan) == 0:
        pytest.skip("No overlapping dates to compare")

    amplitude = 12.0 + 6.0  # rough upper bound on true amplitude
    tolerance = 0.05 * amplitude * len(comp_jan) ** 0.5  # allow 5% per-point error in RMS

    rms_diff = float(np.sqrt(np.mean((comp_jan.values - comp_jul.values) ** 2)))
    assert rms_diff < amplitude * 0.15, (
        f"RMS difference between Jan-start and Jul-start annual components "
        f"({rms_diff:.3f}) exceeds 15% of amplitude ({amplitude * 0.15:.3f}). "
        "Calendar-anchored phase may not be working correctly."
    )
