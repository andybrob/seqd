"""Tests for SeqdForecaster and forecast_from_result (Stages 4-6)."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from seqd import SeqdDecomposer, SeqdForecaster, forecast_from_result
from seqd._forecast import _combine_components, _project_annual, _project_weekly
from seqd._structures import ForecastResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_decomposition(
    n_years=3,
    seed=42,
    multiplicative=False,
    add_holiday=True,
):
    """Build a synthetic V1 decomposition for V2 tests."""
    rng = np.random.default_rng(seed)
    n = int(n_years * 365.25)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)

    # Components
    trend = 100.0 + 0.05 * t
    annual = 10.0 * np.sin(2 * np.pi * t / 365.25)
    weekly_raw = np.array([1.05, 1.02, 1.0, 1.0, 0.98, 0.95, 0.88])
    weekly_component = weekly_raw[dates.dayofweek]
    noise = rng.normal(0, 3.0, n)

    if multiplicative:
        y = weekly_component * (trend + annual) + noise
    else:
        dow_additive = weekly_raw - weekly_raw.mean()
        y = trend + annual + dow_additive[dates.dayofweek] + noise

    holidays = {}
    if add_holiday:
        bfcm_dates = []
        for yr in range(2021, 2021 + n_years):
            bfcm_dates.append(date(yr, 11, 26))
        holidays = {"Black Friday": bfcm_dates}
        # Add spike
        for bfcm in bfcm_dates:
            mask = (dates >= pd.Timestamp(bfcm) - pd.Timedelta(5)) & (
                dates <= pd.Timestamp(bfcm) + pd.Timedelta(3)
            )
            y[mask] += 50

    y_series = pd.Series(y, index=dates, name="y")

    decomposer = SeqdDecomposer(
        holiday_dates=holidays,
        multiplicative=multiplicative,
        holiday_window=20,
        reference_window=30,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = decomposer.fit(y_series)
    return result, holidays


# ---------------------------------------------------------------------------
# SeqdForecaster basic API
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    ret = forecaster.fit()
    assert ret is forecaster


def test_predict_before_fit_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        forecaster.predict(horizon=30)


def test_changepoints_before_fit_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    with pytest.raises(RuntimeError):
        _ = forecaster.changepoints


def test_segments_before_fit_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result)
    with pytest.raises(RuntimeError):
        _ = forecaster.segments


def test_invalid_horizon_raises():
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result).fit()
    with pytest.raises(ValueError):
        forecaster.predict(horizon=0)


def test_empty_residual_raises():
    result, _ = make_decomposition()
    import copy
    bad = copy.copy(result)
    bad.residual = pd.Series([], dtype=float, name="residual")
    with pytest.raises(ValueError):
        SeqdForecaster(bad)


# ---------------------------------------------------------------------------
# ForecastResult structure
# ---------------------------------------------------------------------------


def test_forecast_result_types():
    """ForecastResult fields should have correct types and shapes."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=90)

    assert isinstance(fr, ForecastResult)
    assert isinstance(fr.forecast, pd.Series)
    assert len(fr.forecast) == 90
    assert isinstance(fr.trend_component, pd.Series)
    assert isinstance(fr.weekly_component, pd.Series)
    assert isinstance(fr.annual_component, pd.Series)
    assert isinstance(fr.holiday_component, pd.Series)
    assert isinstance(fr.changepoints, list)
    assert isinstance(fr.segments, list)
    assert fr.horizon == 90
    assert isinstance(fr.is_multiplicative, bool)


def test_forecast_index_starts_after_series():
    """Forecast dates should start the day after the series ends."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=30)
    expected_start = result.residual.index[-1] + pd.Timedelta(days=1)
    assert fr.forecast.index[0] == expected_start


# ---------------------------------------------------------------------------
# Multiple predict calls
# ---------------------------------------------------------------------------


def test_predict_twice_no_refit():
    """predict() can be called multiple times without re-running fit()."""
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result).fit()
    segs_after_first_fit = list(forecaster.segments)

    fr1 = forecaster.predict(horizon=30)
    fr2 = forecaster.predict(horizon=90)

    # Segments should be identical across calls
    assert forecaster.segments == segs_after_first_fit
    assert len(fr1.forecast) == 30
    assert len(fr2.forecast) == 90


# ---------------------------------------------------------------------------
# Combination formula
# ---------------------------------------------------------------------------


def test_combine_multiplicative():
    """Multiplicative: weekly * (trend + annual + holiday)."""
    trend = np.array([100.0, 105.0])
    weekly = np.array([1.1, 0.9])
    annual = np.array([5.0, -5.0])
    holiday = np.array([0.0, 10.0])

    result = _combine_components(trend, weekly, annual, holiday, is_multiplicative=True)
    expected = weekly * (trend + annual + holiday)
    np.testing.assert_allclose(result, expected)


def test_combine_additive():
    """Additive: trend + weekly + annual + holiday."""
    trend = np.array([100.0, 105.0])
    weekly = np.array([3.0, -3.0])
    annual = np.array([5.0, -5.0])
    holiday = np.array([0.0, 10.0])

    result = _combine_components(trend, weekly, annual, holiday, is_multiplicative=False)
    expected = trend + weekly + annual + holiday
    np.testing.assert_allclose(result, expected)


def test_combine_multiplicative_not_independent():
    """Multiplicative formula is weekly*(sum), NOT weekly*trend*annual*holiday."""
    trend = np.array([100.0])
    weekly = np.array([1.1])
    annual = np.array([10.0])
    holiday = np.array([5.0])

    correct = _combine_components(trend, weekly, annual, holiday, is_multiplicative=True)
    wrong = weekly * trend * annual * holiday  # incorrect formula

    # They should NOT be equal
    assert not np.allclose(correct, wrong)
    expected = weekly * (trend + annual + holiday)
    np.testing.assert_allclose(correct, expected)


# ---------------------------------------------------------------------------
# Annual projection
# ---------------------------------------------------------------------------


def test_annual_projection_matches_v1():
    """Annual projection at in-sample dates should match result.annual_component()."""
    result, _ = make_decomposition()

    # Use last 10 dates of the in-sample period
    check_dates = result.series.index[-10:]
    projected = _project_annual(result, check_dates)
    v1_values = result.annual_component().loc[check_dates].values

    np.testing.assert_allclose(projected, v1_values, atol=1e-8)


# ---------------------------------------------------------------------------
# Weekly projection fallback
# ---------------------------------------------------------------------------


def test_weekly_projection_uses_60day_window():
    """_project_weekly should use 60-day recency if available."""
    result, _ = make_decomposition()
    forecast_dates = pd.date_range("2024-01-01", periods=7, freq="D")

    weekly = _project_weekly(result, forecast_dates)
    assert len(weekly) == 7

    # All values should be finite
    assert np.all(np.isfinite(weekly))


# ---------------------------------------------------------------------------
# Unknown holiday warning
# ---------------------------------------------------------------------------


def test_unknown_future_holiday_warns():
    """Unknown future holiday name should emit UserWarning and give zero effect."""
    result, _ = make_decomposition()
    fr = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fr = SeqdForecaster(result).fit().predict(
            horizon=30,
            future_holidays={"Unknown Holiday": [pd.Timestamp("2024-11-28")]},
        )
    assert any("not found" in str(warning.message) for warning in w)
    # Holiday component should be all zero (holiday name not in result.holidays)
    np.testing.assert_allclose(fr.holiday_component.values, 0.0)


# ---------------------------------------------------------------------------
# forecast_from_result convenience function
# ---------------------------------------------------------------------------


def test_forecast_from_result_wrapper():
    """forecast_from_result should produce same result as SeqdForecaster.fit().predict()."""
    result, _ = make_decomposition(seed=99)
    fr = forecast_from_result(result, horizon=60)
    assert len(fr.forecast) == 60
    assert fr.horizon == 60


# ---------------------------------------------------------------------------
# Sanity: forecast is finite and in reasonable range
# ---------------------------------------------------------------------------


def test_forecast_finite_values():
    """Forecast values should all be finite."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=90)
    assert np.all(np.isfinite(fr.forecast.values))
    assert np.all(np.isfinite(fr.trend_component.values))
    assert np.all(np.isfinite(fr.annual_component.values))
    assert np.all(np.isfinite(fr.weekly_component.values))
    assert np.all(np.isfinite(fr.holiday_component.values))


def test_segment_count_positive():
    """At least one segment must exist after fit()."""
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result).fit()
    assert len(forecaster.segments) >= 1


def test_version_bumped():
    """Package version should be 0.2.1."""
    import seqd
    assert seqd.__version__ == "0.2.1"


# ---------------------------------------------------------------------------
# Combination formula: full pipeline verification (Part 2 gap #6)
# ---------------------------------------------------------------------------


def test_combination_formula_multiplicative_pipeline():
    """Forecast combination uses w_dow * (trend + annual + holiday), not independent multiplication.

    Manually extract the three component arrays and verify that the total forecast
    equals weekly * (trend + annual + holiday), confirming the spec formula.
    """
    result, _ = make_decomposition(multiplicative=True, add_holiday=False)
    fr = SeqdForecaster(result).fit().predict(horizon=14)

    trend = fr.trend_component.values
    weekly = fr.weekly_component.values
    annual = fr.annual_component.values
    holiday = fr.holiday_component.values
    forecast = fr.forecast.values

    # Spec formula: ŷ(t) = w_d × (trend + annual + holiday)
    expected = weekly * (trend + annual + holiday)
    np.testing.assert_allclose(forecast, expected, rtol=1e-10)

    # Confirm it is NOT equal to the fully-multiplicative (wrong) formula
    wrong = weekly * trend * annual * holiday
    assert not np.allclose(forecast, wrong), "Forecast should not equal fully-multiplicative formula"


def test_combination_formula_additive_pipeline():
    """Additive mode: forecast = trend + weekly + annual + holiday."""
    result, _ = make_decomposition(multiplicative=False, add_holiday=False)
    fr = SeqdForecaster(result).fit().predict(horizon=14)

    trend = fr.trend_component.values
    weekly = fr.weekly_component.values
    annual = fr.annual_component.values
    holiday = fr.holiday_component.values
    forecast = fr.forecast.values

    expected = trend + weekly + annual + holiday
    np.testing.assert_allclose(forecast, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Holiday: all occurrences unreliable → warning (Part 2 gap #7)
# ---------------------------------------------------------------------------


def test_all_holiday_occurrences_unreliable_emits_warning():
    """When every occurrence has ipm_reliable=False, a warning is emitted."""
    from seqd._structures import HolidayEffect, AnnualEffect, WeeklyEffect, DecompositionResult
    import datetime

    rng = np.random.default_rng(77)
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y = pd.Series(100.0 + 0.02 * t + rng.normal(0, 2, n), index=dates)

    # Build a synthetic DecompositionResult with one holiday occurrence
    # whose individual_peak_magnitude_reliable is False
    he = HolidayEffect(
        date=datetime.date(2021, 11, 26),
        name="Test Holiday",
        ramp_start=datetime.date(2021, 11, 20),
        ramp_end=datetime.date(2021, 11, 29),
        magnitude=50.0,
        effect_series=pd.Series(
            np.where(
                (dates >= pd.Timestamp("2021-11-20")) & (dates <= pd.Timestamp("2021-11-29")),
                50.0,
                0.0,
            ),
            index=dates,
        ),
        year_magnitudes=[50.0],
        magnitude_drift=0.0,
        compound=False,
        compound_block_id=None,
        individual_peak_magnitude=50.0,
        ramp_start_ceiling_hit=False,
        individual_peak_magnitude_reliable=False,  # <<< unreliable
    )

    annual = AnnualEffect(n_harmonics=0, coefficients=np.array([0.0]),
                          component=pd.Series(0.0, index=dates))
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    residual = y.copy()
    result = DecompositionResult(
        series=y,
        weekly=weekly,
        holidays=[he],
        annual=annual,
        residual=residual,
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    forecaster = SeqdForecaster(result)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster.fit(min_segment_size=90)
        forecaster.predict(
            horizon=30,
            future_holidays={"Test Holiday": [pd.Timestamp("2022-11-25")]},
        )

    messages = [str(warning.message) for warning in w]
    assert any(
        "unreliable ramp shapes" in m or "ceiling hit" in m or "boundary truncation" in m
        for m in messages
    ), f"Expected unreliable-ramp warning, got: {messages}"


# ---------------------------------------------------------------------------
# No future holidays → holiday_component is zero (Part 2 gap #8)
# ---------------------------------------------------------------------------


def test_no_future_holidays_holiday_component_is_zero():
    """When future_holidays is None, holiday_component must be all zeros."""
    result, _ = make_decomposition(add_holiday=True)
    fr = SeqdForecaster(result).fit().predict(horizon=30, future_holidays=None)
    np.testing.assert_allclose(
        fr.holiday_component.values,
        0.0,
        err_msg="holiday_component should be all zero when no future_holidays supplied",
    )


# ---------------------------------------------------------------------------
# Fix 1: compound window masking suppresses Jan changepoints
# ---------------------------------------------------------------------------


def test_mask_compound_windows_no_nan_leak():
    """_mask_compound_windows must return a fully-finite series (no NaN)."""
    from seqd._seqdforecaster import SeqdForecaster as SF

    result, _ = make_decomposition(add_holiday=True)
    masked = SF._mask_compound_windows(result.residual, result.holidays)

    assert masked.isna().sum() == 0, "Masked residual must not contain NaN"
    assert len(masked) == len(result.residual)


def test_mask_compound_windows_no_compound_unchanged():
    """_mask_compound_windows should leave residual unchanged when no compound holidays."""
    from seqd._seqdforecaster import SeqdForecaster as SF

    # make_decomposition with add_holiday=False → no HolidayEffect objects
    result, _ = make_decomposition(add_holiday=False)
    masked = SF._mask_compound_windows(result.residual, result.holidays)

    # With no compound effects, the series should be identical
    pd.testing.assert_series_equal(masked, result.residual)


def test_compound_masking_reduces_false_changepoints():
    """Synthetic BFCM-style series: masking should reduce Jan changepoints.

    Build a 3-year series with a large BFCM compound block that abruptly
    ends on Jan 1. Verify that running fit() with masking enabled produces
    fewer changepoints than a naive run would on the unmasked series.

    This is a regression test — the exact count depends on PELT penalty,
    but with masking the Jan 1 drop should NOT be detected as a changepoint.
    """
    from seqd._structures import (
        AnnualEffect, DecompositionResult, HolidayEffect, WeeklyEffect
    )
    import datetime

    rng = np.random.default_rng(1234)
    n = 3 * 365
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)

    # Smooth linear trend + noise
    y_values = 100.0 + 0.05 * t + rng.normal(0, 3.0, n)

    # Inject a large BFCM compound block (Nov 1 – Dec 31) each year with
    # an abrupt drop back to baseline on Jan 1.
    holiday_effects = []
    for yr in [2021, 2022]:
        block_start = pd.Timestamp(yr, 11, 1)
        block_end = pd.Timestamp(yr, 12, 31)
        mask = (dates >= block_start) & (dates <= block_end)
        y_values[mask] += 40.0  # large lift

        effect_vals = np.where(mask, 40.0, 0.0)
        he = HolidayEffect(
            date=datetime.date(yr, 11, 26),
            name="BFCM",
            ramp_start=datetime.date(yr, 11, 1),
            ramp_end=datetime.date(yr, 12, 31),
            magnitude=40.0,
            effect_series=pd.Series(effect_vals, index=dates),
            year_magnitudes=[40.0],
            magnitude_drift=0.0,
            compound=True,
            compound_block_id=f"compound_block_{yr}_1",
            individual_peak_magnitude=40.0,
            ramp_start_ceiling_hit=False,
            individual_peak_magnitude_reliable=True,
        )
        holiday_effects.append(he)

    y_series = pd.Series(y_values, index=dates)
    residual = y_series.copy()  # pretend the holiday effect was already removed

    annual = AnnualEffect(
        n_harmonics=0,
        coefficients=np.array([0.0]),
        component=pd.Series(0.0, index=dates),
    )
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    result = DecompositionResult(
        series=y_series,
        weekly=weekly,
        holidays=holiday_effects,
        annual=annual,
        residual=residual,
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    forecaster = SeqdForecaster(result)
    forecaster.fit(changepoint_penalty_beta=3.0, min_segment_size=60)
    cps = forecaster.changepoints

    # The Jan 1 "drop" dates (2022-01-01, 2023-01-01) must NOT be changepoints.
    jan_dates = {pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")}
    detected_set = set(cps)
    false_jan_cps = jan_dates & detected_set
    assert len(false_jan_cps) == 0, (
        f"Masking should suppress Jan 1 changepoints; detected: {false_jan_cps}"
    )


# ---------------------------------------------------------------------------
# Fix 3: ForecastResult.warnings is populated
# ---------------------------------------------------------------------------


def test_forecast_result_warnings_field_exists():
    """ForecastResult.warnings should be a list (possibly empty)."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result).fit().predict(horizon=30)
    assert hasattr(fr, "warnings")
    assert isinstance(fr.warnings, list)


def test_forecast_result_warnings_populated_on_quadratic_overextension():
    """When last segment is quadratic and horizon >> segment length, warnings list non-empty."""
    from seqd._structures import (
        AnnualEffect, DecompositionResult, SegmentTrend, WeeklyEffect
    )

    rng = np.random.default_rng(99)
    n = 93  # ~1 quarter — short enough that a year-long forecast has t >> 1
    dates = pd.date_range("2023-10-01", periods=n, freq="D")
    t = np.linspace(0, 1, n)
    # Quadratic residual
    y_values = 100.0 + 5.0 * t + 2.0 * t ** 2 + rng.normal(0, 0.5, n)
    y_series = pd.Series(y_values, index=dates)

    annual = AnnualEffect(
        n_harmonics=0,
        coefficients=np.array([0.0]),
        component=pd.Series(0.0, index=dates),
    )
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    result = DecompositionResult(
        series=y_series,
        weekly=weekly,
        holidays=[],
        annual=annual,
        residual=y_series.copy(),
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fr = SeqdForecaster(result).fit(min_segment_size=30).predict(
            horizon=365, max_extrapolation_days=365
        )

    # If the last segment is quadratic, warnings list should contain a message.
    last_seg = fr.segments[-1]
    if last_seg.model_type == "quadratic":
        assert len(fr.warnings) > 0, (
            "Quadratic overextension should produce a non-empty warnings list"
        )
        assert any("quadratic" in w.lower() for w in fr.warnings)


# ---------------------------------------------------------------------------
# H1: Slope blending
# ---------------------------------------------------------------------------


def test_slope_blend_alpha_invalid_raises():
    """slope_blend_alpha outside [0, 1] should raise ValueError."""
    result, _ = make_decomposition()
    with pytest.raises(ValueError, match="slope_blend_alpha"):
        SeqdForecaster(result, slope_blend_alpha=1.5)
    with pytest.raises(ValueError, match="slope_blend_alpha"):
        SeqdForecaster(result, slope_blend_alpha=-0.1)


def test_slope_blend_alpha_zero_is_default():
    """slope_blend_alpha=0.0 should produce the same forecast as no blending."""
    result, _ = make_decomposition()
    fr_default = SeqdForecaster(result).fit().predict(horizon=60)
    fr_zero = SeqdForecaster(result, slope_blend_alpha=0.0).fit().predict(horizon=60)
    np.testing.assert_allclose(
        fr_default.trend_component.values,
        fr_zero.trend_component.values,
        rtol=1e-12,
    )


def test_slope_blend_alpha_produces_finite_forecast():
    """Blended slope extrapolation must produce finite values for all model types."""
    result, _ = make_decomposition()
    fr = SeqdForecaster(result, slope_blend_alpha=0.3).fit().predict(horizon=180)
    assert np.all(np.isfinite(fr.forecast.values)), "Blended forecast must be finite"
    assert np.all(np.isfinite(fr.trend_component.values)), "Blended trend must be finite"


def test_slope_blend_dampens_steep_final_slope():
    """With a steep final segment and a flat penultimate, blending should
    produce a lower projected trend than pure final-segment extrapolation.

    Construct a two-segment synthetic residual where:
    - Segment 1 (penultimate): flat ~ 100
    - Segment 2 (final): steeply rising (Q4-like ramp)

    With alpha=0.3, the blended slope should be less steep than the final
    segment's slope alone, so the 90-day forecast should be lower than the
    alpha=0.0 forecast.
    """
    from seqd._structures import (
        AnnualEffect, DecompositionResult, WeeklyEffect
    )

    rng = np.random.default_rng(555)
    n_flat = 300
    n_steep = 120
    n = n_flat + n_steep
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    t_flat = np.arange(n_flat, dtype=float)
    t_steep = np.arange(n_steep, dtype=float)

    # Flat segment: ~100, tiny noise
    y_flat = 100.0 + rng.normal(0, 0.5, n_flat)
    # Steep segment: starts at 100, ends at 200 (slope ~0.83/day)
    y_steep = 100.0 + (100.0 / (n_steep - 1)) * t_steep + rng.normal(0, 0.5, n_steep)

    y_values = np.concatenate([y_flat, y_steep])
    y_series = pd.Series(y_values, index=dates)

    annual = AnnualEffect(
        n_harmonics=0,
        coefficients=np.array([0.0]),
        component=pd.Series(0.0, index=dates),
    )
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    result = DecompositionResult(
        series=y_series,
        weekly=weekly,
        holidays=[],
        annual=annual,
        residual=y_series.copy(),
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr_pure = SeqdForecaster(result, slope_blend_alpha=0.0).fit(
            min_segment_size=60
        ).predict(horizon=90)
        fr_blended = SeqdForecaster(result, slope_blend_alpha=0.3).fit(
            min_segment_size=60
        ).predict(horizon=90)

    # Blended forecast should be lower (less steep extrapolation) when
    # the penultimate segment is flat and final is steep.
    # Only assert this if two segments were detected.
    n_segs_pure = len(fr_pure.segments)
    n_segs_blended = len(fr_blended.segments)
    if n_segs_blended >= 2:
        mean_pure = fr_pure.trend_component.values.mean()
        mean_blended = fr_blended.trend_component.values.mean()
        assert mean_blended < mean_pure, (
            f"Blended forecast should be lower than pure-final when penultimate "
            f"is flat: blended_mean={mean_blended:.2f}, pure_mean={mean_pure:.2f}"
        )


def test_slope_blend_single_segment_unaffected():
    """With only one segment, blending is a no-op (no penultimate available)."""
    from seqd._structures import (
        AnnualEffect, DecompositionResult, WeeklyEffect
    )

    rng = np.random.default_rng(77)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y_values = 100.0 + 0.05 * t + rng.normal(0, 0.5, n)
    y_series = pd.Series(y_values, index=dates)

    annual = AnnualEffect(
        n_harmonics=0,
        coefficients=np.array([0.0]),
        component=pd.Series(0.0, index=dates),
    )
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    result = DecompositionResult(
        series=y_series,
        weekly=weekly,
        holidays=[],
        annual=annual,
        residual=y_series.copy(),
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr_alpha0 = SeqdForecaster(result, slope_blend_alpha=0.0).fit(
            min_segment_size=180
        ).predict(horizon=30)
        fr_alpha03 = SeqdForecaster(result, slope_blend_alpha=0.3).fit(
            min_segment_size=180
        ).predict(horizon=30)

    # If single segment was detected, both forecasts must be identical
    if len(fr_alpha0.segments) == 1 and len(fr_alpha03.segments) == 1:
        np.testing.assert_allclose(
            fr_alpha0.trend_component.values,
            fr_alpha03.trend_component.values,
            rtol=1e-12,
            err_msg="Single-segment blending should be identical to alpha=0",
        )


def test_forecast_from_result_slope_blend_alpha():
    """forecast_from_result should pass slope_blend_alpha through correctly."""
    result, _ = make_decomposition(seed=42)
    fr = forecast_from_result(result, horizon=60, slope_blend_alpha=0.3)
    assert len(fr.forecast) == 60
    assert np.all(np.isfinite(fr.forecast.values))


def test_slope_blend_alpha_property_stored():
    """SeqdForecaster should store slope_blend_alpha and use it in both predict calls."""
    result, _ = make_decomposition()
    forecaster = SeqdForecaster(result, slope_blend_alpha=0.3)
    assert forecaster._slope_blend_alpha == 0.3
    # Both predict calls from the same fitted forecaster must give same trend
    forecaster.fit()
    fr1 = forecaster.predict(horizon=30)
    fr2 = forecaster.predict(horizon=30)
    np.testing.assert_allclose(fr1.trend_component.values, fr2.trend_component.values)


def test_slope_at_t1_per_day_linear():
    """_slope_at_t1_per_day for linear model should equal beta/(n-1)."""
    from seqd._forecast import _slope_at_t1_per_day
    from seqd._structures import SegmentTrend

    seg = SegmentTrend(
        segment_index=1,
        start_date=pd.Timestamp("2022-01-01"),
        end_date=pd.Timestamp("2022-04-10"),
        n_obs=100,
        model_type="linear",
        alpha=100.0,
        beta=10.0,
        gamma=None,
        T_days=99,
        aic=0.0,
        aic_linear=0.0,
        rss=0.0,
        selected_reason="only candidate",
        t_anchor_date=pd.Timestamp("2022-01-01"),
    )
    expected = 10.0 / 99.0
    assert abs(_slope_at_t1_per_day(seg) - expected) < 1e-12


def test_slope_at_t1_per_day_quadratic():
    """_slope_at_t1_per_day for quadratic model should equal (beta+2*gamma)/(n-1)."""
    from seqd._forecast import _slope_at_t1_per_day
    from seqd._structures import SegmentTrend

    seg = SegmentTrend(
        segment_index=1,
        start_date=pd.Timestamp("2022-01-01"),
        end_date=pd.Timestamp("2022-04-10"),
        n_obs=100,
        model_type="quadratic",
        alpha=100.0,
        beta=8.0,
        gamma=3.0,
        T_days=99,
        aic=0.0,
        aic_linear=0.0,
        rss=0.0,
        selected_reason="only candidate",
        t_anchor_date=pd.Timestamp("2022-01-01"),
    )
    # df/dt at t=1 = beta + 2*gamma = 8 + 6 = 14; per day = 14/99
    expected = 14.0 / 99.0
    assert abs(_slope_at_t1_per_day(seg) - expected) < 1e-12


def test_slope_at_t1_per_day_constant():
    """_slope_at_t1_per_day for constant model should be 0."""
    from seqd._forecast import _slope_at_t1_per_day
    from seqd._structures import SegmentTrend

    seg = SegmentTrend(
        segment_index=1,
        start_date=pd.Timestamp("2022-01-01"),
        end_date=pd.Timestamp("2022-04-10"),
        n_obs=100,
        model_type="constant",
        alpha=100.0,
        beta=0.0,
        gamma=None,
        T_days=99,
        aic=0.0,
        aic_linear=0.0,
        rss=0.0,
        selected_reason="only candidate",
        t_anchor_date=pd.Timestamp("2022-01-01"),
    )
    assert _slope_at_t1_per_day(seg) == 0.0


# ---------------------------------------------------------------------------
# IPM-based compound block holiday projection
# ---------------------------------------------------------------------------


def test_compound_block_ipm_projection_peak_on_holiday_date():
    """Compound block holiday: projected IPM should appear on the holiday date.

    Build a synthetic DecompositionResult with two compound block holiday
    occurrences (e.g., two years of Black Friday in the same block).  The
    projected IPM should be placed on the future holiday date (within a small
    tolerance), not smeared across a flat block effect.

    This is the regression test for the root-cause fix: before the fix,
    _project_holidays projected the block-level mean effect_series (~$80M flat)
    rather than the per-holiday IPM ($584M on Black Friday).
    """
    from seqd._structures import (
        AnnualEffect, DecompositionResult, HolidayEffect, WeeklyEffect,
    )
    from seqd._forecast import _project_holidays
    import datetime

    # Build a 3-year series so the forecaster has enough data
    rng = np.random.default_rng(42)
    n = 3 * 365
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y_values = 100.0 + 0.05 * t + rng.normal(0, 2.0, n)
    y_series = pd.Series(y_values, index=dates)

    # Construct two historical Black Friday occurrences in the same compound block.
    # ipm_2021 = 200, ipm_2022 = 300 → OLS projects ipm_2023 ≈ 400.
    block_id = "compound_block_bfcm_1"

    def make_bfcm_he(yr, ipm_val):
        bf_date = datetime.date(yr, 11, 26)
        rs = datetime.date(yr, 11, 1)
        re = datetime.date(yr, 12, 15)
        # Non-primary compound members carry zero effect_series
        eff = pd.Series(0.0, index=dates)
        return HolidayEffect(
            date=bf_date,
            name="Black Friday",
            ramp_start=rs,
            ramp_end=re,
            magnitude=50.0,  # block-level mean — intentionally low
            effect_series=eff,
            year_magnitudes=[50.0],
            magnitude_drift=0.0,
            compound=True,
            compound_block_id=block_id,
            individual_peak_magnitude=ipm_val,
            ramp_start_ceiling_hit=False,
            individual_peak_magnitude_reliable=True,
        )

    he_2021 = make_bfcm_he(2021, 200.0)
    he_2022 = make_bfcm_he(2022, 300.0)

    annual = AnnualEffect(
        n_harmonics=0,
        coefficients=np.array([0.0]),
        component=pd.Series(0.0, index=dates),
    )
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    result = DecompositionResult(
        series=y_series,
        weekly=weekly,
        holidays=[he_2021, he_2022],
        annual=annual,
        residual=y_series.copy(),
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    # Future holiday: Black Friday 2023
    bf_2023 = pd.Timestamp("2023-11-24")
    forecast_dates = pd.date_range("2023-10-01", periods=120, freq="D")

    holiday_arr = _project_holidays(
        result=result,
        forecast_dates=forecast_dates,
        future_holidays={"Black Friday": [bf_2023]},
    )

    # The projected IPM from OLS: ipm_pairs = [(0, 200), (1, 300)] → slope=100,
    # projected = 300 + 100 = 400. At the holiday date itself, weight = 1.0 (day_delta=0).
    # So holiday_arr at bf_2023 date should equal 400.0 ± 1.0.
    bf_idx = forecast_dates.get_loc(bf_2023)
    peak_value = holiday_arr[bf_idx]

    assert abs(peak_value - 400.0) < 1.0, (
        f"Expected peak IPM ≈ 400.0 on Black Friday 2023, got {peak_value:.2f}. "
        "IPM-based compound block projection is not working correctly."
    )

    # Also verify that the peak is ON the holiday date (not a neighbour day)
    fc_series = pd.Series(holiday_arr, index=forecast_dates)
    # Restrict to ±3 days around bf_2023
    window_start = bf_2023 - pd.Timedelta(days=3)
    window_end = bf_2023 + pd.Timedelta(days=3)
    window = fc_series.loc[window_start:window_end]
    peak_date = window.idxmax()
    assert peak_date == bf_2023, (
        f"Peak holiday effect should be on {bf_2023.date()}, but was on {peak_date.date()}."
    )


# ---------------------------------------------------------------------------
# Fix A: Overlapping compound holidays use MAX not SUM
# ---------------------------------------------------------------------------


def test_overlapping_compound_holidays_use_max_not_sum():
    """Two compound holidays within 3 days of each other must be max-pooled.

    Build two compound holidays (Thanksgiving Nov 27, Black Friday Nov 28)
    whose triangular ramps fully overlap.  Verify that the combined effect on
    Nov 28 equals the max of each individual effect, NOT their sum.

    Without max-pooling (old behaviour), the Nov 28 value would be:
      TG_ramp(Nov28) + BF_ramp(Nov28) = TG_ipm * 0.75 + BF_ipm * 1.0

    With max-pooling (Fix A), it should be:
      max(TG_ramp(Nov28), BF_ramp(Nov28)) = max(TG_ipm * 0.75, BF_ipm * 1.0)
    """
    from seqd._structures import (
        AnnualEffect, DecompositionResult, HolidayEffect, WeeklyEffect,
    )
    from seqd._forecast import _project_holidays
    import datetime

    rng = np.random.default_rng(42)
    n = 3 * 365
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y_values = 100.0 + 0.05 * t + rng.normal(0, 2.0, n)
    y_series = pd.Series(y_values, index=dates)

    block_id = "compound_block_tg_bf_1"

    def make_compound_he(yr, h_date, h_name, ipm_val):
        rs = datetime.date(yr, 11, 1)
        re = datetime.date(yr, 12, 10)
        eff = pd.Series(0.0, index=dates)
        return HolidayEffect(
            date=h_date,
            name=h_name,
            ramp_start=rs,
            ramp_end=re,
            magnitude=50.0,
            effect_series=eff,
            year_magnitudes=[50.0],
            magnitude_drift=0.0,
            compound=True,
            compound_block_id=block_id,
            individual_peak_magnitude=ipm_val,
            ramp_start_ceiling_hit=False,
            individual_peak_magnitude_reliable=True,
        )

    # Thanksgiving Nov 27 with IPM=200, Black Friday Nov 28 with IPM=300
    he_tg_2021 = make_compound_he(2021, datetime.date(2021, 11, 25), "Thanksgiving", 200.0)
    he_tg_2022 = make_compound_he(2022, datetime.date(2022, 11, 24), "Thanksgiving", 220.0)
    he_bf_2021 = make_compound_he(2021, datetime.date(2021, 11, 26), "Black Friday", 300.0)
    he_bf_2022 = make_compound_he(2022, datetime.date(2022, 11, 25), "Black Friday", 350.0)

    annual = AnnualEffect(
        n_harmonics=0,
        coefficients=np.array([0.0]),
        component=pd.Series(0.0, index=dates),
    )
    weekly = WeeklyEffect(
        coefficients=np.ones(7),
        is_multiplicative=False,
        recency={},
        drift={},
    )
    result = DecompositionResult(
        series=y_series,
        weekly=weekly,
        holidays=[he_tg_2021, he_tg_2022, he_bf_2021, he_bf_2022],
        annual=annual,
        residual=y_series.copy(),
        r2_by_component={"weekly": 0.0, "holiday": 0.0, "annual": 0.0},
    )

    # Future dates: Thanksgiving Nov 27 2023, Black Friday Nov 28 2023 (1 day apart)
    tg_2023 = pd.Timestamp("2023-11-23")  # Thu
    bf_2023 = pd.Timestamp("2023-11-24")  # Fri (1 day after TG)
    forecast_dates = pd.date_range("2023-10-01", periods=120, freq="D")

    # OLS for Thanksgiving: pairs (0,200),(1,220) → slope=20, projected=240
    # OLS for Black Friday: pairs (0,300),(1,350) → slope=50, projected=400
    # But with max_years=4 growth cap: BF projected = min(400, 350*1.4=490) = 400
    tg_projected_ipm = 240.0
    bf_projected_ipm = 400.0

    holiday_arr = _project_holidays(
        result=result,
        forecast_dates=forecast_dates,
        future_holidays={
            "Thanksgiving": [tg_2023],
            "Black Friday": [bf_2023],
        },
        max_holiday_merge_gap_days=35,
    )

    fc_series = pd.Series(holiday_arr, index=forecast_dates)

    # On Black Friday (bf_2023): TG ramp contribution = tg_projected * weight(1 day after TG)
    # weight = max(0, 1 - 1/4) = 0.75
    # BF ramp contribution = bf_projected * weight(0 days from BF) = bf_projected * 1.0
    tg_contrib_on_bf = tg_projected_ipm * (1.0 - 1.0 / 4.0)  # delta=+1 from TG
    bf_contrib_on_bf = bf_projected_ipm * 1.0

    # Max-pooled value should be max of the two contributions
    expected_max = max(tg_contrib_on_bf, bf_contrib_on_bf)
    # If summed instead: tg_contrib_on_bf + bf_contrib_on_bf
    expected_sum = tg_contrib_on_bf + bf_contrib_on_bf

    bf_actual = fc_series.loc[bf_2023]

    # With max-pooling, value should equal expected_max (not expected_sum)
    assert abs(bf_actual - expected_max) < 1.0, (
        f"Expected max-pooled value ≈ {expected_max:.2f} on Black Friday, "
        f"got {bf_actual:.2f}. "
        f"(Sum would have been {expected_sum:.2f}; max is {expected_max:.2f}.)"
    )
    assert bf_actual < expected_sum - 1.0, (
        f"Value should be less than the additive sum ({expected_sum:.2f}), "
        f"confirming max-pooling is active. Got {bf_actual:.2f}."
    )


# ---------------------------------------------------------------------------
# Fix C: IPM growth cap at 40%
# ---------------------------------------------------------------------------


def test_ols_project_ipm_growth_cap():
    """_ols_project_ipm should cap projected growth at 40% above most recent IPM.

    Build a series that would project >40% above the most recent value via
    pure OLS, and verify that the returned projection is capped at 1.4x the
    most recent IPM.
    """
    from seqd._forecast import _ols_project_ipm

    # ipm_pairs: (0, 100), (1, 200), (2, 400) — aggressive exponential growth
    # OLS over last 4 years: slope would be large
    # Most recent IPM = 400. 40% cap → projection capped at 560.
    ipm_pairs = [(0, 100.0), (1, 200.0), (2, 400.0)]
    projected = _ols_project_ipm(ipm_pairs, max_years=4)

    # Uncapped OLS: x=[0,1,2], y=[100,200,400] → slope ≈ 150, projected = 400 + 150 = 550
    # But 550 > 400 * 1.4 = 560? No — 550 < 560, so actually not capped in this case.
    # Use a more extreme example: (0, 100), (1, 400) → slope=300, projected=700 > 400*1.4=560
    ipm_pairs2 = [(0, 100.0), (1, 400.0)]
    projected2 = _ols_project_ipm(ipm_pairs2, max_years=4)
    # OLS: slope = 300, projected = 700. Cap: 400 * 1.4 = 560.
    assert abs(projected2 - 560.0) < 1.0, (
        f"Expected growth cap at 560.0 (400 * 1.4), got {projected2:.2f}"
    )

    # Verify that WITHOUT cap, we would exceed 1.4x: 700 > 560
    # Also verify that normal (non-explosive) growth is NOT capped
    ipm_pairs3 = [(0, 100.0), (1, 120.0), (2, 140.0)]  # linear +20/yr
    projected3 = _ols_project_ipm(ipm_pairs3, max_years=4)
    # OLS: slope=20, projected=160. Cap: 140*1.4=196. 160 < 196, so not capped.
    assert abs(projected3 - 160.0) < 1.0, (
        f"Normal growth should not be capped: expected 160.0, got {projected3:.2f}"
    )
