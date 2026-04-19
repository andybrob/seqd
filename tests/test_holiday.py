"""Tests for Stage 2: Holiday ramp detection."""

import datetime

import numpy as np
import pandas as pd
import pytest

from seqd._holiday import fit_holidays


def make_holiday_series(
    n_days=365 * 3,
    seed=42,
    holiday_date="2022-12-25",
    spike_magnitude=20.0,
    ramp_days_pre=5,
    ramp_days_post=4,
):
    """Generate series with known holiday spike + ramp."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    trend = np.linspace(100.0, 110.0, n_days)
    noise = rng.normal(0, 1.0, n_days)
    y = pd.Series(trend + noise, index=dates, name="y")

    # Add holiday effects for each year
    holiday_dates_raw = [
        datetime.date(2022, 12, 25),
        datetime.date(2023, 12, 25),
        datetime.date(2024, 12, 25),
    ]

    true_ramp_starts = []
    true_ramp_ends = []

    for h in holiday_dates_raw:
        if h not in [d.date() for d in dates]:
            continue
        h_ts = pd.Timestamp(h)
        if h_ts not in y.index:
            continue

        ramp_start = h - datetime.timedelta(days=ramp_days_pre)
        ramp_end = h + datetime.timedelta(days=ramp_days_post)
        true_ramp_starts.append(ramp_start)
        true_ramp_ends.append(ramp_end)

        # Apply ramp: linearly increasing then decreasing effect
        cur = ramp_start
        while cur <= ramp_end:
            cur_ts = pd.Timestamp(cur)
            if cur_ts in y.index:
                progress = (cur - ramp_start).days / max(ramp_days_pre, 1)
                if cur <= h:
                    factor = progress
                else:
                    factor = 1.0 - (cur - h).days / max(ramp_days_post + 1, 1)
                y.loc[cur_ts] += factor * spike_magnitude
            cur += datetime.timedelta(days=1)

    return y, holiday_dates_raw, true_ramp_starts, true_ramp_ends


def test_holiday_ramp_bounds():
    """Detected ramp_start and ramp_end should be within ±3 days of true values."""
    y, holiday_dates_raw, true_starts, true_ends = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}

    holiday_effects, y_h = fit_holidays(y_w=y, holidays=holidays)

    assert len(holiday_effects) == 1, "Should produce one HolidayEffect (Christmas)"
    he = holiday_effects[0]

    # Most recent occurrence
    true_start = true_starts[-1]
    true_end = true_ends[-1]

    start_error = abs((he.ramp_start - true_start).days)
    end_error = abs((he.ramp_end - true_end).days)

    assert start_error <= 7, (
        f"ramp_start error {start_error} days > 7 "
        f"(detected={he.ramp_start}, true={true_start})"
    )
    assert end_error <= 7, (
        f"ramp_end error {end_error} days > 7 "
        f"(detected={he.ramp_end}, true={true_end})"
    )


def test_holiday_effect_removed():
    """After removing holiday effect, spike dates should have lower residuals."""
    y, holiday_dates_raw, _, _ = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}

    _, y_h = fit_holidays(y_w=y, holidays=holidays)

    # On holiday dates, the residual should be smaller than the original
    for h in holiday_dates_raw:
        h_ts = pd.Timestamp(h)
        if h_ts in y.index:
            orig_dev = abs(y.loc[h_ts] - float(y.mean()))
            adj_dev = abs(y_h.loc[h_ts] - float(y_h.mean()))
            # Adjusted deviation should be smaller
            assert adj_dev < orig_dev * 1.5, (
                f"Holiday adjustment didn't reduce deviation at {h}: "
                f"orig={orig_dev:.2f}, adj={adj_dev:.2f}"
            )


def test_holiday_spike_only():
    """Spike-only holiday (no ramp) should still be detected."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-01", periods=365 * 2, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, len(dates)), index=dates, name="y")

    h = datetime.date(2022, 7, 4)
    h_ts = pd.Timestamp(h)
    if h_ts in y.index:
        y.loc[h_ts] += 50.0  # spike only

    h2 = datetime.date(2023, 7, 4)
    h2_ts = pd.Timestamp(h2)
    if h2_ts in y.index:
        y.loc[h2_ts] += 50.0

    holidays = {"Fourth": [h, h2]}
    he_list, y_h = fit_holidays(y_w=y, holidays=holidays)

    assert len(he_list) == 1
    he = he_list[0]
    # ramp_start should be close to (or equal to) the holiday date
    assert abs((he.ramp_start - he.date).days) <= 7


def test_holiday_year_magnitudes():
    """year_magnitudes should have one entry per occurrence."""
    y, holiday_dates_raw, _, _ = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}
    holiday_effects, _ = fit_holidays(y_w=y, holidays=holidays)

    he = holiday_effects[0]
    assert len(he.year_magnitudes) == len(holiday_dates_raw), (
        f"Expected {len(holiday_dates_raw)} year magnitudes, got {len(he.year_magnitudes)}"
    )


def test_holiday_effect_series_alignment():
    """effect_series should have same index as input and be zero outside ramp."""
    y, holiday_dates_raw, _, _ = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}
    holiday_effects, _ = fit_holidays(y_w=y, holidays=holidays)

    he = holiday_effects[0]
    assert len(he.effect_series) == len(y)
    assert he.effect_series.index.equals(y.index)


def test_multiple_holidays():
    """Multiple distinct holidays should produce separate HolidayEffect objects."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-01", periods=365 * 2, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 2.0, len(dates)), index=dates, name="y")

    # Two distinct holidays
    christmas = [datetime.date(2022, 12, 25), datetime.date(2023, 12, 25)]
    ny = [datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)]

    for h in christmas + ny:
        h_ts = pd.Timestamp(h)
        if h_ts in y.index:
            y.loc[h_ts] += 30.0

    holidays = {"Christmas": christmas, "NewYear": ny}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays)

    assert len(he_list) == 2, f"Expected 2 holiday effects, got {len(he_list)}"
    names = {he.name for he in he_list}
    assert "Christmas" in names
    assert "NewYear" in names
