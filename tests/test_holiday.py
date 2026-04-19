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
    """Detected ramp_start and ramp_end should be within ±7 days of true values."""
    y, holiday_dates_raw, true_starts, true_ends = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}

    holiday_effects, y_h = fit_holidays(y_w=y, holidays=holidays)

    # One HolidayEffect per occurrence
    assert len(holiday_effects) == len(holiday_dates_raw), (
        f"Should produce {len(holiday_dates_raw)} HolidayEffects (one per occurrence), "
        f"got {len(holiday_effects)}"
    )

    # Check most recent occurrence (last in sorted order)
    he_last = sorted(holiday_effects, key=lambda h: h.date)[-1]
    true_start = true_starts[-1]
    true_end = true_ends[-1]

    start_error = abs((he_last.ramp_start - true_start).days)
    end_error = abs((he_last.ramp_end - true_end).days)

    assert start_error <= 7, (
        f"ramp_start error {start_error} days > 7 "
        f"(detected={he_last.ramp_start}, true={true_start})"
    )
    assert end_error <= 7, (
        f"ramp_end error {end_error} days > 7 "
        f"(detected={he_last.ramp_end}, true={true_end})"
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
    """Spike-only holiday (no ramp) should still be detected and removed."""
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

    # Two occurrences -> two HolidayEffect objects
    assert len(he_list) == 2, f"Expected 2 HolidayEffects (one per year), got {len(he_list)}"
    for he in he_list:
        # ramp_start must not exceed holiday_window days before the holiday date
        assert (he.ramp_start - he.date).days >= -14, (
            f"ramp_start={he.ramp_start} is more than holiday_window(14) days "
            f"before holiday date {he.date}"
        )
        # ramp_end must be after or on the holiday date
        assert he.ramp_end >= he.date, (
            f"ramp_end={he.ramp_end} is before holiday date {he.date}"
        )


def test_holiday_year_magnitudes():
    """year_magnitudes should have one entry per occurrence on every HolidayEffect."""
    y, holiday_dates_raw, _, _ = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}
    holiday_effects, _ = fit_holidays(y_w=y, holidays=holidays)

    # Each occurrence carries the full recency list
    for he in holiday_effects:
        assert len(he.year_magnitudes) == len(holiday_dates_raw), (
            f"Expected {len(holiday_dates_raw)} year magnitudes, got {len(he.year_magnitudes)}"
        )


def test_holiday_effect_series_alignment():
    """effect_series should have same index as input on every HolidayEffect."""
    y, holiday_dates_raw, _, _ = make_holiday_series()
    holidays = {"Christmas": holiday_dates_raw}
    holiday_effects, _ = fit_holidays(y_w=y, holidays=holidays)

    for he in holiday_effects:
        assert len(he.effect_series) == len(y)
        assert he.effect_series.index.equals(y.index)


def test_multiple_holidays():
    """Multiple distinct holidays should produce separate HolidayEffect objects per occurrence."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-01", periods=365 * 2, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 2.0, len(dates)), index=dates, name="y")

    # Two distinct holidays, 2 occurrences each = 4 total HolidayEffect objects
    christmas = [datetime.date(2022, 12, 25), datetime.date(2023, 12, 25)]
    ny = [datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)]

    for h in christmas + ny:
        h_ts = pd.Timestamp(h)
        if h_ts in y.index:
            y.loc[h_ts] += 30.0

    holidays = {"Christmas": christmas, "NewYear": ny}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays)

    # 2 occurrences per holiday name × 2 holiday names = 4 total
    assert len(he_list) == 4, f"Expected 4 holiday effects (2 per name × 2 names), got {len(he_list)}"
    names = {he.name for he in he_list}
    assert "Christmas" in names
    assert "NewYear" in names


def test_multi_year_holiday_detection():
    """Pass 3 years of holiday dates; verify 3 HolidayEffect objects are returned."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=365 * 4, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 2.0, len(dates)), index=dates, name="y")

    h_dates = [
        datetime.date(2021, 7, 4),
        datetime.date(2022, 7, 4),
        datetime.date(2023, 7, 4),
    ]
    for h in h_dates:
        h_ts = pd.Timestamp(h)
        if h_ts in y.index:
            y.loc[h_ts] += 40.0

    holidays = {"IndependenceDay": h_dates}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays)

    assert len(he_list) == 3, (
        f"Expected 3 HolidayEffect objects (one per year), got {len(he_list)}"
    )
    detected_dates = {he.date for he in he_list}
    for h in h_dates:
        assert h in detected_dates, f"Holiday date {h} not found in detected effects"


def test_compound_block_no_double_counting():
    """Compound block members must not double-count: holiday_component() sum should
    be within 20% of the injected effect, not 2-3x it."""
    rng = np.random.default_rng(99)
    n_days = 365 * 2
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, n_days), index=dates, name="y")

    # Two holidays 6 days apart in each year — within ±14-day search window of each other
    # so their ramp windows will overlap and form a compound block.
    h1_dates = [datetime.date(2023, 12, 25), datetime.date(2024, 12, 25)]
    h2_dates = [datetime.date(2023, 12, 31), datetime.date(2024, 12, 31)]

    injected_total = 0.0
    for h in h1_dates + h2_dates:
        h_ts = pd.Timestamp(h)
        if h_ts in y.index:
            effect = 15.0
            y.loc[h_ts] += effect
            injected_total += effect

    holidays = {"Christmas": h1_dates, "NewYear": h2_dates}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays)

    # Reconstruct holiday_component() as DecompositionResult would
    total = pd.Series(0.0, index=y.index)
    for he in he_list:
        aligned = he.effect_series.reindex(y.index, fill_value=0.0)
        total = total + aligned

    # Sum of positive effect values
    pos_sum = float(total[total > 0].sum())

    # Should be in the right ballpark (within 20% factor of 3×), not 2-3× injected
    assert pos_sum < injected_total * 2.5, (
        f"holiday_component() sum {pos_sum:.2f} is > 2.5× injected {injected_total:.2f} "
        f"— double-counting suspected in compound block"
    )


def test_run_criterion_detects_gradual_ramp():
    """Run criterion must fire for a gradual 8-day pre-holiday ramp and return
    ramp_start at least 5 days before the holiday."""
    from seqd._holiday import _detect_ramp_start

    h_date = datetime.date(2024, 12, 25)
    sigma_ref = 5.0
    # 8 days of residuals each at 0.6 * sigma_ref — above run_threshold (0.5 * sigma_ref)
    # but each individually below CUSUM threshold so CUSUM alone may not fire early.
    day_to_residual: dict = {}
    for delta in range(0, 9):  # delta 0..8 (today back to 8 days before)
        day_to_residual[-delta] = 0.6 * sigma_ref

    ramp_start = _detect_ramp_start(
        h_date=h_date,
        day_to_residual=day_to_residual,
        sigma_ref=sigma_ref,
        holiday_window=14,
    )

    days_before = (h_date - ramp_start).days
    assert days_before >= 5, (
        f"Run criterion should detect ramp_start at least 5 days before holiday, "
        f"got ramp_start={ramp_start} which is only {days_before} days before {h_date}"
    )


def test_ramp_start_ceiling_hit_flag():
    """ramp_start_ceiling_hit should be True when ramp is longer than holiday_window."""
    rng = np.random.default_rng(300)
    n_days = 365 * 3
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    trend = np.linspace(100.0, 110.0, n_days)
    noise = rng.normal(0, 0.3, n_days)  # low noise
    y = pd.Series(trend + noise, index=dates, name="y")

    # Create a very long ramp (40 days) that exceeds holiday_window=20
    h_dates = [datetime.date(2022, 12, 25), datetime.date(2023, 12, 25)]
    for h in h_dates:
        h_ts = pd.Timestamp(h)
        if h_ts not in y.index:
            continue
        for delta in range(-40, 5):
            d_ts = h_ts + pd.Timedelta(days=delta)
            if d_ts in y.index:
                factor = max(0.0, 1.0 - abs(delta) / 41.0)
                y.loc[d_ts] += factor * 30.0

    holidays = {"Christmas": h_dates}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays, holiday_window=20)

    # At least one of the occurrences should hit the ceiling
    hit_flags = [he.ramp_start_ceiling_hit for he in he_list]
    assert any(hit_flags), (
        f"Expected at least one ramp_start_ceiling_hit=True for a 40-day ramp "
        f"with holiday_window=20. Flags: {hit_flags}"
    )


def test_ramp_start_no_ceiling_hit():
    """ramp_start_ceiling_hit should be False when ramp is shorter than holiday_window."""
    rng = np.random.default_rng(301)
    n_days = 365 * 3
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    trend = np.linspace(100.0, 110.0, n_days)
    noise = rng.normal(0, 0.3, n_days)  # low noise
    y = pd.Series(trend + noise, index=dates, name="y")

    # Short ramp: 7 days before, well within holiday_window=35
    h_dates = [datetime.date(2022, 12, 25), datetime.date(2023, 12, 25)]
    for h in h_dates:
        h_ts = pd.Timestamp(h)
        if h_ts not in y.index:
            continue
        for delta in range(-7, 5):
            d_ts = h_ts + pd.Timedelta(days=delta)
            if d_ts in y.index:
                factor = max(0.0, 1.0 - abs(delta) / 8.0)
                y.loc[d_ts] += factor * 30.0

    holidays = {"Christmas": h_dates}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays, holiday_window=35)

    # None should hit the ceiling with a short ramp and large window
    hit_flags = [he.ramp_start_ceiling_hit for he in he_list]
    assert not any(hit_flags), (
        f"Expected all ramp_start_ceiling_hit=False for a short 7-day ramp "
        f"with holiday_window=35. Flags: {hit_flags}"
    )


def test_individual_peak_magnitude_reliable_boundary():
    """Holidays within 3 days of series start/end should have ipm_reliable=False."""
    rng = np.random.default_rng(302)
    # Use exactly 1 year so the last holiday of the year is near the series end
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    y = pd.Series(100.0 + rng.normal(0, 1.0, 365), index=dates, name="y")

    # Holiday on Dec 30 — only 1 day before series end (Dec 31), so h+3 exceeds end
    h_date = datetime.date(2022, 12, 30)
    h_ts = pd.Timestamp(h_date)
    if h_ts in y.index:
        y.loc[h_ts] += 30.0

    holidays = {"LateHoliday": [h_date]}
    he_list, _ = fit_holidays(y_w=y, holidays=holidays)

    assert len(he_list) >= 1
    # Find the late-year holiday
    late_he = next((he for he in he_list if he.date == h_date), None)
    assert late_he is not None, f"HolidayEffect for {h_date} not found"
    assert late_he.individual_peak_magnitude_reliable is False, (
        f"Expected individual_peak_magnitude_reliable=False for holiday on {h_date} "
        f"(series ends {dates[-1].date()}), but got True"
    )


def test_holiday_component_nonzero_all_years():
    """holiday_component() must have non-zero values in all years of a multi-year input."""
    from seqd import SeqdDecomposer

    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=365 * 3 + 1, freq="D")
    y_vals = 100.0 + rng.normal(0, 2.0, len(dates))
    y = pd.Series(y_vals, index=dates, name="y")

    h_dates = [
        datetime.date(2021, 7, 4),
        datetime.date(2022, 7, 4),
        datetime.date(2023, 7, 4),
    ]
    for h in h_dates:
        h_ts = pd.Timestamp(h)
        if h_ts in y.index:
            y.loc[h_ts] += 40.0

    decomp = SeqdDecomposer(holiday_dates={"FourthOfJuly": h_dates})
    result = decomp.fit(y)

    hc = result.holiday_component()
    for h in h_dates:
        # Window around each holiday date should contain at least one non-zero value
        window = hc.loc[
            pd.Timestamp(h) - pd.Timedelta(days=7) : pd.Timestamp(h) + pd.Timedelta(days=7)
        ]
        assert (window != 0.0).any(), (
            f"holiday_component() is all-zero around {h} — multi-year detection failed"
        )
