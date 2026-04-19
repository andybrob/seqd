# seqd

**Sequential time series decomposition** — cleanly extracts weekly, holiday (with data-driven ramp detection), and annual effects from a daily time series, leaving only trend and noise.

## Overview

`seqd` applies a three-stage sequential pipeline to daily time series:

1. **Weekly** — Robust day-of-week effect estimation (10% trimmed mean, holiday-masked)
2. **Holiday** — Data-driven ramp detection using CUSUM; finds when effects start and when the series returns to baseline
3. **Annual** — BIC-selected Fourier regression (up to 4 harmonics)

The residual after all three stages is trend + noise. All extracted components are recoverable.

## Installation

```bash
pip install seqd
```

Or from source:

```bash
git clone https://github.com/andybrob/seqd.git
cd seqd
pip install -e .
```

## Quick Start

```python
import pandas as pd
from seqd import SeqdDecomposer

# Your daily series
y = pd.Series(...)  # DatetimeIndex required

# Define holidays (any combination of strings, datetime.date, pd.Timestamp)
holidays = {
    "Christmas": ["2022-12-25", "2023-12-25", "2024-12-25"],
    "NewYear":   ["2022-01-01", "2023-01-01", "2024-01-01"],
}

decomp = SeqdDecomposer(holiday_dates=holidays)
result = decomp.fit(y)

# Access components
result.residual              # trend + noise
result.weekly_component()    # DOW effects
result.holiday_component()   # holiday ramp effects
result.annual_component()    # Fourier annual seasonality
result.fitted()              # reconstructed series (residual + all components)

# Diagnostics
print(result.r2_by_component)
# {'weekly': 0.12, 'holiday': 0.03, 'annual': 0.28}

# Weekly drift detection
for dow, info in result.weekly.drift.items():
    print(f"DOW {dow}: {info['classification']} (slope={info['slope']:.4f})")

# Holiday magnitude drift
for he in result.holidays:
    print(f"{he.name}: magnitude drift = {he.magnitude_drift:.2f}/year")
```

## API Reference

### `SeqdDecomposer`

```python
SeqdDecomposer(
    holiday_dates,         # list or dict — see below
    multiplicative=None,   # None = auto-detect; True/False = force
    holiday_window=14,     # ±days around holiday for ramp search
    reference_window=60,   # days for baseline estimation
)
```

**`holiday_dates`** accepts:

| Format | Example |
|--------|---------|
| Flat list of strings | `["2023-12-25", "2024-12-25"]` |
| Flat list of `datetime.date` | `[date(2023, 12, 25)]` |
| Flat list of `pd.Timestamp` | `[pd.Timestamp("2023-12-25")]` |
| Dict (named holidays) | `{"Christmas": ["2023-12-25", "2024-12-25"]}` |

**`.fit(y)`** — accepts a `pd.Series` with `DatetimeIndex` (daily, no duplicate dates).

### `DecompositionResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `series` | `pd.Series` | Original input |
| `weekly` | `WeeklyEffect` | Stage 1 result |
| `holidays` | `list[HolidayEffect]` | Stage 2 results (one per named holiday) |
| `annual` | `AnnualEffect` | Stage 3 result |
| `residual` | `pd.Series` | Trend + noise |
| `r2_by_component` | `dict` | `{"weekly": float, "holiday": float, "annual": float}` |
| `.weekly_component()` | `pd.Series` | Full weekly effect series |
| `.holiday_component()` | `pd.Series` | Sum of all holiday effects |
| `.annual_component()` | `pd.Series` | Annual Fourier component |
| `.fitted()` | `pd.Series` | Reconstructed series |

### `WeeklyEffect`

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `np.ndarray` | Shape (7,), Monday=0 |
| `is_multiplicative` | `bool` | Mode used |
| `recency` | `dict` | `{window_days: DataFrame}` — trailing window estimates |
| `drift` | `dict` | `{dow: {"slope": float, "classification": "stable"\|"drifting"}}` |

### `HolidayEffect`

| Attribute | Type | Description |
|-----------|------|-------------|
| `date` | `datetime.date` | Most recent holiday date |
| `name` | `str` | Holiday name |
| `ramp_start` | `datetime.date` | Detected effect start |
| `ramp_end` | `datetime.date` | Detected effect end |
| `magnitude` | `float` | Mean residual over ramp window |
| `effect_series` | `pd.Series` | Full-length effect (zeros outside ramp) |
| `year_magnitudes` | `list[float]` | Per-occurrence magnitudes |
| `magnitude_drift` | `float` | OLS slope of magnitudes over years |

### `AnnualEffect`

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_harmonics` | `int` | BIC-selected K |
| `coefficients` | `np.ndarray` | `[a0, a1, b1, ...]` |
| `component` | `pd.Series` | Fitted annual component (no intercept) |
| `recency_amplitudes` | `dict` | `{1: amp, 2: amp, 3: amp}` — trailing year amplitudes |

## Design Notes

**Multiplicative vs additive**: Auto-detected by comparing `std(detrended) / mean(y)` to 0.15. Can be forced.

**Holiday ramp detection**: Uses backward CUSUM to find when the effect begins (up to `holiday_window` days before the holiday) and a forward baseline-return scan to find when it ends (returns within 1.5σ for 2 consecutive days). If no recovery is detected within `holiday_window` days after the holiday, the ramp is considered to persist through the full search window.

**Annual seasonality**: Fourier regression with BIC selection over K ∈ {0, 1, 2, 3, 4}. K=0 (no annual component) is the baseline model, so a series with no meaningful annual seasonality will not have a spurious harmonic extracted. BIC is computed manually (no statsmodels dependency).

**`weekly_component()` in multiplicative mode**: Returns `original * (1 - 1/coeff)` — the absolute amount removed at each date. This scales with the level of the series. For a level-independent view of the weekly pattern, use `weekly.coefficients` directly (the 7-element array of multiplicative factors).

**Drift detection**: Weekly DOW coefficients are re-estimated on trailing windows ([60, 90, 365] days, stepped every 7 days). The OLS slope over time classifies each DOW as "stable" or "drifting" (`|slope| > 1% of full-sample coefficient`).

## Dependencies

- `numpy >= 1.24`
- `pandas >= 2.0`
- `scipy >= 1.10`

No statsmodels. No ruptures.

## License

MIT
