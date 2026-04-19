"""Data structures for seqd decomposition results."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WeeklyEffect:
    """Weekly day-of-week effect extracted in Stage 1.

    Attributes
    ----------
    coefficients : np.ndarray
        Shape (7,). Additive offsets or multiplicative factors per DOW (Monday=0).
    is_multiplicative : bool
        Whether multiplicative mode was used.
    recency : dict
        Mapping from window_days (int) to a DataFrame with columns
        [date, dow_0, dow_1, ..., dow_6]. Each row corresponds to a trailing
        window endpoint (stepped every 7 days).
    drift : dict
        Mapping from dow (0-6) to {"slope": float, "classification": str}.
        classification is "stable" or "drifting".
    """

    coefficients: np.ndarray
    is_multiplicative: bool
    recency: Dict[int, pd.DataFrame] = field(default_factory=dict)
    drift: Dict[int, Dict] = field(default_factory=dict)


@dataclass
class HolidayEffect:
    """Holiday effect (including ramp) for a single holiday occurrence.

    Attributes
    ----------
    date : datetime.date
        The holiday date.
    name : str
        Holiday name.
    ramp_start : datetime.date
        First day of the detected ramp/effect window.
    ramp_end : datetime.date
        Last day of the detected ramp/effect window.
    magnitude : float
        Mean residual over the ramp window.
    effect_series : pd.Series
        Full-length series aligned to original index; zeros outside ramp window.
    year_magnitudes : list
        Magnitude for each historical occurrence of this holiday (in year order).
    magnitude_drift : float
        OLS slope of magnitudes over year indices.
    compound : bool
        True if this holiday was merged into a compound block with other holidays.
    compound_block_id : str or None
        Identifier for the compound block (e.g. "compound_block_2025_1"), shared
        across all constituent holidays merged into the same block.
    individual_peak_magnitude : float or None
        Mean residual within ±3 days around this specific holiday date (estimated
        before the compound block merge). Allows distinguishing which day within
        a merged block drove the largest effect.
    """

    date: datetime.date
    name: str
    ramp_start: datetime.date
    ramp_end: datetime.date
    magnitude: float
    effect_series: pd.Series
    year_magnitudes: List[float] = field(default_factory=list)
    magnitude_drift: float = 0.0
    compound: bool = False
    compound_block_id: Optional[str] = None
    individual_peak_magnitude: Optional[float] = None
    ramp_start_ceiling_hit: bool = False
    individual_peak_magnitude_reliable: bool = True


@dataclass
class AnnualEffect:
    """Annual Fourier seasonality extracted in Stage 3.

    Attributes
    ----------
    n_harmonics : int
        Number of Fourier harmonics selected by BIC. K ∈ {0,1,...,6}.
    coefficients : np.ndarray
        Fourier coefficients [a0, a1, b1, a2, b2, ...] where a0 is the intercept.
    component : pd.Series
        Fitted annual component aligned to original index (intercept excluded).
    recency_amplitudes : dict
        Mapping from years (1, 2, 3) to amplitude sqrt(a1^2 + b1^2) estimated
        on that trailing window.
    """

    n_harmonics: int
    coefficients: np.ndarray
    component: pd.Series
    recency_amplitudes: Dict[int, float] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    """Full decomposition result from SeqdDecomposer.fit().

    Attributes
    ----------
    series : pd.Series
        Original input series.
    weekly : WeeklyEffect
        Stage 1 weekly effect.
    holidays : list of HolidayEffect
        Stage 2 holiday effects (one per occurrence (year) per holiday name).
    annual : AnnualEffect
        Stage 3 annual effect.
    residual : pd.Series
        Residual (trend + noise) after removing all components.
    r2_by_component : dict
        Variance explained relative to original series variance.
        Keys: "weekly", "holiday", "annual".
    """

    series: pd.Series
    weekly: WeeklyEffect
    holidays: List[HolidayEffect]
    annual: AnnualEffect
    residual: pd.Series
    r2_by_component: Dict[str, float]

    def weekly_component(self) -> pd.Series:
        """Return the full weekly effect series aligned to original index.

        In **additive** mode this is simply the DOW coefficient for each
        date (a level-independent series of 7 repeating values).

        In **multiplicative** mode the returned series is
        ``original * (1 - 1/coeff)``, i.e. the absolute amount that was
        removed from each observation.  The amplitude therefore scales with
        the local level of the series, which means the series is NOT
        suitable for direct comparison across periods with very different
        levels.  Use ``weekly.coefficients`` (the 7-element array of pure
        multiplicative factors) for a level-independent representation.

        In both modes the identity ``fitted() == series`` holds exactly
        because ``residual + weekly_component() + holiday_component() +
        annual_component() == series``.
        """
        idx = self.series.index
        dow = idx.dayofweek  # Monday=0
        if self.weekly.is_multiplicative:
            # In multiplicative mode: y_w = y / coeff
            # => removed = y - y/coeff = y * (1 - 1/coeff)
            # This additive representation ensures exact reconstruction.
            return pd.Series(
                self.series.values * (1.0 - 1.0 / self.weekly.coefficients[dow]),
                index=idx,
                name="weekly_component",
            )
        else:
            return pd.Series(
                self.weekly.coefficients[dow],
                index=idx,
                name="weekly_component",
            )

    def holiday_component(self) -> pd.Series:
        """Return sum of all holiday effect series."""
        if not self.holidays:
            return pd.Series(0.0, index=self.series.index, name="holiday_component")
        total = pd.Series(0.0, index=self.series.index)
        for h in self.holidays:
            aligned = h.effect_series.reindex(self.series.index, fill_value=0.0)
            total = total + aligned
        total.name = "holiday_component"
        return total

    def annual_component(self) -> pd.Series:
        """Return the annual Fourier component."""
        return self.annual.component.rename("annual_component")

    def fitted(self) -> pd.Series:
        """Return the reconstructed series: residual + all components."""
        w = self.weekly_component()
        h = self.holiday_component()
        a = self.annual_component()
        reconstructed = self.residual + w + h + a
        reconstructed.name = "fitted"
        return reconstructed
