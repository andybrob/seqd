"""SeqdForecaster: V2 changepoint detection, trend fitting, and forecasting."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._changepoint import detect_changepoints
from ._forecast import (
    _combine_components,
    _project_annual,
    _project_holidays,
    _project_trend,
    _project_weekly,
)
from ._structures import DecompositionResult, ForecastResult, SegmentTrend
from ._trend import fit_piecewise_trend


class SeqdForecaster:
    """Post-process a V1 :class:`DecompositionResult` with changepoint
    detection, piecewise trend fitting, and out-of-sample forecasting.

    The ``SeqdForecaster`` is a read-only consumer of the V1 decomposition:
    it does not modify ``result.residual`` or any V1 component.

    Parameters
    ----------
    result : DecompositionResult
        Output of ``SeqdDecomposer.fit()``.  Must have a non-empty residual.

    Raises
    ------
    ValueError
        If ``result.residual`` is empty or has fewer than 2 observations.

    Examples
    --------
    >>> forecaster = SeqdForecaster(result)
    >>> forecaster.fit(changepoint_penalty_beta=3.0, min_segment_size=90)
    >>> fr = forecaster.predict(horizon=365)
    >>> fr.forecast.head()
    """

    def __init__(self, result: DecompositionResult) -> None:
        if len(result.residual) < 2:
            raise ValueError(
                f"result.residual must have at least 2 observations "
                f"(got {len(result.residual)})."
            )
        self._result = result
        self._fitted = False
        self._changepoints: Optional[List[pd.Timestamp]] = None
        self._segments: Optional[List[SegmentTrend]] = None

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_compound_windows(
        residual: pd.Series,
        holiday_effects,
    ) -> pd.Series:
        """Replace residual values inside compound block windows with linear interpolation.

        PELT sees the abrupt drop in the residual at the end of each BFCM compound
        block (Jan 1) as a level-shift changepoint, producing ~6-8 spurious annual
        detections.  Masking and re-interpolating the residual inside every compound
        window removes these artefacts before the changepoint search.

        Parameters
        ----------
        residual : pd.Series
            V1 residual series (daily DatetimeIndex).
        holiday_effects : list of HolidayEffect
            From ``DecompositionResult.holidays``.

        Returns
        -------
        pd.Series
            Copy of residual with compound-window values replaced by linear
            interpolation between the edges of each masked block.
        """
        masked = residual.copy()
        for eff in holiday_effects:
            if eff.compound:
                ramp_start = pd.Timestamp(eff.ramp_start)
                ramp_end = pd.Timestamp(eff.ramp_end)
                mask = (masked.index >= ramp_start) & (masked.index <= ramp_end)
                if mask.sum() > 0:
                    masked.loc[mask] = np.nan
        masked = masked.interpolate(method="linear", limit_direction="both")
        return masked

    def fit(
        self,
        changepoint_penalty_beta: float = 3.0,
        min_segment_size: int = 90,
        aic_linear_delta: float = 2.0,
    ) -> "SeqdForecaster":
        """Run Stages 4 and 5: changepoint detection and piecewise trend fitting.

        Parameters
        ----------
        changepoint_penalty_beta : float
            Multiplier β for the BIC-inspired PELT penalty
            λ = σ̂² × β × ln(n).  Default 3.0.  Reasonable range [1.0, 6.0].
        min_segment_size : int
            Minimum segment length in days for PELT.  Default 90.
            Must satisfy ``n >= 2 * min_segment_size`` for changepoints to
            be detectable.
        aic_linear_delta : float
            AIC threshold δ below which linear is preferred over a
            better-fitting model.  Default 2.0.

        Returns
        -------
        self : SeqdForecaster
            Returns self to allow method chaining: ``fit().predict(...)``.
        """
        residual = self._result.residual

        # Mask compound block windows before changepoint detection.
        # BFCM-style compound effects end abruptly at Dec 31; the resulting
        # level-drop on Jan 1 appears as a false structural break to PELT.
        # Interpolating through those windows suppresses ~6-8 spurious annual
        # changepoints without affecting the final trend fit (which uses the
        # original, unmasked residual below).
        residual_for_detection = self._mask_compound_windows(
            residual, self._result.holidays
        )

        changepoint_dates, _ = detect_changepoints(
            residual=residual_for_detection,
            penalty_beta=changepoint_penalty_beta,
            min_size=min_segment_size,
        )

        segments = fit_piecewise_trend(
            residual=residual,
            changepoint_dates=changepoint_dates,
            aic_linear_delta=aic_linear_delta,
        )

        self._changepoints = changepoint_dates
        self._segments = segments
        self._fitted = True
        return self

    def predict(
        self,
        horizon: int,
        future_holidays: Optional[Dict[str, List[pd.Timestamp]]] = None,
        max_extrapolation_days: int = 365,
    ) -> ForecastResult:
        """Run Stage 6: produce ``horizon``-day-ahead point forecasts.

        Parameters
        ----------
        horizon : int
            Number of future days to forecast (H).  Must be >= 1.
        future_holidays : dict or None
            Mapping from holiday name to list of future ``pd.Timestamp``
            dates for that holiday.  Names must exactly match those in
            ``result.holidays``.  Dates outside the forecast horizon are
            silently ignored.  Default ``None`` (no future holidays;
            holiday component is zero).
        max_extrapolation_days : int
            Emit a ``UserWarning`` when ``horizon > max_extrapolation_days``.
            Default 365.

        Returns
        -------
        ForecastResult
            Contains total forecast and all component series.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called before :meth:`predict`.
        ValueError
            If ``horizon < 1``.
        """
        if not self._fitted:
            raise RuntimeError(
                "fit() must be called before predict(). "
                "Call SeqdForecaster.fit() first."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1 (got {horizon}).")

        result = self._result
        segments = self._segments
        changepoints = self._changepoints

        # Build forecast date index: last date + 1 to last date + horizon
        last_date = result.residual.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # --- Trend projection ---
        last_segment = segments[-1]
        trend_arr, forecast_warnings = _project_trend(
            last_segment=last_segment,
            horizon=horizon,
            max_extrapolation_days=max_extrapolation_days,
        )

        # --- Weekly projection ---
        weekly_arr = _project_weekly(
            result=result,
            forecast_dates=forecast_dates,
        )

        # --- Annual projection ---
        annual_arr = _project_annual(
            result=result,
            forecast_dates=forecast_dates,
        )

        # --- Holiday projection ---
        if future_holidays is None:
            future_holidays = {}
        holiday_arr = _project_holidays(
            result=result,
            forecast_dates=forecast_dates,
            future_holidays=future_holidays,
        )

        # --- Combine ---
        is_mult = result.weekly.is_multiplicative
        forecast_arr = _combine_components(
            trend=trend_arr,
            weekly=weekly_arr,
            annual=annual_arr,
            holiday=holiday_arr,
            is_multiplicative=is_mult,
        )

        return ForecastResult(
            forecast=pd.Series(forecast_arr, index=forecast_dates, name="forecast"),
            trend_component=pd.Series(
                trend_arr, index=forecast_dates, name="trend_component"
            ),
            weekly_component=pd.Series(
                weekly_arr, index=forecast_dates, name="weekly_component"
            ),
            annual_component=pd.Series(
                annual_arr, index=forecast_dates, name="annual_component"
            ),
            holiday_component=pd.Series(
                holiday_arr, index=forecast_dates, name="holiday_component"
            ),
            changepoints=list(changepoints),
            segments=list(segments),
            horizon=horizon,
            is_multiplicative=is_mult,
            warnings=list(forecast_warnings),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def changepoints(self) -> List[pd.Timestamp]:
        """Detected changepoint dates.  Available after :meth:`fit`.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "fit() must be called before accessing changepoints."
            )
        return list(self._changepoints)

    @property
    def segments(self) -> List[SegmentTrend]:
        """Fitted segment trend objects.  Available after :meth:`fit`.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "fit() must be called before accessing segments."
            )
        return list(self._segments)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def forecast_from_result(
    result: DecompositionResult,
    horizon: int,
    future_holidays: Optional[Dict[str, List[pd.Timestamp]]] = None,
    changepoint_penalty_beta: float = 3.0,
    min_segment_size: int = 90,
    aic_linear_delta: float = 2.0,
    max_extrapolation_days: int = 365,
) -> ForecastResult:
    """Convenience wrapper: fit and forecast in one call.

    Equivalent to::

        SeqdForecaster(result).fit(
            changepoint_penalty_beta=changepoint_penalty_beta,
            min_segment_size=min_segment_size,
            aic_linear_delta=aic_linear_delta,
        ).predict(
            horizon=horizon,
            future_holidays=future_holidays,
            max_extrapolation_days=max_extrapolation_days,
        )

    Parameters
    ----------
    result : DecompositionResult
        Output of ``SeqdDecomposer.fit()``.
    horizon : int
        Number of future days to forecast.
    future_holidays : dict or None
        See :meth:`SeqdForecaster.predict`.
    changepoint_penalty_beta : float
        See :meth:`SeqdForecaster.fit`.
    min_segment_size : int
        See :meth:`SeqdForecaster.fit`.
    aic_linear_delta : float
        See :meth:`SeqdForecaster.fit`.
    max_extrapolation_days : int
        See :meth:`SeqdForecaster.predict`.

    Returns
    -------
    ForecastResult
    """
    return (
        SeqdForecaster(result)
        .fit(
            changepoint_penalty_beta=changepoint_penalty_beta,
            min_segment_size=min_segment_size,
            aic_linear_delta=aic_linear_delta,
        )
        .predict(
            horizon=horizon,
            future_holidays=future_holidays,
            max_extrapolation_days=max_extrapolation_days,
        )
    )
