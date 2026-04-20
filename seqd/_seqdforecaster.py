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
    slope_blend_alpha : float
        Weight placed on the penultimate segment's slope when blending for
        trend extrapolation (H1 fix).  Value must be in ``[0.0, 1.0]``.
    ipm_max_years : int
        Maximum number of most-recent historical years used when fitting the
        OLS trend on individual peak magnitudes (IPM) for compound holiday
        projection.  Default ``4``.  Reducing this value down-weights early
        high-growth years, useful for concave growth series.  Must be >= 1.
    ipm_decay_halflife : float
        Exponential decay half-life in years for down-weighting older IPM
        observations within the OLS fit window.  ``0.0`` (default) uses
        uniform weights (standard OLS).  Values in ``(0, 2]`` progressively
        down-weight older years: e.g. ``1.0`` gives year N-1 half the weight
        of year N.

        ``blended_slope = alpha * slope_penultimate + (1 - alpha) * slope_final``

        Default ``0.5`` — equal weight on the penultimate and final segment
        slopes.  This is structurally motivated: the final segment often ends
        in Q4 (seasonal acceleration), making its slope steeper than the
        underlying structural growth rate.  Giving equal weight to the
        penultimate slope is more conservative and less likely to over-
        extrapolate.  At ``alpha=0.0`` no blending occurs (final-segment-only,
        legacy behaviour).  At ``alpha=1.0`` the penultimate slope is used
        entirely.

        When only one segment exists, blending is skipped regardless of
        ``alpha``.
    trend_yoy_blend : float
        Weight on a trend-growth-implied IPM projection for compound holiday
        blocks (0 = pure OLS, 1 = pure trend-implied).  Default ``0.0``
        (backward-compatible).

        The trend-implied projection scales the most-recent historical IPM
        by the de-seasonalized YoY growth ratio observed in the V1 residual
        (recent 90-day window vs the same window one year prior).  This
        prevents the OLS slope from over-extrapolating BFCM magnitude when
        the underlying business growth rate is moderate: holiday-season lift
        is economically a multiplier on the baseline level, not an independent
        quantity.
    use_adaptive_ipm : bool
        Convenience preset that enables recency-weighted IPM projection with
        trend-blending.  When ``True``, overrides the following parameters
        regardless of the values passed for them:

        - ``trend_yoy_blend = 0.5``  (equal blend of OLS and trend-implied IPM)
        - ``ipm_decay_halflife = 1.5``  (recent years weighted more heavily)

        Default ``False``.  Use this when recent business momentum should
        inform holiday projection — for example, when the growth rate has
        shifted meaningfully in the past 1–2 years and you want the IPM
        projection to reflect that recency.

        ``slope_blend_alpha`` and ``ipm_max_years`` are **not** overridden by
        this preset and retain their passed (or default) values.

    Raises
    ------
    ValueError
        If ``result.residual`` is empty or has fewer than 2 observations.
    ValueError
        If ``slope_blend_alpha`` is not in ``[0.0, 1.0]``.

    Examples
    --------
    >>> forecaster = SeqdForecaster(result, slope_blend_alpha=0.5, trend_yoy_blend=0.5)
    >>> forecaster.fit(changepoint_penalty_beta=3.0, min_segment_size=60)
    >>> fr = forecaster.predict(horizon=365)
    >>> fr.forecast.head()

    Using the adaptive IPM preset:

    >>> forecaster = SeqdForecaster(result, use_adaptive_ipm=True)
    >>> forecaster.fit(changepoint_penalty_beta=3.0, min_segment_size=60)
    >>> fr = forecaster.predict(horizon=365)
    """

    # Adaptive IPM preset values
    _ADAPTIVE_IPM_TREND_YOY_BLEND: float = 0.5
    _ADAPTIVE_IPM_DECAY_HALFLIFE: float = 1.5

    def __init__(
        self,
        result: DecompositionResult,
        slope_blend_alpha: float = 0.5,
        trend_yoy_blend: float = 0.0,
        ipm_max_years: int = 4,
        ipm_decay_halflife: float = 0.0,
        use_adaptive_ipm: bool = False,
    ) -> None:
        # Apply adaptive IPM preset overrides before validation
        if use_adaptive_ipm:
            trend_yoy_blend = self._ADAPTIVE_IPM_TREND_YOY_BLEND
            ipm_decay_halflife = self._ADAPTIVE_IPM_DECAY_HALFLIFE

        if len(result.residual) < 2:
            raise ValueError(
                f"result.residual must have at least 2 observations "
                f"(got {len(result.residual)})."
            )
        if not (0.0 <= slope_blend_alpha <= 1.0):
            raise ValueError(
                f"slope_blend_alpha must be in [0.0, 1.0] (got {slope_blend_alpha})."
            )
        if not (0.0 <= trend_yoy_blend <= 1.0):
            raise ValueError(
                f"trend_yoy_blend must be in [0.0, 1.0] (got {trend_yoy_blend})."
            )
        if ipm_max_years < 1:
            raise ValueError(
                f"ipm_max_years must be >= 1 (got {ipm_max_years})."
            )
        if ipm_decay_halflife < 0.0:
            raise ValueError(
                f"ipm_decay_halflife must be >= 0.0 (got {ipm_decay_halflife})."
            )
        self._result = result
        self._slope_blend_alpha = slope_blend_alpha
        self._trend_yoy_blend = trend_yoy_blend
        self._ipm_max_years = ipm_max_years
        self._ipm_decay_halflife = ipm_decay_halflife
        self._use_adaptive_ipm = use_adaptive_ipm
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
        min_segment_size: int = 60,
        aic_linear_delta: float = 2.0,
        enable_bfcm_carveout: bool = True,
        bfcm_carveout_min_overlap_days: int = 30,
    ) -> "SeqdForecaster":
        """Run Stages 4 and 5: changepoint detection and piecewise trend fitting.

        Parameters
        ----------
        changepoint_penalty_beta : float
            Multiplier β for the BIC-inspired PELT penalty
            λ = σ̂² × β × ln(n).  Default 3.0.  Reasonable range [1.0, 6.0].
        min_segment_size : int
            Minimum segment length in days for PELT.  Default 60.
            Lowered from 90 (v0.2.0) to improve detection of mid-year trend
            breaks.  Must satisfy ``n >= 2 * min_segment_size`` for
            changepoints to be detectable.
        aic_linear_delta : float
            AIC threshold δ below which linear is preferred over a
            better-fitting model.  Default 2.0.
        enable_bfcm_carveout : bool
            When ``True`` (default), if the final trend segment overlaps a
            compound holiday block window by more than
            ``bfcm_carveout_min_overlap_days`` days, the segment trend is
            re-fitted on the pre-compound-block sub-window only.  This
            prevents the linear interpolation bridge created by compound
            masking (which slopes upward through the BFCM ramp) from
            inflating the final segment slope and producing spurious growth
            forecasts.
        bfcm_carveout_min_overlap_days : int
            Minimum overlap (in days) between the final segment and any
            compound block window before the carve-out is applied.
            Default 30.

        Returns
        -------
        self : SeqdForecaster
            Returns self to allow method chaining: ``fit().predict(...)``.
        """
        residual = self._result.residual

        # Mask compound block windows before changepoint detection AND trend
        # fitting.  BFCM-style compound effects end abruptly at Dec 31; the
        # resulting level-drop on Jan 1 appears as a false structural break to
        # PELT.  More critically, if the final segment includes the BFCM ramp,
        # fitting the trend on the raw residual inflates the estimated slope,
        # producing spurious +100%+ YoY growth in forecasts.  Using the
        # compound-masked residual (linearly interpolated through each block)
        # for both steps removes both artefacts.
        masked_residual = self._mask_compound_windows(
            residual, self._result.holidays
        )

        changepoint_dates, _ = detect_changepoints(
            residual=masked_residual,
            penalty_beta=changepoint_penalty_beta,
            min_size=min_segment_size,
        )

        segments = fit_piecewise_trend(
            residual=masked_residual,
            changepoint_dates=changepoint_dates,
            aic_linear_delta=aic_linear_delta,
        )

        # BFCM carve-out: if the final segment spans a compound holiday block
        # window by a significant margin, re-fit the trend on only the
        # pre-compound sub-window.  The compound masking step above linearly
        # interpolates through each block, creating an upward-sloping bridge
        # from the pre-BFCM level to the post-BFCM level.  When this bridge
        # falls inside the final segment, OLS fits an inflated slope, causing
        # over-extrapolation in the forecast horizon.
        if enable_bfcm_carveout and len(segments) > 0:
            final_seg = segments[-1]
            final_seg_start = pd.Timestamp(final_seg.start_date)
            final_seg_end = pd.Timestamp(final_seg.end_date)

            # Find the latest compound block that overlaps the final segment.
            # GUARD: only apply carveout when the series end falls INSIDE the
            # compound block window (i.e. the block is still "in flight" at
            # the training cutoff).  When the series ends AFTER the block_end,
            # the post-BFCM recovery data is already captured in the final
            # segment's OLS fit, so no carveout is needed.  Applying the
            # carveout to a fully-settled block would truncate the segment to
            # the tiny pre-block stub, producing a wildly over-fitted slope.
            carveout_cutoff: Optional[pd.Timestamp] = None
            series_end = final_seg_end
            for eff in self._result.holidays:
                if eff.compound:
                    block_start = pd.Timestamp(eff.ramp_start)
                    block_end = pd.Timestamp(eff.ramp_end)
                    # Only apply carveout if the block is unfinished at series end
                    if series_end > block_end:
                        continue  # block already settled — no carveout needed
                    # Check overlap with final segment
                    overlap_start = max(final_seg_start, block_start)
                    overlap_end = min(final_seg_end, block_end)
                    overlap_days = (overlap_end - overlap_start).days + 1
                    if overlap_days >= bfcm_carveout_min_overlap_days:
                        # Use the day before the block starts as the cutoff
                        candidate = block_start - pd.Timedelta(days=1)
                        if carveout_cutoff is None or candidate > carveout_cutoff:
                            carveout_cutoff = candidate

            if carveout_cutoff is not None and carveout_cutoff > final_seg_start:
                # Re-fit the final segment on the sub-window ending at carveout_cutoff
                pre_block_residual = masked_residual.loc[
                    final_seg_start:carveout_cutoff
                ]
                if len(pre_block_residual) >= 2:
                    carveout_segments = fit_piecewise_trend(
                        residual=pre_block_residual,
                        changepoint_dates=[],  # no sub-changepoints for this window
                        aic_linear_delta=aic_linear_delta,
                    )
                    if carveout_segments:
                        # Replace the final segment with the carveout-fitted segment,
                        # preserving the original segment index.
                        carved = carveout_segments[0]
                        segments[-1] = SegmentTrend(
                            segment_index=final_seg.segment_index,
                            start_date=carved.start_date,
                            end_date=carved.end_date,
                            n_obs=carved.n_obs,
                            model_type=carved.model_type,
                            alpha=carved.alpha,
                            beta=carved.beta,
                            gamma=carved.gamma,
                            T_days=carved.T_days,
                            aic=carved.aic,
                            aic_linear=carved.aic_linear,
                            rss=carved.rss,
                            selected_reason=carved.selected_reason
                            + " [bfcm_carveout]",
                            t_anchor_date=carved.t_anchor_date,
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
        max_holiday_merge_gap_days: int = 35,
        ramp_half_width: int = 3,
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
        max_holiday_merge_gap_days : int
            Maximum date gap in days between future holiday dates for
            grouping them into a compound proximity group for max-pooled
            ramp projection.  Should match the value used in the V1
            decomposer.  Default 35.
        ramp_half_width : int
            Half-width of the triangular ramp used for compound holiday
            projection (days).  Default 3.  Increasing this spreads the
            projected holiday lift over a wider window; decreasing it
            concentrates the effect closer to the holiday date.  Must be
            >= 0.

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
        penultimate_segment = segments[-2] if len(segments) >= 2 else None
        trend_arr, forecast_warnings = _project_trend(
            last_segment=last_segment,
            horizon=horizon,
            max_extrapolation_days=max_extrapolation_days,
            penultimate_segment=penultimate_segment,
            slope_blend_alpha=self._slope_blend_alpha,
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
            max_holiday_merge_gap_days=max_holiday_merge_gap_days,
            trend_yoy_blend=self._trend_yoy_blend,
            ramp_half_width=ramp_half_width,
            ipm_max_years=self._ipm_max_years,
            ipm_decay_halflife=self._ipm_decay_halflife,
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
    min_segment_size: int = 60,
    aic_linear_delta: float = 2.0,
    max_extrapolation_days: int = 365,
    slope_blend_alpha: float = 0.5,
    max_holiday_merge_gap_days: int = 35,
    trend_yoy_blend: float = 0.0,
    ramp_half_width: int = 3,
    ipm_max_years: int = 4,
    enable_bfcm_carveout: bool = True,
    ipm_decay_halflife: float = 0.0,
) -> ForecastResult:
    """Convenience wrapper: fit and forecast in one call.

    Equivalent to::

        SeqdForecaster(result, slope_blend_alpha=slope_blend_alpha,
                       trend_yoy_blend=trend_yoy_blend).fit(
            changepoint_penalty_beta=changepoint_penalty_beta,
            min_segment_size=min_segment_size,
            aic_linear_delta=aic_linear_delta,
        ).predict(
            horizon=horizon,
            future_holidays=future_holidays,
            max_extrapolation_days=max_extrapolation_days,
            max_holiday_merge_gap_days=max_holiday_merge_gap_days,
            ramp_half_width=ramp_half_width,
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
        See :meth:`SeqdForecaster.fit`.  Default 60.
    aic_linear_delta : float
        See :meth:`SeqdForecaster.fit`.
    max_extrapolation_days : int
        See :meth:`SeqdForecaster.predict`.
    slope_blend_alpha : float
        See :class:`SeqdForecaster`.  Default 0.5.
    max_holiday_merge_gap_days : int
        See :meth:`SeqdForecaster.predict`.  Default 35.
    trend_yoy_blend : float
        See :class:`SeqdForecaster`.  Default 0.0 (pure OLS).
    ramp_half_width : int
        See :meth:`SeqdForecaster.predict`.  Default 3.
    ipm_max_years : int
        See :class:`SeqdForecaster`.  Default 4.
    enable_bfcm_carveout : bool
        See :meth:`SeqdForecaster.fit`.  Default ``True``.
    ipm_decay_halflife : float
        See :class:`SeqdForecaster`.  Default 0.0 (uniform weights).

    Returns
    -------
    ForecastResult
    """
    return (
        SeqdForecaster(
            result,
            slope_blend_alpha=slope_blend_alpha,
            trend_yoy_blend=trend_yoy_blend,
            ipm_max_years=ipm_max_years,
            ipm_decay_halflife=ipm_decay_halflife,
        )
        .fit(
            changepoint_penalty_beta=changepoint_penalty_beta,
            min_segment_size=min_segment_size,
            aic_linear_delta=aic_linear_delta,
            enable_bfcm_carveout=enable_bfcm_carveout,
        )
        .predict(
            horizon=horizon,
            future_holidays=future_holidays,
            max_extrapolation_days=max_extrapolation_days,
            max_holiday_merge_gap_days=max_holiday_merge_gap_days,
            ramp_half_width=ramp_half_width,
        )
    )
