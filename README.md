# seqd

**Sequential time series decomposition with data-driven ramp detection.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Notation](#2-notation)
3. [Model Specification](#3-model-specification)
4. [Stage 1: Weekly Effect Estimation](#4-stage-1-weekly-effect-estimation)
5. [Stage 2: Holiday Effect Estimation](#5-stage-2-holiday-effect-estimation)
6. [Stage 3: Annual Effect Estimation](#6-stage-3-annual-effect-estimation)
7. [Recency Diagnostics Framework](#7-recency-diagnostics-framework)
8. [R² Decomposition](#8-r-decomposition)
9. [API Reference](#9-api-reference)
10. [Known Limitations](#10-known-limitations)
11. [Quick Start Example](#11-quick-start-example)

---

## 1. Overview

`seqd` decomposes a daily time series into four additive components: a weekly
day-of-week pattern, holiday ramp effects, annual Fourier seasonality, and a
residual representing trend plus noise.  The intended domain is high-frequency
business metrics — revenue, orders, page views — where three structurally
distinct non-trend signals coexist and interact, and where holidays induce
effects that extend days or weeks beyond the calendar date itself.

The fundamental identification problem with simultaneous decomposition tools
such as Prophet is underdetermination: with a shared parameter space and a
flexible trend, the optimizer can trade variance across components without
bound.  A holiday spike can be absorbed partially by the trend, partially by
the annual component, and partially by the holiday component in proportions
that vary with initialisation and regularisation choices, producing
non-interpretable coefficient estimates.  `seqd` resolves this by imposing a
strict sequential structure: each stage operates on the *residual* produced by
the previous stage, so later stages never have access to variance already
claimed.  The estimation problem is exactly determined at each stage — there is
no shared objective function, no competing gradients, and no regularisation
required.

The key mathematical insight is that the three signal types are approximately
orthogonal in the frequency domain: weekly effects live at periods of exactly
7 days and its aliases; holiday effects are sparse, aperiodic, and concentrated
around known calendar dates; annual effects live at periods near 365.25 days
and its subharmonics.  Sequential extraction exploits this near-orthogonality
to isolate each signal cleanly without iteration.  Residual contamination
across stages is small for well-separated series and is quantified by the
marginal R² decomposition described in Section 8.

---

## 2. Notation

| Symbol | Definition |
|--------|------------|
| $y_t$ | Observed daily time series value at date $t$ |
| $T_t$ | Local trend component (7-day centered rolling median) |
| $W_t$ | Weekly day-of-week seasonal component |
| $H_t$ | Holiday effect component |
| $S_t$ | Annual Fourier seasonal component |
| $\epsilon_t$ | Residual (trend + noise) after all components removed |
| $y_t^{(w)}$ | Residual after weekly removal |
| $y_t^{(h)}$ | Residual after holiday removal |
| $\mu_t$ | 7-day centered rolling median of $y_t$ (window 7, min periods 4) |
| $y_{dt}$ | Locally detrended series: $y_t / \mu_t$ (multiplicative) or $y_t - \mu_t$ (additive) |
| $\hat{w}_d$ | Estimated weekly coefficient for day-of-week $d \in \{0,\ldots,6\}$ (Monday = 0) |
| $\mathcal{H}$ | Set of all known holiday dates supplied by the caller |
| $\mathcal{H}_{\pm k}$ | Dates within $k$ days of any holiday in $\mathcal{H}$ |
| $h$ | A specific holiday date |
| $\mathcal{R}_h$ | Reference window for holiday $h$ |
| $R$ | Reference window length in days (default 60) |
| $k_0$ | Reference gap: days excluded symmetrically around $h$ from baseline estimation (default 14) |
| $W$ | Holiday search window half-width in days (default 14) |
| $\hat{B}(t)$ | OLS linear baseline fit on reference window $\mathcal{R}_h$ |
| $r(t)$ | Baseline-subtracted residual: $y_t^{(w)} - \hat{B}(t)$ |
| $\hat{\sigma}_{ref}$ | Robust scale estimate from reference window residuals |
| $S_\delta$ | CUSUM statistic at backward offset $\delta$ from $h$ |
| $\hat{t}_{start}$ | Detected ramp start date |
| $\hat{t}_{end}$ | Detected ramp end date |
| $\hat{H}(t)$ | Estimated holiday effect at date $t$ |
| $P$ | Annual period: 365.25 days |
| $K$ | Number of Fourier harmonics (BIC-selected) |
| $\mathbf{X}(t, K)$ | Fourier design matrix at time $t$ with $K$ harmonics |
| $\hat{\boldsymbol{\gamma}}$ | Fourier regression coefficients $[\hat{a}_0, \hat{a}_1, \hat{b}_1, \ldots, \hat{a}_K, \hat{b}_K]^\top$ |
| $\hat{S}(t)$ | Fitted annual component (intercept excluded) |
| $\text{CV}$ | Coefficient of variation of the detrended series: $\text{std}(y_{dt}) / |\bar{y}|$ |
| $\text{TrMean}_p(\cdot)$ | Trimmed mean cutting proportion $p$ from each tail |
| $\text{MAD}(\cdot)$ | Median absolute deviation |

---

## 3. Model Specification

### 3.1 Additive Model

$$y_t = \hat{w}_{\text{DOW}(t)} + \hat{H}(t) + \hat{S}(t) + \hat{a}_0 + \epsilon_t$$

where $\hat{a}_0$ is the intercept absorbed into the annual regression and $\epsilon_t$ is the final residual (trend + noise).

### 3.2 Multiplicative Model

In multiplicative mode the weekly component enters as a scaling factor:

$$y_t = \hat{w}_{\text{DOW}(t)} \cdot y_t^{(\text{dow})} \quad \Leftrightarrow \quad y_t^{(w)} = y_t / \hat{w}_{\text{DOW}(t)}$$

The holiday and annual stages always operate additively on $y_t^{(w)}$.

### 3.3 Auto-Detection Criterion

When `multiplicative=None`, the mode is chosen as follows.  Let $\mu_t$ be the 7-day centered rolling median.

**Primary rule:** compute the locally detrended series $y_t - \mu_t$.  If

$$\text{CV} = \frac{\text{std}(y_t - \mu_t)}{|\bar{y}|} > 0.15$$

multiplicative mode is selected.

**Secondary rule:** if $\text{CV} \leq 0.15$ and the series contains at least $4$ complete chunks of $\max(30,\lfloor n/10 \rfloor)$ days, compute the Pearson correlation between chunk-level standard deviations of $(y_t - \mu_t)$ and chunk-level means of $\mu_t$.  If this correlation exceeds 0.7, multiplicative mode is selected.  This catches series where the headline CV is modest but variance clearly scales with the local level.

### 3.4 Identification Strategy

Each stage conditions on the residual from the prior stage:

$$y \;\xrightarrow{\;\text{Stage 1}\;}\; y^{(w)} \;\xrightarrow{\;\text{Stage 2}\;}\; y^{(h)} \;\xrightarrow{\;\text{Stage 3}\;}\; \epsilon$$

No component is re-estimated after a later stage runs.  This is the defining property of sequential decomposition and is what makes the R² decomposition in Section 8 well-defined and unambiguous.

---

## 4. Stage 1: Weekly Effect Estimation

### 4.1 Local Detrending

Compute the 7-day centered rolling median:

$$\mu_t = \operatorname{Median}\!\left(y_{t-3}, \ldots, y_{t+3}\right)$$

requiring a minimum of 4 valid observations; edge positions with fewer than 4 observations in the window are filled with the series mean $\bar{y}$.

Locally detrended series:

$$y_{dt} = \begin{cases} y_t / \mu_t & \text{multiplicative} \\ y_t - \mu_t & \text{additive} \end{cases}$$

### 4.2 Holiday Exclusion Mask

Let $\mathcal{H}$ denote the set of all known holiday dates.  Define the proximity exclusion set:

$$\mathcal{H}_{\pm 4} = \{t : \exists\, h \in \mathcal{H},\; |t - h| \leq 4 \text{ days}\}$$

Observations in $\mathcal{H}_{\pm 4}$ are excluded from DOW coefficient estimation to prevent holiday contamination of the weekly effect.  If a particular day-of-week has no unmasked observations remaining, the mask is relaxed for that DOW and all its observations are used as a fallback.

### 4.3 DOW Coefficient Estimation

For each $d \in \{0, 1, \ldots, 6\}$ (Monday = 0):

$$\hat{w}_d^{(\text{raw})} = \operatorname{TrMean}_{10\%}\!\bigl(\{y_{dt} : \operatorname{DOW}(t) = d,\; t \notin \mathcal{H}_{\pm 4}\}\bigr)$$

The 10% trimmed mean discards the lowest and highest $\lfloor 0.10 \cdot n_d \rfloor$ values before averaging, providing robustness against outliers and residual holiday contamination.

### 4.4 Normalization

**Additive** (zero-sum constraint):

$$\hat{w}_d = \hat{w}_d^{(\text{raw})} - \frac{1}{7}\sum_{d'=0}^{6} \hat{w}_{d'}^{(\text{raw})}$$

**Multiplicative** (unit-mean constraint):

$$\hat{w}_d = \frac{\hat{w}_d^{(\text{raw})}}{\dfrac{1}{7}\displaystyle\sum_{d'=0}^{6} \hat{w}_{d'}^{(\text{raw})}}$$

These constraints ensure the weekly component has no net contribution to the series level.

### 4.5 Weekly-Adjusted Residual

$$y_t^{(w)} = \begin{cases} y_t / \hat{w}_{\operatorname{DOW}(t)} & \text{multiplicative} \\ y_t - \hat{w}_{\operatorname{DOW}(t)} & \text{additive} \end{cases}$$

In multiplicative mode, a safety guard replaces any coefficient with absolute value $< 10^{-10}$ with 1.0 to prevent division by near-zero values.

### 4.6 Recency Diagnostics

Rolling trailing-window estimates are computed for window lengths $w \in \{60, 90, 365\}$ days.  Windows longer than the series are skipped.  For each window $w$, the endpoint is stepped every 7 days backward through the series.  At each endpoint $\tau$, the trailing slice $[i_\tau - w + 1, i_\tau]$ is extracted and $\hat{w}_d^{(w, \tau)}$ is recomputed with the same holiday-masked trimmed-mean procedure and within-window normalization.

**Drift detection** uses the 365-day window, or the largest available window when the series is shorter than 365 days.  Let $\hat{w}_d^{(\tau)}$ be the normalized DOW-$d$ coefficient at step $\tau \in \{0, \ldots, M-1\}$.  The drift slope is:

$$\hat{\beta}_d^{(\text{drift})} = \frac{\displaystyle\sum_{\tau=0}^{M-1}\!\left(\tau - \bar{\tau}\right)\!\left(\hat{w}_d^{(\tau)} - \overline{w_d}\right)}{\displaystyle\sum_{\tau=0}^{M-1}\!\left(\tau - \bar{\tau}\right)^2}$$

Classification:

$$\operatorname{class}_d = \begin{cases} \text{"drifting"} & \text{if } \left|\hat{\beta}_d^{(\text{drift})}\right| > 0.01 \cdot \left|\hat{w}_d^{(\text{full})}\right| \\ \text{"stable"} & \text{otherwise} \end{cases}$$

where $\hat{w}_d^{(\text{full})}$ is the full-sample coefficient.  When $\hat{w}_d^{(\text{full})} = 0$, an absolute tolerance of $10^{-6}$ is used.

---

## 5. Stage 2: Holiday Effect Estimation

Stage 2 operates on $y_t^{(w)}$.  Each holiday occurrence (one date per year per holiday name) is processed independently through Sections 5.1–5.4, then overlapping ramp windows are resolved in Section 5.5.

### 5.1 Local Baseline Estimation

For a holiday at date $h$, define the reference window (using signed day offsets $\tau(t) = (t - h)$ in days):

$$\mathcal{R}_h = \bigl\{t : -R \leq \tau(t) \leq -k_0\bigr\} \cup \bigl\{t : k_0 \leq \tau(t) \leq R\bigr\}$$

with $R = 60$ (default `reference_window`) and $k_0 = 14$ (default `reference_gap`).  Days within 3 days of any *other* holiday in $\mathcal{H}$ are additionally excluded from $\mathcal{R}_h$.  If the resulting reference set has fewer than 10 observations, the occurrence is skipped.

Fit OLS linear trend on $\mathcal{R}_h$:

$$\hat{B}(t) = \hat{\alpha} + \hat{\beta}\,\tau(t), \qquad [\hat{\alpha},\,\hat{\beta}]^\top = \left(\mathbf{X}_{ref}^\top \mathbf{X}_{ref}\right)^{-1} \mathbf{X}_{ref}^\top \mathbf{y}_{ref}$$

where $\mathbf{X}_{ref} = [\mathbf{1},\, \boldsymbol{\tau}_{ref}]$ is the $|\mathcal{R}_h| \times 2$ design matrix and $\mathbf{y}_{ref}$ contains the corresponding values of $y_t^{(w)}$.

Compute baseline-subtracted residuals over the search span $[h - W,\, h + W]$:

$$r(t) = y_t^{(w)} - \hat{B}(t), \qquad t \in [h - W,\; h + W]$$

Robust scale estimate on the reference residuals $\{r(t) : t \in \mathcal{R}_h\}$:

$$\hat{\sigma}_{ref} = \frac{\operatorname{MAD}\!\left(\{r(t) : t \in \mathcal{R}_h\}\right)}{0.6745}$$

where $\operatorname{MAD}(x) = \operatorname{Median}(|x_i - \operatorname{Median}(x)|)$ and the divisor 0.6745 $= \Phi^{-1}(3/4)$ is the Gaussian-consistency correction, so that $\hat{\sigma}_{ref}$ is asymptotically unbiased for $\sigma$ when residuals are normal.  If $\hat{\sigma}_{ref} = 0$ or is non-finite, the fallback is $\operatorname{std}(r(\mathcal{R}_h))$ or 1.0.

### 5.2 Ramp-Start Detection (Dual Criterion)

Ramp-start detection scans backward from $h$, examining $r(h - \delta)$ for $\delta = 0, 1, \ldots, W$.  Two criteria run in parallel; the result is their union.

**Criterion 1 — CUSUM:**

$$S_0 = 0, \qquad S_\delta = \max\!\left(0,\; S_{\delta-1} + \left|r(h - \delta)\right| - \hat{\sigma}_{ref}\right)$$

The CUSUM accumulates the excess of $|r(t)|$ above $\hat{\sigma}_{ref}$, resetting to zero whenever the cumulative excess goes negative.  Let $\delta^*_{\text{reset}}$ be the most recent reset index before the statistic first exceeds $2\hat{\sigma}_{ref}$:

$$\hat{t}_{start}^{(C)} = h - \delta^*_{\text{reset}} \quad \text{(date one step after the last reset)}$$

The implementation tracks `last_reset` — the date immediately preceding the last zero-crossing — and sets $\hat{t}_{start}^{(C)} = \text{last\_reset} + 1\text{ day}$ at each threshold crossing.  If $S_\delta \leq 2\hat{\sigma}_{ref}$ for all $\delta$, the criterion does not fire and $\hat{t}_{start}^{(C)} = h$.

**Criterion 2 — Run length:**

A *run* is a maximal sequence of consecutive backward steps $\delta$ in which $|r(h - \delta)| > 0.5\,\hat{\sigma}_{ref}$.  Let $\mathcal{L}$ be the set of farthest-back dates (earliest calendar dates) reached by all runs of length $\geq 4$.  If $\mathcal{L} \neq \emptyset$:

$$\hat{t}_{start}^{(R)} = \min\{\ell : \ell \in \mathcal{L}\}$$

(earliest date, i.e. farthest from $h$).  If $\mathcal{L} = \emptyset$, the criterion does not fire and $\hat{t}_{start}^{(R)} = h$.

**Combined:**

$$\hat{t}_{start} = \min\!\left(\hat{t}_{start}^{(C)},\; \hat{t}_{start}^{(R)}\right)$$

CUSUM is sensitive to concentrated, high-amplitude ramps.  The run criterion is sensitive to sustained, low-amplitude ramps lasting several weeks.  The union handles both shapes.

### 5.3 Ramp-End Detection

Scan forward from $h + 1$, tracking consecutive days where $|r(t)| < 1.5\,\hat{\sigma}_{ref}$:

Before checking the threshold criterion, the residuals are smoothed with a 3-point median filter: $\tilde{r}(t) = \operatorname{Median}(r(t-1), r(t), r(t+1))$ (boundary values default to 0).  The smoothing suppresses false early exits caused by isolated noisy days within an otherwise active ramp.

$$\hat{t}_{end} = \min\!\left\{t > h : |\tilde{r}(t)| < 1.5\,\hat{\sigma}_{ref} \;\text{ and }\; |\tilde{r}(t+1)| < 1.5\,\hat{\sigma}_{ref}\right\} - 1\text{ day}$$

That is, the ramp end is set to the first day of the first pair of consecutive below-threshold smoothed residual days, minus one, so that the window closes on the last day the series is still elevated.  If no such pair exists within $W$ days after $h$, the fallback is:

$$\hat{t}_{end} = h + W$$

### 5.4 Per-Occurrence Holiday Effect and Residual

$$\hat{H}(t) = \begin{cases} r(t) & t \in [\hat{t}_{start},\; \hat{t}_{end}] \\ 0 & \text{otherwise} \end{cases}$$

The holiday-adjusted residual (after compound block resolution in Section 5.5) is:

$$y_t^{(h)} = y_t^{(w)} - \sum_{\text{occurrences}} \hat{H}(t)$$

where the sum runs over all `effect_series` values across all `HolidayEffect` objects (with compound-block non-primary members contributing zero, per Section 5.5).

### 5.5 Compound Block Handling

After individual occurrence processing, all detected ramp windows are sorted by $\hat{t}_{start}$ and tested for overlap.  Two occurrences $i$ and $j$ are merged if:

$$[\hat{t}_{start}^{(i)},\; \hat{t}_{end}^{(i)}] \cap [\hat{t}_{start}^{(j)},\; \hat{t}_{end}^{(j)}] \neq \emptyset$$

Overlapping occurrences are merged greedily (by ramp-start order) into a *compound block* with:

$$t_{start}^{(\text{block})} = \min_{i \in \mathcal{B}} \hat{t}_{start}^{(i)}, \qquad t_{end}^{(\text{block})} = \max_{i \in \mathcal{B}} \hat{t}_{end}^{(i)}$$

The canonical effect series for a compound block is the pointwise mean of the constituent individual effect series:

$$\hat{H}^{(\text{block})}(t) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \hat{H}_i(t)$$

Block magnitude is the mean of $\hat{H}^{(\text{block})}(t)$ over $[t_{start}^{(\text{block})}, t_{end}^{(\text{block})}]$.

**Double-counting prevention.** The *primary member* of a compound block is defined as the member with the earliest holiday date $h$.  The primary member's `effect_series` is set to $\hat{H}^{(\text{block})}(\cdot)$.  All other members receive a zero `effect_series`.  When `holiday_component()` sums across all `HolidayEffect` objects, each compound block contributes exactly one copy of its canonical effect.

### 5.6 Per-Occurrence Recency and Drift

Each `HolidayEffect` stores `year_magnitudes`: the ordered list of per-occurrence magnitudes across all years for the same holiday name, computed from occurrence-level (pre-merge) ramp windows:

$$\operatorname{year\_magnitudes} = \left[\hat{m}_1, \hat{m}_2, \ldots, \hat{m}_Y\right], \qquad \hat{m}_y = \operatorname{mean}\!\left(\hat{H}_y(t) : t \in [\hat{t}_{start}^{(y)},\; \hat{t}_{end}^{(y)}]\right)$$

**Magnitude drift** is the OLS slope of year magnitudes over year index $y = 0, 1, \ldots, Y-1$:

$$\hat{\beta}^{(\text{mag})} = \frac{\displaystyle\sum_{y=0}^{Y-1}(y - \bar{y})(\hat{m}_y - \bar{m})}{\displaystyle\sum_{y=0}^{Y-1}(y - \bar{y})^2}$$

If $Y < 2$, `magnitude_drift` is set to 0.

**Individual peak magnitude** is the mean of the occurrence-level (pre-merge) effect over the $\pm 3$-day window centred on $h$:

$$\operatorname{individual\_peak\_magnitude}_h = \frac{1}{|\mathcal{N}_h|}\sum_{t \in \mathcal{N}_h} \hat{H}_h(t), \qquad \mathcal{N}_h = \{t : |t - h| \leq 3 \text{ days}\}$$

This allows comparing the peak contributions of constituent holidays within a compound block, independently of the canonical block-level effect.

---

## 6. Stage 3: Annual Effect Estimation

### 6.1 Fourier Basis

Let $t_i$ denote the integer number of days from January 1 of the first calendar year in the series to observation $i$.  Specifically, if the series starts on date $d_0$, then $t_i = (d_i - \text{Jan 1 of } d_0\text{'s year})$ in days.  This calendar-anchored time index ensures that Fourier phases are aligned to the calendar year rather than the arbitrary start date of the series, so that the estimated $\hat{a}_k$ and $\hat{b}_k$ coefficients carry consistent seasonal-phase interpretations across datasets with different start dates.  For $K$ harmonics, the Fourier design matrix is:

$$\mathbf{X}(t, K) = \left[1,\; \cos\frac{2\pi t}{P},\; \sin\frac{2\pi t}{P},\; \cos\frac{4\pi t}{P},\; \sin\frac{4\pi t}{P},\; \ldots,\; \cos\frac{2K\pi t}{P},\; \sin\frac{2K\pi t}{P}\right]$$

where $P = 365.25$ days.  This gives an $n \times (2K+1)$ design matrix including an intercept column.

### 6.2 Harmonic Selection via BIC

To prevent a strong linear trend from competing with Fourier harmonics in the BIC criterion, $y_t^{(h)}$ is linearly detrended before model selection.  Let:

$$\tilde{y}_t^{(h)} = y_t^{(h)} - \hat{\alpha}_{dt} - \hat{\beta}_{dt}\,t$$

where $[\hat{\alpha}_{dt}, \hat{\beta}_{dt}]$ are OLS estimates from regressing $y_t^{(h)}$ on $[1, t]$.  This detrended series is used **only for BIC selection**; the final component fit in Section 6.3 uses the original $y_t^{(h)}$.

BIC is evaluated over $K \in \{0, 1, 2, 3, 4, 5, 6\}$:

$$\operatorname{BIC}(K) = n \log\!\left(\frac{\operatorname{RSS}(K)}{n}\right) + (2K+1)\log(n)$$

where $\operatorname{RSS}(K) = \sum_{t=0}^{n-1}\!\left(\tilde{y}_t^{(h)} - \hat{\tilde{y}}_t^{(K)}\right)^2$ is the residual sum of squares from fitting $\mathbf{X}(t, K)$ on the detrended series, and $K = 0$ (intercept only, one parameter) is the no-seasonality baseline.  The selected order is $\hat{K} = \arg\min_K \operatorname{BIC}(K)$.  Any $K$ whose fitted values are non-finite is skipped.

### 6.3 Final Estimation

Fit OLS on the original (non-detrended) $y_t^{(h)}$ with the selected $\hat{K}$:

$$\hat{\boldsymbol{\gamma}} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}^{(h)}, \qquad \hat{\boldsymbol{\gamma}} = [\hat{a}_0,\, \hat{a}_1,\, \hat{b}_1,\, \ldots,\, \hat{a}_{\hat{K}},\, \hat{b}_{\hat{K}}]^\top$$

The annual component (intercept excluded) is:

$$\hat{S}(t) = \sum_{k=1}^{\hat{K}} \left[\hat{a}_k \cos\frac{2\pi k t}{P} + \hat{b}_k \sin\frac{2\pi k t}{P}\right]$$

If $\hat{K} = 0$, the annual component is identically zero; only the intercept $\hat{a}_0$ is estimated.

The final residual is:

$$\hat{\epsilon}_t = y_t^{(h)} - \hat{S}(t) - \hat{a}_0$$

### 6.4 Recency Amplitudes

For trailing windows of 1, 2, and 3 years, a $K = 1$ Fourier model is fit on the linearly detrended sub-series.  With estimated coefficients $[\hat{a}_0', \hat{a}_1', \hat{b}_1']$, the fundamental amplitude is:

$$A_{(\text{years})} = \sqrt{(\hat{a}_1')^2 + (\hat{b}_1')^2}$$

This provides a diagnostic for whether annual seasonality amplitude is stable, growing, or decaying over recent history.  Windows shorter than 30 observations yield `NaN`.

---

## 7. Recency Diagnostics Framework

For all three stages, rolling trailing-window estimators track whether an estimated effect is stable or changing over recent data.

**General form.** Let $\hat{\theta}^{(w,\tau)}$ denote the estimate of parameter $\theta$ on the trailing $w$-day window ending at step $\tau$.  For the weekly stage, steps are placed every 7 days (to keep each DOW's representation stable).  For the annual stage, amplitude is evaluated on fixed-width trailing windows of 1, 2, and 3 years.

**Drift slope.** The drift is the OLS slope of $\{\hat{\theta}^{(w,\tau)}\}_\tau$ over equally spaced step indices $\tau = 0, 1, \ldots, M-1$:

$$\hat{\beta}^{(\text{drift})} = \frac{\displaystyle\sum_{\tau}(\tau - \bar{\tau})(\hat{\theta}^{(w,\tau)} - \bar{\theta})}{\displaystyle\sum_{\tau}(\tau - \bar{\tau})^2}$$

**Classification.** With $\hat{\theta}^{(\text{full})}$ the full-sample estimate:

$$\text{classification} = \begin{cases} \text{"drifting"} & \left|\hat{\beta}^{(\text{drift})}\right| > 0.01 \cdot \left|\hat{\theta}^{(\text{full})}\right| \\ \text{"stable"} & \text{otherwise} \end{cases}$$

The threshold 0.01 corresponds to a drift of 1% of the full-sample effect per step.  For a zero full-sample estimate, an absolute tolerance of $10^{-6}$ is used.

---

## 8. R² Decomposition

`seqd` reports a marginal sequential R² decomposition attributing variance reduction to each stage in extraction order.

Let $\operatorname{Var}(\cdot)$ denote population variance (divisor $n$).  Define:

$$R^2_{\text{weekly}} = 1 - \frac{\operatorname{Var}(y^{(w)})}{\operatorname{Var}(y)}$$

$$R^2_{\text{holiday}} = \frac{\operatorname{Var}(y^{(w)}) - \operatorname{Var}(y^{(h)})}{\operatorname{Var}(y)}$$

$$R^2_{\text{annual}} = \frac{\operatorname{Var}(y^{(h)}) - \operatorname{Var}(\hat{\epsilon})}{\operatorname{Var}(y)}$$

Each value is the fraction of original variance removed at that stage.  Values are clipped to $[0, 1]$.

**Additivity.** Provided no stage clips to zero:

$$R^2_{\text{weekly}} + R^2_{\text{holiday}} + R^2_{\text{annual}} = 1 - \frac{\operatorname{Var}(\hat{\epsilon})}{\operatorname{Var}(y)}$$

This identity holds exactly when every stage weakly reduces residual variance.  The decomposition is order-dependent: the weekly component receives credit for all variance it removes from the raw series, including any incidental annual or trend covariance it absorbs.

---

## 9. API Reference

### `SeqdDecomposer`

```python
SeqdDecomposer(
    holiday_dates,
    multiplicative=None,
    holiday_window=14,
    max_holiday_window=None,
    reference_window=60,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holiday_dates` | `list` or `dict` | required | Holiday dates. Flat list (`["2023-12-25", ...]`) or named dict (`{"Christmas": [...], ...}`). Elements may be `str`, `datetime.date`, or `pd.Timestamp`. A flat list groups dates by calendar (month, day): all years with the same month-day are treated as the same holiday. |
| `multiplicative` | `bool` or `None` | `None` | Force `True` (multiplicative) or `False` (additive). `None` = auto-detect per Section 3.3. |
| `holiday_window` | `int` | `14` | Half-width $W$ of the holiday search window in days. Controls how far before and after the holiday date the ramp detection and ramp-end scan extend. Recommendation: use 35 or more for extended retail events such as Black Friday / Cyber Monday. |
| `max_holiday_window` | `int` or `None` | `None` | When set, overrides `holiday_window` as the effective search window passed to the holiday stage. Useful when `holiday_window` is set to a conservative default but a specific run requires a wider window without changing the base parameter. When `None`, `holiday_window` is used as-is. |
| `reference_window` | `int` | `60` | Length $R$ of the pre/post-holiday baseline window in days. A warning is issued when `len(y) < 2 * reference_window`. |

**`fit(y)`**

| Argument | Type | Constraints |
|----------|------|-------------|
| `y` | `pd.Series` | `DatetimeIndex`, daily frequency (no gaps > 1 day), no duplicate dates, no `NaN` values. Values are cast internally to `float64`. |

Returns `DecompositionResult`.  Raises `ValueError` on constraint violations; issues `UserWarning` when the series is shorter than `2 * reference_window` or shorter than 365 days.

---

### `DecompositionResult`

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `series` | `pd.Series` | Original input series. |
| `weekly` | `WeeklyEffect` | Stage 1 result. |
| `holidays` | `list[HolidayEffect]` | Stage 2 results: one `HolidayEffect` per occurrence (year) per holiday name. |
| `annual` | `AnnualEffect` | Stage 3 result. |
| `residual` | `pd.Series` | Final residual $\hat{\epsilon}_t$ (trend + noise). |
| `r2_by_component` | `dict[str, float]` | Keys `"weekly"`, `"holiday"`, `"annual"`. Values are marginal R² contributions clipped to $[0,1]$. |
| `.weekly_component()` | `pd.Series` | Full-length weekly effect series aligned to original index. Additive mode: DOW coefficients broadcast to each date. Multiplicative mode: $y_t(1 - 1/\hat{w}_d)$, i.e. the absolute amount removed (scales with series level; for a level-independent view use `weekly.coefficients`). |
| `.holiday_component()` | `pd.Series` | Sum of all `HolidayEffect.effect_series`, reindexed with zero fill. |
| `.annual_component()` | `pd.Series` | Annual Fourier component $\hat{S}(t)$ (intercept excluded). |
| `.fitted()` | `pd.Series` | Reconstructed series: `residual + weekly_component() + holiday_component() + annual_component()`. Satisfies exact reconstruction. |

---

### `WeeklyEffect`

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | `np.ndarray` shape (7,) | DOW coefficients $\hat{w}_0, \ldots, \hat{w}_6$ (Monday = 0). Additive: level-independent offsets with zero mean. Multiplicative: scaling factors with unit mean. |
| `is_multiplicative` | `bool` | Mode used for Stage 1. |
| `recency` | `dict[int, pd.DataFrame]` | Keys are window sizes in days: 60, 90, 365 (any window exceeding series length is absent). Each DataFrame has columns `date, dow_0, ..., dow_6`; rows correspond to trailing window endpoints stepped every 7 days, in ascending date order. |
| `drift` | `dict[int, dict]` | Keys are DOW integers 0–6. Each value is `{"slope": float, "classification": "stable" | "drifting"}`. |

---

### `HolidayEffect`

| Attribute | Type | Description |
|-----------|------|-------------|
| `date` | `datetime.date` | The specific holiday occurrence date $h$. |
| `name` | `str` | Holiday name. |
| `ramp_start` | `datetime.date` | Detected effect start $\hat{t}_{start}$. For compound-block members, the block-level merged start. |
| `ramp_end` | `datetime.date` | Detected effect end $\hat{t}_{end}$. For compound-block members, the block-level merged end. |
| `magnitude` | `float` | Mean of `effect_series` over the ramp window. For compound-block members, the block-level mean. |
| `effect_series` | `pd.Series` | Full-length effect series aligned to original index. For the primary member of a compound block: the canonical block effect $\hat{H}^{(\text{block})}(t)$. For non-primary compound-block members: identically zero (prevents double-counting). |
| `year_magnitudes` | `list[float]` | Ordered per-occurrence magnitudes across all years for this holiday name, computed from pre-merge occurrence-level ramp windows. |
| `magnitude_drift` | `float` | OLS slope of `year_magnitudes` over year index $y = 0, \ldots, Y-1$. Zero if $Y < 2$. |
| `compound` | `bool` | `True` if this occurrence was merged into a compound block. |
| `compound_block_id` | `str` or `None` | Shared identifier for all members of the same compound block (e.g. `"compound_block_2023_1"`). `None` for non-compound occurrences. |
| `individual_peak_magnitude` | `float` or `None` | Mean of the pre-merge occurrence effect over $[h-3, h+3]$. Unreliable near series boundaries; see `individual_peak_magnitude_reliable`. |
| `ramp_start_ceiling_hit` | `bool` | `True` if the detected ramp start reached or exceeded the search boundary $h - W$ (i.e. the true ramp may begin earlier than detected). |
| `individual_peak_magnitude_reliable` | `bool` | `False` if the $\pm 3$-day window extends outside the series bounds, or if the occurrence's `ramp_end` is at or beyond the last date in the series (meaning no post-holiday reference exists to anchor the baseline). |

---

### `AnnualEffect`

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_harmonics` | `int` | BIC-selected harmonic count $\hat{K} \in \{0,1,2,3,4,5,6\}$. Zero means no annual seasonality was detected. |
| `coefficients` | `np.ndarray` | Fourier coefficients $[\hat{a}_0, \hat{a}_1, \hat{b}_1, \ldots, \hat{a}_{\hat{K}}, \hat{b}_{\hat{K}}]$. Length $2\hat{K}+1$; fit on the original (non-detrended) $y_t^{(h)}$. |
| `component` | `pd.Series` | The annual component $\hat{S}(t)$ aligned to original index (intercept excluded). |
| `recency_amplitudes` | `dict[int, float]` | Keys 1, 2, 3 (trailing years). Values: $\sqrt{\hat{a}_1'^2 + \hat{b}_1'^2}$ from a $K=1$ fit on the linearly detrended sub-series. `NaN` if the window is shorter than 30 days. |

---

## 10. Known Limitations

**CUSUM threshold is fixed.** The CUSUM threshold $2\hat{\sigma}_{ref}$ and run threshold $0.5\hat{\sigma}_{ref}$ are constants.  For series with heavy-tailed noise the false-positive rate for ramp detection may be elevated; for very smooth series the detector may be over-sensitive.  A data-adaptive threshold derived from the empirical null distribution of the CUSUM statistic would be more principled but is not implemented.

**Fourier phase tied to calendar year of series start.** The time index $t$ is anchored to January 1 of the calendar year in which the series begins, not to January 1 of a fixed reference year.  Two series that start in different calendar years will have $t = 0$ at different absolute dates (e.g. Jan 1 2020 vs Jan 1 2022), so their `annual.coefficients` will carry a phase offset proportional to the difference in start-year day-of-year position.  The `annual.component` series is always correctly aligned to the input index, so reconstruction and prediction within the fitted range are unaffected.

**Annual BIC detrend is linear only.** The detrend applied before BIC selection removes a single OLS linear trend.  For series with exponential or power-law growth, the residual after linear detrending retains curvature that can bias harmonic selection — typically toward higher $\hat{K}$ to absorb the unremoved curvature.  A log-transform or more flexible pre-detrending step applied externally before calling `fit()` will mitigate this.

**`individual_peak_magnitude` unreliable at data boundaries.** For the last holiday occurrence in the series, the post-holiday ramp-end scan may have no data within $W$ days.  The occurrence-level effect series is accordingly truncated, and `individual_peak_magnitude` (which uses a $\pm 3$-day window from that series) may be based on few or no post-holiday observations.  The `individual_peak_magnitude_reliable` flag is set to `False` whenever the $\pm 3$-day window extends outside the series bounds, or whenever `ramp_end >= series.index[-1]` (i.e. the ramp reaches or runs past the last observation, meaning no post-holiday reference data exists to anchor the baseline).

**`holiday_window` default of 14 is too small for extended retail events.** Black Friday / Cyber Monday buildups in retail data routinely begin 3–4 weeks before the date; the post-event hangover can extend 5–7 days.  The default `holiday_window=14` will systematically underdetect the pre-event ramp (CUSUM cannot accumulate sufficient signal beyond 14 days) and will force `ramp_end = h + 14` when no recovery is found.  Values of 35 or greater are recommended for BFCM and similar multi-week promotional events.

**BIC harmonic ceiling of $K = 6$.** The harmonic search is restricted to $K \in \{0,1,2,3,4,5,6\}$.  For most daily business series this ceiling is sufficient; very unusual annual shapes with fine sub-annual structure are the exception.  Residual annual structure visible in `result.residual` after Stage 3 is a signal that the series may require pre-processing or a domain-specific model rather than additional harmonics.  The ceiling is not currently a user-controllable parameter.

---

## 11. Quick Start Example

```python
import pandas as pd
import numpy as np
from datetime import date
from seqd import SeqdDecomposer

# Generate synthetic daily retail-like series
np.random.seed(42)
dates = pd.date_range('2021-01-01', '2024-12-31', freq='D')
t = np.arange(len(dates))
y = pd.Series(
    100 + 0.05 * t                          # trend
    + 15 * np.sin(2 * np.pi * t / 365.25)  # annual
    + 10 * (t % 7 < 5)                      # weekday premium
    + np.random.normal(0, 5, len(t)),        # noise
    index=dates
)

# Add a BFCM-style spike
for yr in [2021, 2022, 2023, 2024]:
    bf_approx = pd.Timestamp(f'{yr}-11-26')
    mask = (dates >= bf_approx - pd.Timedelta(days=10)) & (dates <= bf_approx + pd.Timedelta(days=5))
    y[mask] += 80

holidays = {
    "Black Friday": [date(2021, 11, 26), date(2022, 11, 25), date(2023, 11, 24), date(2024, 11, 29)],
    "Christmas":    [date(yr, 12, 25) for yr in [2021, 2022, 2023, 2024]],
}

result = SeqdDecomposer(
    holiday_dates=holidays,
    holiday_window=35,
    reference_window=60,
).fit(y)

print(result.r2_by_component)
# {'weekly': ..., 'holiday': ..., 'annual': ...}

print(result.weekly.coefficients)
# array of 7 DOW offsets, Monday=0

print(result.residual.describe())
# trend + noise only
```

**Reconstruction identity** — the fitted values exactly reproduce the original series:

```python
import numpy as np
assert np.allclose(result.fitted(), result.series)
```

**Inspecting holiday effects:**

```python
for he in result.holidays:
    print(
        f"{he.name} {he.date}: ramp {he.ramp_start} → {he.ramp_end}, "
        f"magnitude={he.magnitude:.1f}, drift={he.magnitude_drift:.2f}/yr"
    )
```

**Annual seasonality summary:**

```python
ann = result.annual
print(f"Harmonics selected (BIC): {ann.n_harmonics}")
print(f"Recency amplitudes (1/2/3 yr): {ann.recency_amplitudes}")
```

**Weekly drift diagnostics:**

```python
for dow, info in result.weekly.drift.items():
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"{day_names[dow]}: {info['classification']} (slope={info['slope']:.5f})")
```
