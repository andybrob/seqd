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
10. [Recommended Configurations](#10-recommended-configurations)
11. [Known Limitations](#11-known-limitations)
12. [Quick Start Example](#12-quick-start-example)

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

where $[\hat{\alpha}_{dt}, \hat{\beta}_{dt}]$ are OLS estimates from regressing $y_t^{(h)}$ on $[1, t]$.  This detrended series is used for **both BIC selection and the final coefficient estimation** (see Section 6.3).

BIC is evaluated over $K \in \{0, 1, \ldots, \texttt{max\_harmonics}\}$ (default range $\{0, 1, 2, 3, 4, 5, 6\}$):

$$\operatorname{BIC}(K) = n \log\!\left(\frac{\operatorname{RSS}(K)}{n}\right) + (2K+1)\log(n)$$

where $\operatorname{RSS}(K) = \sum_{t=0}^{n-1}\!\left(\tilde{y}_t^{(h)} - \hat{\tilde{y}}_t^{(K)}\right)^2$ is the residual sum of squares from fitting $\mathbf{X}(t, K)$ on the detrended series, and $K = 0$ (intercept only, one parameter) is the no-seasonality baseline.  The selected order is $\hat{K} = \arg\min_K \operatorname{BIC}(K)$.  Any $K$ whose fitted values are non-finite is skipped.

### 6.3 Final Estimation

Fit OLS on the **linearly detrended** $\tilde{y}_t^{(h)}$ with the selected $\hat{K}$:

$$\hat{\boldsymbol{\gamma}} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \tilde{\mathbf{y}}^{(h)}, \qquad \hat{\boldsymbol{\gamma}} = [\hat{a}_0,\, \hat{a}_1,\, \hat{b}_1,\, \ldots,\, \hat{a}_{\hat{K}},\, \hat{b}_{\hat{K}}]^\top$$

**Rationale:** fitting on raw $y_t^{(h)}$ with a strong trend embeds the mean level in the intercept, producing a *fixed* seasonal amplitude that over-subtracts in early years and under-subtracts in later years.  Fitting on the linearly detrended series isolates the pure cyclical component so the Fourier coefficients represent deviations from the local trend rather than from the global mean.  This reduces residual autocorrelation at lag 364.

The annual component (intercept excluded) is subtracted from the **original** (non-detrended) series to produce the final residual.  Because the intercept of the Fourier fit on the detrended series absorbs the mean of $\tilde{y}_t^{(h)}$ (not of $y_t^{(h)}$), excluding it from the removed component preserves the correct mean level:

$$\hat{S}(t) = \sum_{k=1}^{\hat{K}} \left[\hat{a}_k \cos\frac{2\pi k t}{P} + \hat{b}_k \sin\frac{2\pi k t}{P}\right]$$

If $\hat{K} = 0$, the annual component is identically zero; only the intercept $\hat{a}_0$ is estimated (and not removed).

The final residual is:

$$\hat{\epsilon}_t = y_t^{(h)} - \hat{S}(t)$$

Note: the stored `AnnualEffect.coefficients` array $[\hat{a}_0, \hat{a}_1, \hat{b}_1, \ldots]$ contains coefficients estimated on the **linearly detrended** series $\tilde{y}_t^{(h)}$.  When projecting the annual component out-of-sample, only the harmonic terms (intercept excluded) are applied.

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
| `max_harmonics` | `int` | `6` | Upper bound on BIC harmonic search: $K \in \{0, 1, \ldots, \texttt{max\_harmonics}\}$. **Recommendation: for series with 3 or more years of daily data, setting `max_harmonics=8` may capture finer within-year structure** (e.g. distinct spring and autumn shoulders) that K=6 cannot represent. Must be >= 0. |

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
| `n_harmonics` | `int` | BIC-selected harmonic count $\hat{K} \in \{0,1,\ldots,\texttt{max\_harmonics}\}$ (default range 0–6). Zero means no annual seasonality was detected. |
| `coefficients` | `np.ndarray` | Fourier coefficients $[\hat{a}_0, \hat{a}_1, \hat{b}_1, \ldots, \hat{a}_{\hat{K}}, \hat{b}_{\hat{K}}]$. Length $2\hat{K}+1$; fit on the original (non-detrended) $y_t^{(h)}$. |
| `component` | `pd.Series` | The annual component $\hat{S}(t)$ aligned to original index (intercept excluded). |
| `recency_amplitudes` | `dict[int, float]` | Keys 1, 2, 3 (trailing years). Values: $\sqrt{\hat{a}_1'^2 + \hat{b}_1'^2}$ from a $K=1$ fit on the linearly detrended sub-series. `NaN` if the window is shorter than 30 days. |

---

## 10. Recommended Configurations

### 10.1 Retail / E-commerce with 3+ Years of Daily Data

```python
result = SeqdDecomposer(
    holiday_dates=holidays,
    holiday_window=50,           # wide window for BFCM buildups
    reference_window=60,
    max_holiday_merge_gap_days=35,
    max_harmonics=8,             # allow finer annual structure on long series
).fit(y)
```

**Why `max_harmonics=8`?** With 3+ years of daily data the BIC has sufficient
observations to distinguish genuine K=7 and K=8 harmonics from noise. These
harmonics represent within-quarter patterns (~46-day period for K=8) that are common
in e-commerce revenue: post-holiday lulls, back-to-school, and mid-summer peaks.
Setting `max_harmonics` above 8 is not recommended (see Known Limitations below).

### 10.2 Adaptive IPM for Recent Momentum

Use `use_adaptive_ipm=True` when the business growth rate has shifted meaningfully in
the past 1–2 years and you want the holiday magnitude projection to reflect that
recency rather than fitting a uniform OLS slope over all historical years:

```python
forecaster = SeqdForecaster(
    result,
    slope_blend_alpha=0.5,
    use_adaptive_ipm=True,       # overrides trend_yoy_blend=0.5, ipm_decay_halflife=1.5
)
forecaster.fit(
    changepoint_penalty_beta=3.0,
    min_segment_size=60,
    aic_linear_delta=2.0,
    enable_bfcm_carveout=True,
)
```

The `use_adaptive_ipm` preset sets `trend_yoy_blend=0.5` (equal blend of OLS and
trend-implied IPM projection) and `ipm_decay_halflife=1.5` (exponential half-life of
1.5 years, giving the most recent holiday year ~50% more weight than the year two
years ago). `slope_blend_alpha` and `ipm_max_years` are not overridden.

---

## 11. Known Limitations

**CUSUM threshold is fixed.** The CUSUM threshold $2\hat{\sigma}_{ref}$ and run threshold $0.5\hat{\sigma}_{ref}$ are constants.  For series with heavy-tailed noise the false-positive rate for ramp detection may be elevated; for very smooth series the detector may be over-sensitive.  A data-adaptive threshold derived from the empirical null distribution of the CUSUM statistic would be more principled but is not implemented.

**Fourier phase tied to calendar year of series start.** The time index $t$ is anchored to January 1 of the calendar year in which the series begins, not to January 1 of a fixed reference year.  Two series that start in different calendar years will have $t = 0$ at different absolute dates (e.g. Jan 1 2020 vs Jan 1 2022), so their `annual.coefficients` will carry a phase offset proportional to the difference in start-year day-of-year position.  The `annual.component` series is always correctly aligned to the input index, so reconstruction and prediction within the fitted range are unaffected.

**Annual BIC detrend is linear only.** The detrend applied before BIC selection removes a single OLS linear trend.  For series with exponential or power-law growth, the residual after linear detrending retains curvature that can bias harmonic selection — typically toward higher $\hat{K}$ to absorb the unremoved curvature.  A log-transform or more flexible pre-detrending step applied externally before calling `fit()` will mitigate this.

**`individual_peak_magnitude` unreliable at data boundaries.** For the last holiday occurrence in the series, the post-holiday ramp-end scan may have no data within $W$ days.  The occurrence-level effect series is accordingly truncated, and `individual_peak_magnitude` (which uses a $\pm 3$-day window from that series) may be based on few or no post-holiday observations.  The `individual_peak_magnitude_reliable` flag is set to `False` whenever the $\pm 3$-day window extends outside the series bounds, or whenever `ramp_end >= series.index[-1]` (i.e. the ramp reaches or runs past the last observation, meaning no post-holiday reference data exists to anchor the baseline).

**`holiday_window` default of 14 is too small for extended retail events.** Black Friday / Cyber Monday buildups in retail data routinely begin 3–4 weeks before the date; the post-event hangover can extend 5–7 days.  The default `holiday_window=14` will systematically underdetect the pre-event ramp (CUSUM cannot accumulate sufficient signal beyond 14 days) and will force `ramp_end = h + 14` when no recovery is found.  Values of 35 or greater are recommended for BFCM and similar multi-week promotional events.

**BIC harmonic ceiling.** The harmonic search ceiling is controlled by `max_harmonics` (default `6`), restricting $K \in \{0,1,\ldots,\texttt{max\_harmonics}\}$.  For most daily business series the default of 6 is sufficient.  **For series with 3 or more years of daily data**, users may set `max_harmonics=8` to allow the BIC to capture finer within-year structure (e.g. distinct spring and autumn shoulders, or sub-quarterly revenue patterns) that the K=6 ceiling cannot represent.  Setting `max_harmonics` above 8 is not recommended: beyond K=8, the Fourier basis becomes dense relative to the annual period (each harmonic covers ~45 days), risking over-fit and reduced out-of-sample accuracy.  Residual annual structure visible in `result.residual` after Stage 3 is a signal that the series may benefit from a higher ceiling or from external pre-processing.

---

## 12. Quick Start Example

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

---

## Stage 4–6: Trend Analysis and Forecasting (V2)

V2 extends `seqd` with three post-processing stages that operate on `result.residual`
— the trend-plus-noise series left by V1. The sequential identification principle is
preserved: V2 consumes V1 output read-only and adds no parameters to the V1 stages.

### Notation (V2 extensions)

The following symbols supplement the V1 notation table in Section 2.

| Symbol | Definition |
|--------|------------|
| $r_t$ | V1 residual series `result.residual`; trend + noise after all V1 stages |
| $n$ | Length of $r_t$ |
| $\tau_1 < \tau_2 < \cdots < \tau_K$ | Changepoint indices (0-based positions in $r_t$); each is the first index of the new segment |
| $\tau_0 = 0$, $\tau_{K+1} = n$ | Sentinel boundaries |
| $\mathcal{S}_j$ | Segment $j$: observations $r_{\tau_{j-1}}, \ldots, r_{\tau_j - 1}$ |
| $n_j$ | Segment length: $n_j = \tau_j - \tau_{j-1}$ |
| $t^{(j)}_i$ | Normalised time within segment $j$: $t^{(j)}_i = (i-1)/(n_j - 1)$, mapping to $[0,1]$ |
| $T_{\text{days}}^{(j)}$ | Segment length in calendar days ($= n_j$ for a contiguous daily series) |
| $f_j(t)$ | Fitted parametric trend for segment $j$ |
| $\hat{\sigma}^2$ | Robust variance of $r_t$: $\hat{\sigma}^2 = \bigl(\text{MAD}(r_t)/0.6745\bigr)^2$ |
| $\lambda$ | Per-changepoint PELT penalty: $\lambda = \hat{\sigma}^2 \cdot \beta \cdot \ln(n)$ |
| $\beta$ | Penalty multiplier (default 3.0) |
| $H$ | Forecast horizon in days |
| $\hat{y}(t)$ | Total point forecast at future date $t$ |
| $\hat{w}_d$ | Weekly DOW coefficient for day $d$ from the 60-day recency window |

---

### Stage 4: Changepoint Detection

#### 4.1 Objective

Find the partition $\{\tau_1, \ldots, \tau_K\}$ of $\{0, \ldots, n-1\}$ that minimises
the penalised segmentation cost:

$$\min_{\tau_1 < \cdots < \tau_K} \left[ \sum_{j=1}^{K+1} C(\tau_{j-1},\, \tau_j - 1) \;+\; K \cdot \lambda \right]$$

#### 4.2 Cost Function

The `"l2"` (least-squares mean-shift) cost for segment $[a, b]$ is:

$$C(a, b) = \sum_{t=a}^{b} \!\left(r_t - \bar{r}_{[a,b]}\right)^2, \qquad \bar{r}_{[a,b]} = \frac{1}{b-a+1}\sum_{t=a}^{b} r_t$$

This is the residual sum of squares from fitting a constant within the segment.
`ruptures` evaluates $C(a,b)$ in $O(1)$ per segment using precomputed prefix sums.

#### 4.3 Penalty

$$\lambda = \hat{\sigma}^2 \cdot \beta \cdot \ln(n)$$

where $\hat{\sigma}^2 = \bigl(\text{MAD}(r_t)/0.6745\bigr)^2$ is the robust
(MAD-based) variance estimate and $\ln(n)$ is the BIC-standard penalty for one
additional parameter. The multiplier $\beta = 3$ (default) corresponds to
$1.5\times$ the two-parameter BIC cost; it suppresses over-segmentation in the
presence of residual autocorrelation.

#### 4.4 Algorithm

PELT (Pruned Exact Linear Time) solves the above minimisation exactly in $O(n)$
amortised time. Minimum segment size is 90 days (≈ one quarter), preventing false
changepoints within sub-annual seasonal cycles.

---

### Stage 5: Piecewise Trend Fitting

#### 5.1 Segment Normalisation

For segment $\mathcal{S}_j$ of length $n_j$, the normalised time coordinate is:

$$t^{(j)}_i = \frac{i - 1}{n_j - 1}, \qquad i = 1, \ldots, n_j, \quad t^{(j)} \in [0, 1]$$

This normalisation is essential for numerical stability: raw day indices produce
design-matrix condition numbers $\sim 10^6$ for 2-year segments.

#### 5.2 Candidate Models

Four parametric families are considered, all with $t = t^{(j)}_i \in [0,1]$:

| Label | Formula | Parameters | Applicability |
|-------|---------|-----------|--------------|
| L (Linear) | $f_L(t) = \alpha + \beta t$ | $k_L = 2$ | Always |
| G (Log) | $f_G(t) = \alpha + \beta \ln(1 + t \cdot T_{\text{days}}^{(j)})$ | $k_G = 2$ | Always; $\ln(\cdot) = 0$ at $t=0$, grows slowly for $t > 1$ |
| E (Exponential) | $f_E(t) = \alpha \exp(\beta t)$ | $k_E = 2$ | Only when all $r^{(j)}_i > 0$ |
| Q (Quadratic) | $f_Q(t) = \alpha + \beta t + \gamma t^2$ | $k_Q = 3$ | Only when $n_j \geq 5$ |

Model G uses the regressor $\ln(1 + t \cdot T_{\text{days}}^{(j)})$, which equals
zero at $t=0$ (so $\alpha$ is the segment-start value) and $\ln(1 + T_{\text{days}})$
at $t=1$. This anchors the curvature to the physical segment length rather than the
abstract $[0,1]$ coordinate.

Model E is fit via log-linearisation: $z_i = \ln r^{(j)}_i = \ln\alpha + \beta t^{(j)}_i$,
then OLS on $(1, t^{(j)})$ against $z$. RSS for AIC is computed on the original scale.

#### 5.3 AIC Selection with Linear Parsimony

For each applicable model $m$:

$$\text{AIC}_m = n_j \ln\!\left(\frac{\text{RSS}_m}{n_j}\right) + 2k_m$$

Identify $m^* = \arg\min_m \text{AIC}_m$, then apply the linear parsimony rule:

$$\hat{m} = \begin{cases} L & \text{if } \text{AIC}_L - \text{AIC}_{m^*} < 2 \\ m^* & \text{otherwise} \end{cases}$$

A difference of less than 2 AIC units provides no meaningful evidence against the
simpler linear model (Burnham & Anderson 2002). Linear extrapolation beyond the segment
boundary is also more stable than exponential or quadratic extrapolation.

#### 5.4 Segment Record and Extrapolation

Each segment produces a `SegmentTrend` storing $(\alpha, \beta, \gamma, T_{\text{days}},
\text{model\_type}, \text{AIC}, \text{RSS})$ and a `t_anchor_date` equal to the
segment's start date. For any date $d$ (in-sample or future):

$$t(d) = \frac{(d - \text{t\_anchor\_date}).\text{days}}{n_j - 1}$$

The reusable function `evaluate_segment(segment, t)` returns $f_j(t)$ for any $t$,
including $t > 1$ for out-of-sample projection.

---

### Stage 6: Out-of-Sample Forecasting

#### 6.1 Trend Projection

Let $\mathcal{S}_{K+1}$ be the final segment (containing the last $n_{K+1}$ in-sample
observations). For forecast step $h = 1, \ldots, H$:

$$t_h = \frac{(n_{K+1} - 1) + h}{n_{K+1} - 1}$$

This maps the segment start to $t = 0$, the last observed day to $t = 1$, and the first
forecast step to $t = n_{K+1}/(n_{K+1}-1) > 1$.

$$\hat{y}_{\text{trend}}(h) = \text{evaluate\_segment}\!\left(\mathcal{S}_{K+1},\; t_h\right)$$

#### 6.2 Annual Projection

Using the same calendar-anchored time index as V1 Stage 3 (anchor = Jan 1 of the series'
first year):

$$t_{\text{future}} = (d_{\text{future}} - t_0).\text{days}$$

$$\hat{y}_{\text{annual}}(d_{\text{future}}) = \sum_{k=1}^{\hat{K}} \left[\hat{a}_k \cos\!\frac{2\pi k\, t_{\text{future}}}{P} + \hat{b}_k \sin\!\frac{2\pi k\, t_{\text{future}}}{P}\right]$$

The intercept $\hat{a}_0$ is excluded (absorbed into trend level). If $\hat{K} = 0$,
the annual component is identically zero.

#### 6.3 Holiday Projection

For each future holiday occurrence, project the magnitude one step ahead using OLS on
the historical `year_magnitudes` series (linear extrapolation even with two observations,
to preserve trend direction). Scale the most recent reliable ramp shape by the projected
magnitude. If the most recent occurrence's shape is unreliable (`ramp_start_ceiling_hit`
or `individual_peak_magnitude_reliable = False`), fall back to the most recent reliable
occurrence.

#### 6.4 Forecast Combination

**This formula is derived from V1's reconstruction identity.** In V1 multiplicative mode,
the weekly component is a multiplicative factor applied to the weekly-adjusted series;
holidays and annual operate additively on that adjusted series. The V2 reconstruction
follows directly:

**Multiplicative mode** (`result.weekly.is_multiplicative = True`):

$$\hat{y}(t) = \hat{w}_{\text{DOW}(t)} \times \Bigl[\hat{y}_{\text{trend}}(t) + \hat{y}_{\text{annual}}(t) + \hat{y}_{\text{holiday}}(t)\Bigr]$$

where $\hat{w}_d$ is the 60-day recency DOW coefficient (unit-mean, $\bar{w} = 1$).

**Additive mode** (`result.weekly.is_multiplicative = False`):

$$\hat{y}(t) = \hat{y}_{\text{trend}}(t) + \hat{w}_{\text{DOW}(t)} + \hat{y}_{\text{annual}}(t) + \hat{y}_{\text{holiday}}(t)$$

where $\hat{w}_d$ is the 60-day recency DOW offset (zero-sum, $\sum_d \hat{w}_d = 0$).

**Why weekly multiplies the sum (not each term independently):** holiday and annual
effects were estimated on the weekly-adjusted series $y_t^{(w)} = y_t / \hat{w}_d$.
They are additive offsets to the adjusted level, not to the raw level. Applying $\hat{w}_d$
multiplicatively to each term independently would double-apply the DOW scaling to
components that already live in the adjusted domain.

---

### V2 API

#### `SeqdForecaster`

```python
from seqd import SeqdForecaster

# Step 1 — fit V1 decomposition as usual
result = SeqdDecomposer(holiday_dates=holidays, holiday_window=50).fit(y)

# Step 2 — construct forecaster and run Stages 4 + 5
forecaster = SeqdForecaster(result)
forecaster.fit(
    changepoint_penalty_beta=3.0,   # β for λ = σ̂² β ln(n); range [1, 6]
    min_segment_size=60,             # minimum segment days; default 60 (v0.2.1+)
    aic_linear_delta=2.0,            # AIC parsimony threshold for linear preference
)

# Inspect changepoints and segment fits after fit()
print(forecaster.changepoints)   # list[pd.Timestamp]
for seg in forecaster.segments:
    print(seg.segment_index, seg.model_type, seg.alpha, seg.beta, seg.aic)

# Step 3 — run Stage 6 and obtain forecast components
fr = forecaster.predict(
    horizon=365,
    future_holidays={"Black Friday": [pd.Timestamp(2026, 11, 27)]},
    max_extrapolation_days=365,
)

# ForecastResult components (all pd.Series with DatetimeIndex)
fr.forecast          # total point forecast
fr.trend_component   # trend extrapolation alone
fr.weekly_component  # DOW factors (multiplicative) or offsets (additive)
fr.annual_component  # Fourier annual projection
fr.holiday_component # projected holiday ramp effects

# Multiple horizons without re-fitting
fr_short = forecaster.predict(horizon=90)
fr_long  = forecaster.predict(horizon=730)
```

#### `forecast_from_result` (convenience)

```python
from seqd import forecast_from_result

fr = forecast_from_result(
    result,
    horizon=365,
    future_holidays={"Black Friday": [pd.Timestamp(2026, 11, 27)]},
    changepoint_penalty_beta=3.0,
    min_segment_size=60,
    aic_linear_delta=2.0,
    max_extrapolation_days=365,
)
```

#### `ForecastResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `pd.Series` | Total point forecast; DatetimeIndex from last date +1 to last date +H |
| `trend_component` | `pd.Series` | Piecewise trend extrapolation |
| `weekly_component` | `pd.Series` | DOW coefficients (multiplicative: unit-mean factors; additive: zero-sum offsets) |
| `annual_component` | `pd.Series` | Fourier annual projection (intercept excluded) |
| `holiday_component` | `pd.Series` | Projected holiday ramp effects; zero where no future holiday provided |
| `changepoints` | `list[pd.Timestamp]` | Stage 4 detected changepoint dates |
| `segments` | `list[SegmentTrend]` | Stage 5 fitted segment records, ordered by `segment_index` |
| `horizon` | `int` | $H$ |
| `is_multiplicative` | `bool` | From `result.weekly.is_multiplicative` |

#### `SegmentTrend` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `segment_index` | `int` | 1-based segment number |
| `start_date`, `end_date` | `pd.Timestamp` | Segment date range |
| `n_obs` | `int` | Number of observations |
| `model_type` | `str` | `"linear"` \| `"log"` \| `"exp"` \| `"quadratic"` \| `"constant"` |
| `alpha`, `beta` | `float` | Intercept and slope parameters |
| `gamma` | `float` or `None` | Quadratic parameter; `None` unless `model_type == "quadratic"` |
| `T_days` | `int` | Segment length in days (used in log model regressor) |
| `aic` | `float` | AIC of selected model |
| `aic_linear` | `float` | AIC of linear fit (always computed; reference for parsimony decision) |
| `rss` | `float` | RSS of selected model |
| `selected_reason` | `str` | `"lowest AIC"` / `"linear preference, ΔAIC=…"` / `"only candidate"` / `"linear only"` |
| `t_anchor_date` | `pd.Timestamp` | = `start_date`; $t(d) = (d - \text{t\_anchor}).\text{days} / (n_{\text{obs}} - 1)$ |
