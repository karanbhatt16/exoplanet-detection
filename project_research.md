# AI-enabled Detection of Exoplanets from Noisy Light Curves

## 1) Problem Framing (What We Are Building)
We need an end-to-end AI pipeline that takes noisy stellar light curves and:

- detects periodic dip-like events,
- classifies them into astrophysical categories (transit, eclipsing binary, blends, stellar variability/other),
- reports event significance/confidence,
- estimates transit parameters for likely planet candidates (period, depth, duration),
- produces interpretable visual outputs and concise reporting.

Target data scale: one TESS sector high-cadence dataset (roughly 20k-30k light curves).

## 2) Data Sources and Practical Access

### Primary data
- TESS data products and catalogs are available via MAST/STScI resources (TIC/CTL page was provided in the problem statement).
- Use SPOC light curves where possible first (quality flags, mission-standard products), then fallback to custom extraction if needed.

### Suggested ingestion APIs
- `lightkurve`: convenient search/download and light curve operations for TESS/Kepler.
- `astroquery.mast`: direct programmatic access to MAST products when finer control is needed.

### Labels for supervised classification
- Use provided curated labeled set (known exoplanets, eclipsing binaries, false positives, blends).
- Keep an explicit train/validation/test split by target ID (not random rows), to avoid leakage across segments from the same star.

## 3) Chosen Tech Stack (Decision)

### Core language and environment
- Python 3.11+
- Jupyter for exploration; Python package/modules for production pipeline.

### Scientific/time-series stack
- `numpy`, `scipy`, `pandas`
- `astropy` (units, time-series, BoxLeastSquares)
- `lightkurve` (TESS light curve handling)

### ML stack
- `scikit-learn` (baseline models, metrics, calibration)
- `xgboost` or `lightgbm` (tabular feature classifier)
- `pytorch` (optional stage-2 1D CNN/Transformer for morphology)

### Modeling and fitting stack
- Detection baseline: `astropy.timeseries.BoxLeastSquares`
- Optional detection upgrade: `transitleastsquares` (TLS; physically-motivated transit shape)
- Parameter inference (advanced): `pymc` + `exoplanet` for uncertainty-aware fits (optional in phase-2)

### Visualization and reporting
- `matplotlib`, `seaborn`, `plotly` (optional interactive)
- `jupyter-book` or plain Markdown + figures for final 3-page report.

### Why this stack
- Fast to prototype, widely used in astronomy, strong scientific credibility, and easy transition from notebook to reproducible scripts.

## 4) Physics and Signal Formulas (Core)

## 4.1 Flux normalization and intensity relations

Given measured flux series $F(t)$, normalize by robust baseline $F_0(t)$ (from detrending):

$$
f(t) = \frac{F(t)}{F_0(t)}
$$

Relative flux variation:

$$
\Delta f(t) = f(t) - 1
$$

Magnitude-flux conversion (if needed):

$$
m_1 - m_2 = -2.5\log_{10}\left(\frac{F_1}{F_2}\right)
$$

## 4.2 Transit depth and radius ratio

For a small planet and weak limb-darkening approximation:

$$
\delta \approx \frac{\Delta F}{F} \approx \left(\frac{R_p}{R_*}\right)^2
$$

Hence:

$$
\frac{R_p}{R_*} \approx \sqrt{\delta}
$$

## 4.3 Periodic box model (BLS)

BLS models transit as a periodic box with parameters $(P, \tau, t_0, \delta)$ where:

- $P$: orbital period,
- $\tau$: transit duration,
- $t_0$: reference/mid-transit time,
- $\delta$: depth.

Weighted in-transit flux estimate:

$$
y_{\mathrm{in}} = \frac{\sum_{\mathrm{in}} y_n / \sigma_n^2}{\sum_{\mathrm{in}} 1/\sigma_n^2}
$$

Weighted out-of-transit estimate:

$$
y_{\mathrm{out}} = \frac{\sum_{\mathrm{out}} y_n / \sigma_n^2}{\sum_{\mathrm{out}} 1/\sigma_n^2}
$$

Gaussian log-likelihood objective (up to additive constant):

$$
\log L(P,\tau,t_0) = -\frac{1}{2}\sum_{\mathrm{in}}\frac{(y_n-y_{\mathrm{in}})^2}{\sigma_n^2}
-\frac{1}{2}\sum_{\mathrm{out}}\frac{(y_n-y_{\mathrm{out}})^2}{\sigma_n^2}
$$

This underpins periodogram scoring for candidate detection.

## 4.4 Transit duration relation (approximate geometry)

Common approximation for total transit duration $T_{14}$:

$$
T_{14} \approx \frac{P}{\pi}
\arcsin\left(
\frac{R_*}{a}
\frac{\sqrt{(1+k)^2-b^2}}{\sin i}
\right), \quad k=\frac{R_p}{R_*}
$$

where $a$ is semi-major axis, $b$ impact parameter, and $i$ inclination.

## 4.5 Kepler relation for consistency checks

$$
a^3 = \frac{G(M_* + M_p)P^2}{4\pi^2} \approx \frac{GM_*P^2}{4\pi^2}
$$

Useful for validating whether fitted $P$, $T_{14}$, and host-star priors are physically consistent.

## 4.6 Signal significance / SNR

A practical transit SNR proxy:

$$
\mathrm{SNR}_{\mathrm{multi}} \approx \frac{\delta}{\sigma_{\mathrm{dur}}}\sqrt{N_{\mathrm{tr}}}
$$

where $\sigma_{\mathrm{dur}}$ is noise at transit-duration timescale (CDPP-like) and $N_{\mathrm{tr}}$ is number of observed transits.

Also compute empirical false alarm probability (FAP) via bootstrap/permutation on detrended residuals.

## 5) Event Taxonomy and Classification Logic

### Target classes (initial)
- `planet_transit`
- `eclipsing_binary`
- `blend_or_background`
- `stellar_variability_or_other`

### High-value discriminative features
- Period, duration, depth, SNR, number of transits.
- Odd-even depth mismatch (binary indicator).
- Secondary eclipse near phase 0.5 (binary/blend indicator).
- Transit shape statistics (U-shape vs V-shape, ingress/egress fraction).
- Local out-of-transit variability amplitude (starspots/rotational modulation).
- Centroid shift / contamination flags if available.
- Data quality and crowding metrics.

### Recommended classifier strategy
1. Stage A (rule-based vetting): remove obvious artifacts and low-significance false alarms.
2. Stage B (ML classifier): gradient-boosted trees on engineered features.
3. Stage C (optional morphology model): 1D CNN on phase-folded snippets for harder edge cases.
4. Calibrate probabilities (Platt/Isotonic) and report confidence with reliability curves.

## 6) Proposed Pipeline Architecture

1. Ingest: download/query light curves + metadata.
2. Quality filtering: remove bad cadences (quality flags, NaN/outliers).
3. Detrending/normalization: remove long-term trends while preserving transit timescales.
4. Search: BLS/TLS periodogram to detect candidate periodic dips.
5. Candidate feature extraction: depth, period, duration, shape, odd-even, secondary, noise metrics.
6. Classification: assign astrophysical class with confidence.
7. Parameter estimation: refine transit parameters with local model fit.
8. Reporting: plots + candidate tables + uncertainty/confidence summaries.

## 7) Data Preprocessing Standards (Important)
- Use robust clipping (e.g., MAD-based) to remove extreme outliers.
- Keep detrending window significantly larger than expected transit duration to avoid signal suppression.
- Preserve per-cadence uncertainties where available.
- Track every transform in metadata for reproducibility.

## 8) Evaluation Plan

### Detection metrics
- Recall at fixed false positive rate.
- Precision-recall AUC for candidate detection.
- Period recovery tolerance (e.g., within 1%).

### Classification metrics
- Macro F1 and per-class recall (important for minority classes).
- Confusion matrix.
- Calibration quality (ECE/Brier score).

### Parameter estimation metrics
- MAE/RMSE for depth, period, duration against known labels/candidates.
- Coverage of uncertainty intervals (if Bayesian fitting used).

## 9) Risks and Mitigations
- Blends/crowding can mimic transits: include contamination and centroid-based features.
- Correlated noise inflates significance: use residual bootstrap and robust noise estimates.
- Class imbalance: weighted losses and stratified evaluation.
- Overfitting to curated data: keep holdout science sector evaluation separate.

## 10) Delivery Plan for This Project

### Phase 1: Research + design (current)
- finalize stack,
- finalize formulas and decision criteria,
- define requirements and success metrics.

### Phase 2: Baseline implementation
- ingest + preprocess + BLS detection + feature model classifier.

### Phase 3: Refinement
- confidence calibration,
- optional TLS and/or Bayesian transit fitting,
- stronger visual diagnostics.

### Phase 4: Final outputs
- ranked candidate catalog,
- plots,
- concise 3-page report.

## 11) Key References to Use in Report
- TESS TIC/CTL and MAST resources (problem-provided STScI page).
- Astropy BoxLeastSquares docs and Kovacs et al. (2002) BLS reference.
- Lightkurve project documentation/repository.
- Transit Least Squares (Hippke & Heller, 2019) for detection improvements.
- exoplanet/PyMC ecosystem for probabilistic parameter estimation (optional advanced stage).

## 12) Immediate Next Actions
1. Set up Python environment and install core packages.
2. Pull a small pilot subset (few hundred light curves) for fast iteration.
3. Implement baseline preprocessing + BLS search notebook.
4. Build first feature table and train a baseline classifier.
5. Evaluate and lock thresholds for confidence reporting.
