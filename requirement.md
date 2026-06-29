# Requirements Specification

## Project Title
AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

## 1. Objective
Develop a robust AI-driven pipeline that detects periodic dips in noisy light curves, classifies the underlying astrophysical cause, and estimates transit parameters with confidence metrics.

## 2. Scope

### In scope
- Ingest TESS light curve datasets (high-cadence sector data).
- Preprocess and detrend noisy light curves.
- Detect periodic dip candidates.
- Classify dip events into astrophysical classes.
- Estimate transit period, depth, and duration for transit-like events.
- Report confidence/significance for each detected event.
- Produce visualizations for detections and classifications.
- Deliver a concise report (max 3 pages).

### Out of scope (for baseline release)
- Full atmospheric characterization of planets.
- Follow-up spectroscopy and external observational validation.
- Real-time telescope operations integration.

## 3. Data Requirements

### Mandatory datasets
- TESS raw/light curve data from MAST/STScI resources (as provided in problem statement).
- Curated labeled dataset containing known exoplanets, eclipsing binaries, false positives, and related classes.

### Data scale target
- One sector high-cadence data containing approximately 20k-30k light curves.

### Data quality requirements
- Preserve timestamps, flux, and uncertainty columns.
- Apply mission quality flag filtering.
- Maintain metadata for source ID, sector, cadence, and processing steps.

## 4. Functional Requirements

1. Data ingestion module
- Must query and download sector-wise light curves.
- Must support batch processing and resumable downloads.

2. Preprocessing module
- Must remove invalid cadences (NaN, quality-flagged points).
- Must normalize and detrend each light curve.
- Must preserve transit-scale variability.

3. Detection module
- Must identify periodic dip candidates using BLS (minimum baseline).
- Must return candidate period, epoch, duration, and depth proxies.
- Must compute candidate significance score (SNR/log-likelihood/FAP proxy).

4. Feature extraction module
- Must compute period, depth, duration, transit count, odd-even depth difference, and shape metrics.
- Must compute noise statistics on transit timescales.

5. Classification module
- Must classify each candidate into:
  - planet_transit
  - eclipsing_binary
  - blend_or_background
  - stellar_variability_or_other
- Must output class probabilities/confidence.

6. Parameter estimation module
- For likely transit class, must estimate:
  - orbital period
  - transit depth
  - transit duration
- Must provide uncertainty or confidence interval estimates.

7. Visualization module
- Must plot raw and detrended light curves.
- Must plot phase-folded candidate signals.
- Must annotate predicted class and confidence on plots.

8. Output and reporting module
- Must export candidate table (CSV/Parquet) with key features and predictions.
- Must include confidence/significance and estimated transit parameters.
- Must support generation of figures and summary text for final report.

## 5. Non-Functional Requirements

1. Reproducibility
- All runs must be configurable and repeatable with fixed random seeds.
- Processing steps must be logged.

2. Performance
- Baseline run on pilot subset (<=500 light curves) should complete within practical development time.
- Full-sector batch processing should support chunking and parallel execution.

3. Robustness
- Pipeline must handle missing values and uneven cadence gracefully.
- Errors in single target processing must not terminate full batch.

4. Explainability
- Each classification must include top contributing features (or equivalent explanation).

5. Maintainability
- Modular code structure with separate components for ingest, preprocess, detect, classify, and report.

## 6. Suggested Technical Stack
### Backend & Data
- Python 3.11+
- numpy, scipy, pandas
- astropy, lightkurve, astroquery
- scikit-learn, xgboost/lightgbm
- matplotlib/seaborn
- optional advanced: transitleastsquares, pymc, exoplanet
### Frontend
- react js

## 7. Acceptance Criteria

1. Detection completeness
- Pipeline identifies periodic dip candidates from test set and reports significance.

2. Classification quality
- Model produces confusion matrix and class-wise precision/recall/F1.
- Meets agreed baseline performance threshold after pilot benchmarking.

3. Parameter estimation quality
- Period, depth, and duration estimates are reported for transit-like detections.
- Error metrics against known labeled examples are included.

4. Deliverables completeness
- Candidate catalog exported.
- Visualizations generated.
- 3-page report draft generated with methods, assumptions, tools, and uncertainty handling.

## 8. Deliverables
1. Source code for full pipeline.
2. Configuration files and environment requirements.
3. Candidate output table with class and confidence.
4. Diagnostic plots and phase-folded detection visuals.
5. Final report (max 3 pages).

## 9. Milestone Plan

1. Milestone 1: Research and architecture freeze
- Stack finalized, formulas documented, requirements approved.

2. Milestone 2: Baseline pipeline
- Ingest + preprocess + BLS + baseline classifier.

3. Milestone 3: Evaluation and refinement
- Metrics, threshold tuning, confidence calibration, error analysis.

4. Milestone 4: Final packaging
- Outputs, visuals, and final report.

## 10. Assumptions
- Curated labeled dataset is accessible and sufficiently representative.
- Required star metadata for feature engineering is available or can be cross-matched.
- Compute resources are adequate for sector-scale batch processing.

## 11. Dependencies and Risks
- Dependency on external archives availability/network throughput.
- Risk of false positives due to blends/systematics.
- Risk of class imbalance and domain shift from curated labels to science data.

## 12. Traceability to Problem Statement
- Periodic dip detection: covered by Detection module requirements.
- Multi-class categorization: covered by Classification module requirements.
- Science dataset application: covered by Data + Output requirements.
- Significance reporting: covered by Detection and Output requirements.
- Transit parameter estimation: covered by Parameter estimation requirements.
