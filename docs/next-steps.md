# Next Steps

## 1. Define an Audit Protocol First
- Pick fairness metrics to govern on: signed residual gap, MAE gap, calibration gap.
- Set thresholds before analysis (example: MAE gap <= 1.5, signed residual gap <= 1.0).
- Define minimum support rules (example: group n >= 30; intersectional group n >= 50).
- Add a short "Audit Policy" markdown section in the notebook.

## 2. Add Uncertainty, Not Just Point Estimates
- Bootstrap every fairness metric by group (for example, 1000 resamples).
- Report 95% confidence intervals next to each gap.
- Flag disparities only when confidence intervals exclude 0 and exceed practical thresholds.
- Add a compact table: metric, estimate, CI, threshold, pass/fail.

## 3. Add Temporal Robustness
- Replace random-only split with rolling or forward split by season.
- Evaluate fairness metrics per test-season window.
- Track drift to identify groups that degrade over time.
- Plot fairness-over-time with confidence bands.

## 4. Expand Intersectional Analysis Safely
- Evaluate Region x LeagueTier and Region x PredFeeBucket first.
- For sparse cells, use pooled "Other" buckets or hierarchical shrinkage.
- Report group coverage and excluded cells explicitly.
- Keep paradox checks at both aggregate and stratified levels.

## 5. Add Mitigation Experiments as Controlled Comparisons
- Baseline RF (current).
- Reweighted RF (group x fee-band sample weights).
- Conservative RF (regularization-like settings).
- Post-hoc calibration variants.
- Compare all approaches on one frontier chart: RMSE vs fairness gap vs calibration gap.

## 6. Move to Decision-Ready Reporting
- Build a final audit scorecard with pass/fail by metric.
- Add top risks and top mitigations.
- Add a model card section: intended use, non-use, limitations, monitoring cadence.
- Include a residual-risk statement describing what remains uncertain.

## 7. Operationalize Monitoring
- Create a reusable function that accepts new-season data and outputs the same audit artifacts.
- Add alert rules when fairness metrics cross thresholds.
- Version audit outputs by run date and model version.
