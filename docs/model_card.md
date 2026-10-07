# Model card

**Task.** Rank machines by probability of failure in the next 24 hours; flag those above a cost-optimal threshold.

**Data.** Synthetic, 60 machines, 120 days, hourly telemetry, about 2.8 % positive rows. Not representative of any real equipment.

**Models compared.** Never-fails baseline, logistic regression, random forest, HistGradientBoosting, LightGBM, IsolationForest (unsupervised).

**Evaluation.** Chronological blocks (train, calibration, threshold, test). Metrics on the test block: PR-AUC (base rate 0.027), precision@5 per day, recall, total hypothetical cost versus "do nothing" and "inspect all". See `dashboard/demo_data/comparison.csv` for the last run.

**Result in the committed demo (one run, synthetic).** PR-AUC about 0.15 to 0.18 for the real models, against 0.027 for chance. The task is deliberately hard (25 % of failures have no visible degradation, 30 % of degradation episodes are near misses). Logistic regression had the lowest test cost; HistGradientBoosting was the selected champion on calibration PR-AUC.

**Explanations.** SHAP for tree models, coefficients for logistic regression, occlusion as fallback.

**Intended use.** Portfolio demonstration of an ML platform. **Not** for maintenance decisions.

**Known limits.** Synthetic data; hypothetical costs; small number of positives per block, so metrics are noisy; no confidence intervals; no fairness or safety analysis, which a real deployment would need.
