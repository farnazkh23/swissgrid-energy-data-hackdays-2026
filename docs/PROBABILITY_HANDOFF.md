# Probability-model handoff

The canonical input is [`artifacts/real_backtest/oof_predictions.csv`](../artifacts/real_backtest/oof_predictions.csv).
It contains 2,016 aligned hourly realizations from 12 rolling validation folds;
each fold is a 168-hour validation block. The target window is 2026-05-29
00:00 UTC through 2026-08-20 23:00 UTC.

Every row has one target timestamp, its one-hour `horizon`, the corresponding
`issue_time`, and actual/predicted values for CH, DE, FR, and IT. The four
residual columns are `actual - prediction`. `fold_id`, `week_start`,
`hour_of_day`, and `day_of_week` are included for fold-aware or calendar-aware
diagnostics. No raw provider rows are included.

## Why these residuals are OOF

The handoff uses the validated `persistence` baseline: for a target at time
`t`, the prediction is the latest target value at `t - 1 hour`. A fresh
rolling fit/state is used for each fold, and the prediction for a validation
timestamp is made from information strictly before its `issue_time`.
Validation actuals are joined only after prediction. Thus these are genuine
out-of-fold residuals, not fitted predictions. The export is deterministic and
rejects missing values, target misalignment, duplicate target timestamps, and
duplicate `(fold_id, timestamp)` keys.

The summary file [`artifacts/real_backtest/oof_residual_summary.json`](../artifacts/real_backtest/oof_residual_summary.json)
contains row/fold counts, date range, per-target residual mean and sample
standard deviation, the residual correlation matrix, and missing/duplicate
checks.

## Suggested downstream use

Let `e_t = [CH_residual, DE_residual, FR_residual, IT_residual]` and let
`m_t` be the future point forecast from the production baseline.

For a first multivariate Gaussian calibration, estimate the residual mean
`mu` and sample covariance `Sigma` from `e_t`. Generate scenarios as

```text
y_t ~ Normal(m_t + mu, Sigma)
```

Keep the four residual components together; independently sampling each
country loses the cross-country dependence shown by the correlation matrix.
Check covariance positive-semidefiniteness and apply only a documented,
deterministic numerical regularisation if necessary.

For a Student-t alternative, fit degrees of freedom, location, and scale to
the residual vectors and compare rolling tail coverage, multivariate
log-score, and extreme-event calibration against the Gaussian. For an
empirical bootstrap, resample complete four-dimensional residual vectors and
add them to `m_t`; consider a time/block bootstrap or calendar stratification
if residual autocorrelation or hour-of-week effects are material.

Do not fit or assess the probability model on training residuals. Training
residuals are in-sample and optimistic. Use these OOF residuals for calibration;
for an unbiased final comparison, reserve later folds (or use a nested rolling
calibration/evaluation split) so the residuals used to fit a distribution are
not the same residuals used to report its score.
