# G4 probabilistic sampling and dependency foundation

This branch implements the uncertainty/dependency/sampling layer that turns a
point forecast into the competition's required probabilistic output. It does
not implement point forecasting (Champion/specialist layers, G2/G3), does not
choose the official target contract, and does not implement
`edh2026.local_scoring`. It consumes out-of-fold (OOF) residuals only.

## Scope

`Y = mu + epsilon`. `mu` comes from the Champion/specialist layer (G2/G3, not
this branch). This branch estimates `epsilon` — its bias, variance, and
cross-target dependency — from OOF residuals only, and turns a fitted model
plus a `mu` vector into exactly `n_samples` deterministic integer draws per
target. It also backtests and compares candidate uncertainty methods; it does
not decide which one ships without human review of that comparison.

Real G2 OOF data for the four challenge targets is not available yet. The
module is source-agnostic: `ResidualObservation.from_oof` bridges an existing
single-target `OOFResult` (already implemented in G2) into this layer, and
`ResidualPanel.from_target_residuals` inner-joins per-target residual streams
on shared `(issue_time, target_time)` pairs once four such streams exist. All
tests here run on synthetic, seeded residuals with a known covariance
structure, standing in for the still-pending real OOF panel.

## Modules

| Module                  | Responsibility                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uncertainty_model.py`  | `ResidualObservation`, `ResidualPanel`, `UncertaintyModel`, `fit_uncertainty_model`, `sample_distribution`, `calibrate_scale`, and residual diagnostics |
| `sampler_evaluation.py` | `weekly_folds`, `evaluate_sampler` (walk-forward backtest), `compare_samplers`, `capture_sharpness_curve`                                               |

Each has a matching `tests/test_<module>.py`.

## Residual evidence

`ResidualObservation` is one OOF residual (`truth - point forecast`) for one
target/timestamp/fold, with the same chronology guard used elsewhere in the
codebase (`horizon == target_time - issue_time`). `ResidualPanel` holds joint
residual vectors across targets, built by an inner join on shared timestamps
across each target's OOF residuals — rows without a residual for every target
are dropped, never imputed. `ResidualPanel.identity_hash()` gives a
content-addressed identity for provenance, matching the `identity()` /
`validate_digest()` convention used by `feature_registry.py` and
`manifest.py`.

## Uncertainty methods

`fit_uncertainty_model(panel, method=..., fit_cutoff=...)` implements all six
requested methods over a common bucketed bias/covariance representation:

- `independent_gaussian` — single bucket, diagonal covariance only.
- `correlated_gaussian` — single bucket, full covariance.
- `horizon_covariance` — one bucket (mean + covariance) per distinct horizon.
- `hour_of_day_covariance` — one bucket per target-time hour (0-23).
- `empirical_bootstrap` — same bucketing as correlated Gaussian, but sampling
  draws raw residual vectors with replacement instead of a parametric fit.
- `student_t` — full covariance plus a caller-supplied `degrees_of_freedom`
  (must exceed 2 for finite variance), sampled as a normal/chi-square mixture.

Fitting fails closed on: rows issued after `fit_cutoff` (no future leakage),
non-positive-semi-definite covariance (validated eagerly via a Cholesky
factorization at fit time, not first at sampling time), and any bucket with
fewer than two OOF rows.

## Sampling

`sample_distribution(model, means, bucket_key=..., seed=..., n_samples=300)`
draws exactly `n_samples` joint vectors deterministically (seeded
`random.Random`, no global RNG state) from the fitted bucket, adds them to the
caller's `means` vector, and returns one integer tuple per target. Unseen
bucket keys fail closed rather than falling back to a default distribution.
Joint draws use a manual Cholesky factorization (pure Python, no numpy
dependency, consistent with this project's zero-runtime-dependency stance) so
correlation between targets is preserved in every draw, not applied
independently per target.

## Calibration

`calibrate_scale(model, calibration_panel, target_coverage=...)` fits a single
per-bucket variance-scale multiplier by bisection search against empirical
coverage, using only a **separate** calibration-fold panel — never the rows
used to fit `model`, and never the rows used later to evaluate it. This maps
directly onto the existing `Fold.calibration` partition already produced by
`splits.RollingOrigin.split`, so the three-way separation (fit / calibrate /
evaluate) that the rest of this codebase already enforces for point models
applies here too. Bias correction does not need a separate mechanism: the
fitted `bias` vector already is the OOF residual mean per bucket, applied
during sampling.

## Evaluation

`sampler_evaluation.evaluate_sampler` runs an **expanding-window walk-forward**
backtest over consecutive weekly folds (`weekly_folds`, default 168 rows):
week _i_ is scored using a model fit only on weeks `0..i-1`. This is
deliberately not leave-one-week-out — training on weeks that occur after the
held-out week would leak future information the same way an in-sample
covariance would. The first week is training-only history, matching the
challenge's unseen-future-week evaluation.

Each `WeeklyScore` reports capture (empirical coverage of the fitted central
interval), sharpness (mean interval width), MAE, bias, and per-target/per-
horizon breakdowns. `evaluate_sampler` accepts an optional `score_fn(capture,
sharpness, mae, rows)`; pass the official `edh2026.local_scoring`-derived loss
once available. Without one, a documented placeholder composite loss is used
— treat its numbers as directional only, not official.

`compare_samplers` attaches `KEEP` / `DROP` / `UNDECIDED` to each method
relative to a baseline (default `independent_gaussian`) from paired weekly
score deltas only, reusing the `KEEP_DROP` vocabulary from `ablation.py`.
`capture_sharpness_curve` sweeps a variance-scale multiplier to show the
capture/sharpness tradeoff described in the mission brief.

## What is not decided here

- Which of the six methods becomes the shipped Champion sampler — that is an
  output of `compare_samplers` plus human review on real OOF data, not an
  automatic promotion.
- The official score formula — `_default_score` is a placeholder loss and is
  explicitly documented as such.
- Whether weekly folds should be calendar-week-aligned rather than row-count
  chunks of `week_hours` — the current implementation assumes contiguous
  hourly cadence in the supplied panel; revisit once real cadence is confirmed.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

On Windows, `raw_store.py`'s content-addressed writes now work (see the
Windows portability fix in the same area of history), but every `put()` does
a real `fsync`, which is materially slower on Windows than Linux; the full
suite takes minutes rather than seconds there. This module's own tests run in
well under a second since they do not touch `RawStore`.
