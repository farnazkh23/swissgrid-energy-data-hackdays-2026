# G2 Champion modeling and evaluation foundation

This branch implements only the modeling/evaluation foundation. Ingestion,
source adapters, G0 target semantics, and historical GOTTI/BELA/DHAM code are
unchanged. G3 has not started. No dependency is installed and no model is
promoted automatically.

## Files

All modules are under `src/swissgrid_forecaster/`, with matching
`tests/test_<module>.py` files:

| Module | Responsibility |
| --- | --- |
| `feature_registry.py` | Immutable typed feature definitions, audits, dependency DAG and content identity |
| `fold_features.py` | Point-in-time elapsed-time transforms and training-only scaler |
| `splits.py` | Samples, data manifests, rolling origins and isolated holdout access |
| `metrics.py` | Point, quantile, interval and binary-probability metrics |
| `model_contracts.py` | Fit/predict contexts, prediction schema and estimator guards |
| `champion.py` | Baseline candidates, optional estimators and OOF recommendation |
| `oof.py` | Fresh-per-fold execution and checked prediction evidence |
| `model_registry.py` | Immutable model entries and explicit lifecycle restrictions |

## Feature definitions and identity

`FeatureDefinition` contains `feature_id`, `feature_family`, `feature_name`,
`country_scope`, `source`, `known_at_rule`, `horizon_available`, `transform_name`,
`transform_version`, `parent_features`, `units`, `leakage_risk`,
`missingness_policy`, and `enabled`. Supported horizons are explicit positive
elapsed durations. Missingness policies are `propagate` and `error`.

Definition versions are SHA-256 hashes of canonical JSON metadata, including
transform version, parents, units and enabled status. Registry identity sorts
features by ID and includes every definition version; input registry ordering
does not affect it. Tuple ordering within a definition is identity-bearing.
Feature IDs must be unique, parents must exist, and cycles (including self
cycles) fail. `topological_order()` puts parents before children deterministically.
`known_at()` takes the maximum of a feature's and all ancestors' knowledge clocks.

`FeatureAudit` is separate evidence keyed by feature version and evaluation ID:
Pearson (`pearson`), Spearman (`spearman`), `mutual_information`,
`univariate_score`, `oof_gain`, `stability`, and `keep_drop`. Editing an audit
cannot change feature identity. These are evidence fields, not implementations
of feature selection or statistical estimation. Rule/source strings are explicit
metadata; callers must execute their configured availability policies.

## Fold-local transforms

`TimeTransform.fit(train_rows, cutoff)` accepts only training evidence available
by cutoff. Its artifact records fit cutoff, feature version, training row IDs,
and source versions. Failed refits invalidate previous artifacts. Transforms
filter future knowledge and future events before inspecting eligible series
values. Callers supply one already point-in-time-resolved scalar series; duplicate
eligible event times fail to prevent accidental mixing of vintages or series.

- Lag uses an exact `issue_time - window` match; no nearest-row substitution.
- Rolling mean, population standard deviation, min and max use `(issue-window,
  issue]`. Missing values propagate; they are never silently dropped.
- Rate of change requires exact endpoints and returns change per elapsed second.
- Missingness flag reports absent/latest-missing input as 1, otherwise 0.
- Freshness is age in seconds since the latest eligible event, including an
  explicitly missing event; no event gives `None`.
- Calendar emits UTC weekday (Monday=0), hour and month at issue time.

No window uses a row count. All input clocks must be timezone-aware and normalize
to UTC. No civil-time seasonality or Swiss calendar is assumed. The optional
`FoldStandardScaler` computes mean and population standard deviation from its
training rows only; a constant series uses scale 1. It rejects missing training
values and preserves missing prediction values. There is no global normalization
or implicit imputation. Feature engineering remains an explicit caller step:
construct each sample's features as of its issue time, using fold-local artifacts.

## Rolling-origin boundaries

`Sample` records unique row ID, issue/target times, feature/label knowledge times,
numeric feature tuple, optional target and partition. Forecast target time must
follow issue time; label knowledge cannot precede target time. This label rule
models realized outcomes, not advance schedules used as features.

For origin `O`, purge `P`, label delay `L`, and embargo `E`:

1. Training issues are in `[O-P-train_window, O-P)`, with target strictly before
   the window end and label knowledge at or before fit cutoff `O-P-L`.
   Training features must already have been known at their own historical issue.
2. Validation issues are in `[O, O+validation_window)`, with targets strictly
   before that window's end.
3. Calibration begins after the validation window plus `E`; targets must finish
   strictly before its end. These rows are reserved and unused by G2 models.
4. The next origin is at least `O+step`, and also late enough that its fit cutoff
   follows the previous calibration end by `E`. Thus `step` is a minimum stride;
   small requested steps are extended to preserve embargo and nonoverlap.
5. No evaluation window may extend past `holdout_start`. Any pre-holdout issue
   whose target reaches the holdout is excluded. `holdout()` is a separate,
   explicit accessor for issues at or after that boundary.

Historical validation/calibration observations may enter a later fold's training
set only after these temporal and label-availability restrictions permit them.
This is rolling-origin backtesting, not permanently withheld calibration across
all origins. The holdout is permanently excluded. Splitting checks row IDs and
clocks without hashing or using holdout target values. Fold IDs hash the UTC
origin and complete split configuration, independently of row ordering and target
values. Data manifests separately hash all training values, features and clocks.

## Model and prediction contracts

Every candidate implements `fit(train_X, train_y, context)` and
`predict(X, context)`, carrying `model_id`, `version`, `fit_cutoff`,
`feature_manifest_hash`, and `training_data_manifest_hash`. Unfitted provenance
is `None`; completed fits carry concrete hashes and cutoff.

`FitContext` verifies the manifested training rows, partitions, historical feature
availability, label availability and cutoff. Fit arrays must match those rows
exactly. Predict rejects changed feature manifests, wrong feature widths,
future-known features and a fit cutoff later than issue. No target unit or sign
is inferred. Separate immutable predictions represent scalar points,
probabilities, ordered quantiles, or equal-weight empirical distribution samples.
Missing/nonfinite predictions and crossed quantiles fail.

Candidates:

- `Persistence`: latest label in the fold's fitted training history. History is
  fixed for that fold; prediction does not ingest new labels or update state.
- `SeasonalPersistence(season)`: exact elapsed-time target-minus-season lookup
  in fitted history. Missing seasonal history fails without fallback.
- `HistoricalConditional(condition_columns)`: empirical training-label samples
  grouped by explicitly selected feature columns. Empty columns select an
  unconditional distribution. `None` is an explicit conditioning value; unseen
  groups fail. No binning or target-derived group fitting is implicit.
- `OptionalEstimator('ridge')`: scikit-learn Ridge, if installed.
- `OptionalEstimator('quantile_boosting', quantile=...)`: scikit-learn gradient
  boosting with quantile loss and fixed random seed, if installed.

Optional estimators import lazily, require complete features, and never install
packages. Algorithm versions include explicit hyperparameters; reproducible
cross-environment deployment must additionally pin the optional library/runtime
versions. G2 does not serialize estimator objects or implement deployment.

## OOF evidence and recommendation

`run_oof(plan, samples, model_factories, feature_manifest_hash)` constructs a
fresh, unfitted model for each candidate and fold. Reused objects and duplicate
model IDs fail. Fit metadata is checked against the expected context. Prediction
receives one validation sample at a time, with its target removed and label clock
redacted to target time; later-issue features cannot leak through a mixed-time
prediction batch. Neither calibration nor holdout enters model calls.

Each `OOFRecord` contains fold ID, row ID, issue/target times, model ID/version,
feature manifest hash, training cutoff, training data manifest hash and typed
prediction values (including quantile levels when applicable). `OOFResult`
reconstructs the declared rolling plan from its evidence and verifies fold,
validation membership, timestamps, manifests, candidate identities and uniqueness.
These checks detect inconsistent evidence; they are not cryptographic signatures
or a sandbox for arbitrary third-party estimator code.

`select_champion(result, metric=...)` accepts validated rolling OOF evidence only.
Every declared candidate must cover every validation case exactly once. Missing
cases or an entirely missing candidate fail. Scores pool observations equally;
ties break by model ID/version. Explicit objectives are MAE, RMSE, pinball at a
specified quantile, or Brier. For point objectives empirical samples reduce to
their mean. Incompatible prediction types fail. The return value is
`((model_id, version), scores)` and is a recommendation only. Labels are scored
offline after release; this is not an evaluation-availability scheduler.

`ModelRegistry` rejects duplicate IDs even across versions. Its states are
`CANDIDATE`, `CHAMPION`, `REJECTED`, `FROZEN`. New registrations must be candidates;
explicit rejection/freezing is supported, and these states are terminal in G2.
Promotion raises `NotImplementedError`. Artifact identity includes model/version,
cutoff and both manifests; lifecycle state is deliberately separate.

## Metrics and G0 configuration

MAE, RMSE, forecast-minus-truth bias, pinball loss, inclusive interval coverage,
mean interval width (sharpness), and binary Brier score are implemented. Inputs
must be finite, nonempty, aligned and explicitly complete. Crossed intervals,
invalid quantiles and invalid probabilities fail. CRPS and WIS raise
`NotImplementedError`; no proxy scores are returned.

The existing `TargetContract` remains configurable and unchanged. Target name,
unit, sign, aggregation, horizon, resolution, release policy, conditioning,
season and evaluation objective must be supplied for the actual experiment.
Run one target contract per evaluation and preserve it alongside the result.
G2 does not choose a Swissgrid sign convention or final physical target.

## Verification

Run from the repository root, without dependency installation:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Result on Python 3.14: **73 tests discovered; 72 passed, 1 skipped**. The skip is
execution of the optional sklearn estimators because sklearn is absent; explicit
missing-dependency behavior is tested. Adversarial coverage includes future
knowledge, elapsed windows, missingness, invalid refits, dependency cycles,
holdout isolation, purge/embargo boundaries, label cutoff, manifest tampering,
OOF membership/completeness, hidden validation labels, future-batch leakage,
model identity, registry transitions and metric edge cases.
