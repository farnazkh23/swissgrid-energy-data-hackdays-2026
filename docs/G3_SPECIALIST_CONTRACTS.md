# G3 specialist and evaluation contracts

This branch freezes the interfaces needed to add real specialists after data
inspection. It does not implement specialist science, Final Brain, dynamic
trust, calibration, ensemble weighting, API changes, real-data adapters, or
historical GOTTI/BELA/DHAM changes.

## Scope and invariants

The G3 boundary is deliberately narrow:

- specialists emit compressed typed briefs, never raw-agent transcripts;
- self-declared confidence is descriptive metadata, not trust or an ensemble
  weight;
- a specialist cannot assign itself an ensemble weight;
- a forecast cannot claim calibration without attached OOF calibration
  evidence;
- every evidence brief has an explicit `known_at` cutoff;
- every target forecast has an explicit `fit_cutoff`;
- future-informed or malformed briefs fail closed with `ValueError`;
- a specialist may emit only an `EvidenceBrief`; a target forecast is not
  mandatory;
- ablation and contribution claims are OOF-only, never in-sample claims.

These contracts are evidence carriers and validation boundaries. They do not
decide trust, combine forecasts, promote a champion, or infer target/sign/unit
semantics.

## `EvidenceBrief`

`EvidenceBrief` is for a specialist that explains state or evidence and need
not forecast total net position. It contains:

| Field | Meaning |
| --- | --- |
| `specialist_id` | Stable registered producer ID |
| `issue_time` | Forecast/evidence issue clock, normalized to UTC |
| `horizon_applicability` | Positive horizons to which this evidence applies |
| `evidence_family` | Registered family, such as `Demand` or `Market` |
| `state_features` | Compact named scalar state/features |
| `direction`, `tendency` | Optional descriptive directional signals |
| `uncertainty` | Compact descriptive uncertainty metadata |
| `freshness` | Optional nonnegative age of the evidence |
| `missingness` | Named missing inputs or evidence gaps |
| `source_ids` | Source identifiers used by the specialist |
| `known_at` | Required knowledge cutoff; it must be at or before `issue_time` |
| `quality_flags`, `warnings` | Descriptive quality and caveat strings |
| `provenance` | Compact string metadata for lineage |

State/features and structured uncertainty are scalar key/value pairs (a single
numeric uncertainty is also accepted as the `value` entry) so the brief stays
compact and serializable. Raw records, arbitrary objects, and transcripts are
not part of the interface.

## `TargetForecastBrief`

`TargetForecastBrief` is for a specialist directly forecasting the configured
target. It contains:

| Field | Meaning |
| --- | --- |
| `specialist_id` | Stable registered producer ID |
| `issue_time`, `target_time`, `horizon` | Explicit UTC chronology; horizon must match the two clocks |
| `model_id`, `version` | Forecast model identity |
| `point_prediction` | Optional point forecast |
| `p10`, `p25`, `p50`, `p75`, `p90` | Optional ordered quantile values |
| `samples` | Optional finite empirical samples |
| `uncertainty` | Compact descriptive uncertainty metadata |
| `source_manifest_ids`, `feature_manifest_id` | Source/feature lineage IDs; at least one is required |
| `fit_cutoff` | Required model knowledge/training cutoff; it must be at or before `issue_time` |
| `oof_metrics` | Optional OOF evaluation metrics |
| `calibration_claimed` | Explicit claim bit, requiring attached evidence |
| `calibration_evidence` | OOF folds, metrics, and knowledge clock supporting the claim |
| `warnings`, `provenance` | Caveats and compact lineage metadata |

At least one of a point, quantile, or samples representation is required.
Crossed quantiles, nonfinite values, inconsistent horizons, missing manifests,
future fit cutoffs, and future-known calibration evidence fail closed.

Calibration evidence is represented by `CalibrationEvidence` and must include
an evaluation ID, nonempty OOF fold IDs, metrics, and a `known_at` clock no
later than the forecast issue. The presence of uncertainty metadata alone does
not establish calibration or trust.

## Registry and runner

`SpecialistDefinition` registers metadata only: ID, family, version, allowed
brief types, and positive horizon applicability. The registry supports these
future families without implementing their science:

`History/Analogue`, `Demand`, `Weather/Renewables`, `Generation/Outage`,
`Hydro/Storage`, `Neighbour/Flow`, `Market`, `News/Event`, and `Tail/Extreme`.

IDs are unique even across versions. The registry has no weight, trust, or
promotion operation.

`SpecialistRunner` receives a `SpecialistRunContext` and invokes a future
specialist's `run`, `produce`, or callable interface. It accepts only a typed
`EvidenceBrief` or `TargetForecastBrief`, verifies registered identity/family,
brief type, issue clocks, requested horizon, and the context knowledge cutoff.
Raw dictionaries and transcripts are rejected. `run_many` returns briefs in
deterministic specialist-ID order.

## Ablation and contribution evidence

`AblationRecord` compares a baseline model with a removed model or specialist.
`ContributionRecord` records a component's comparison contribution. Both
support:

- baseline model ID and optional version;
- removed/component ID and explicit kind (`model` or `specialist`);
- metric name and finite `metric_delta`;
- nonempty OOF fold IDs;
- positive horizon;
- optional nonnegative uncertainty around the delta;
- optional regime and dependency/correlation group;
- `KEEP`, `DROP`, or `UNDECIDED` plus supporting evidence;
- evaluation ID and an explicit delta definition.

Their `evaluation_scope` is fixed to `OOF`. Any in-sample scope is rejected,
and a `KEEP`/`DROP` decision must carry evidence. The contract records the
comparison and does not infer whether a positive delta is good: callers must
state the metric and delta convention. Dependency grouping is metadata for
future Final Brain/trust work, not dynamic trust or a weighting rule.

## Verification

Run from the repository root without installing dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```
