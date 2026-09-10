# Energy Data Hackdays 2026 — Swissgrid team

This repository is the Python foundation for our Swissgrid forecasting team. It
contains PIT data contracts, audits, rolling validation, baseline models, typed
specialist interfaces, forecast artifacts, and local serving/acceptance checks.
Challenge data and the submission job remain in the Databricks workspace.

## The challenge in one page

We must produce a one-week forecast submission for four challenge-defined
targets. Their identities, units, sign conventions, issue schedule, and
aggregation rules are not frozen here. Do not infer them from column names or
fixtures: target semantics are **confirmed only by the Databricks challenge
documentation/data owner**.

The submission shape is confirmed:

| Field | Requirement |
| --- | --- |
| Rows | Exactly 168 rows, in the challenge timestamp order |
| Time field | One timestamp column |
| Forecast fields | Exactly four forecast columns; the final challenge-defined names are still under investigation |
| Values | Each forecast field is `array<int>` |
| Samples | Exactly 300 non-null integer samples in every array |
| Submission | Use `edh2026.trigger_submission.submit_prediction_table` from Databricks |

The four forecast-column names above are placeholders, not names to implement.
The repository's `TargetContract` stays configurable until the contract is
resolved.

### Evaluation strategy

- Use `edh2026.local_scoring` against a local/validation table while iterating.
- Trigger official submission from Databricks with
  `edh2026.trigger_submission.submit_prediction_table`.
- The official cooldown is 30 minutes; final evaluation is on an unseen/random
  week, so rolling validation and PIT discipline matter more than leaderboard-only tuning.
- Keep exact target names and scoring details in the Databricks challenge README
  once confirmed; this repository does not invent them.

## Intended architecture

```text
Databricks + external evidence
            → PIT normalization
            → feature/state layer
            → Champion + specialists
            → probabilistic distribution
            → 300 samples
            → weekly validation/local scoring
            → Databricks submission
```

The diagram is the team architecture, not a claim that every block is finished.
Today the code is strongest in contracts, evidence lineage, audits, rolling OOF,
baseline candidates, and artifact validation. The Databricks table adapter and
production multi-target workflow still need assembly around the confirmed target contract.

## What is implemented now

The implemented modules are deliberately fail-closed where target meaning or PIT evidence is missing:

- **Evidence/source foundation:** `observation_schema`, `availability`, `asof_resolver`, `raw_store`, `manifest`, `ingestion`, `source_contracts`, `source_registry`, `dataset_contracts`, `target_contract`, and `target_config` normalize UTC clocks, retain revision lineage, resolve evidence as-of an issue time, and preserve exact raw bytes plus receipt manifests.
- **Real-data handoff/audits:** `real_dataset_loader` reads CSV/JSON and optional Parquet; `real_source_adapter` maps provider fields; `data_quality`, `data_audit`, `feature_audit`, and `readiness` check missingness, cadence, duplicates, coverage, clocks, freshness, PIT risk, and feature evidence.
- **Features/evaluation:** `feature_registry` and `fold_features` provide versioned fold-local transforms. `splits` implements rolling origins with purge, embargo, label delay, and holdout isolation; `oof`, `metrics`, and `baseline_runner` produce/check OOF evidence.
- **Champion candidates:** `champion` contains persistence, seasonal persistence, historical-conditional empirical samples, and optional lazy scikit-learn Ridge/quantile-boosting adapters. Selection is an OOF recommendation; there is no promotion mechanism.
- **Runs/acceptance:** `run_real_champion` performs the audited real-data handoff; `run_manifest` records content-addressed identity; `acceptance` fails closed on missing PIT evidence, provenance, or hashes.
- **Specialist boundary:** `specialist_contracts`, `specialist_registry`, and `specialist_runner` validate typed briefs; `ablation` and `contribution` record OOF comparison evidence. These are interfaces, not specialist science.
- **Artifacts/serving:** `forecast_output`, `histogram`, `api_models`, `api_schema`, `artifact_repository`, `forecast_service`, `health`, and optional `api` validate or expose persisted artifacts without refitting.
- **Development fixture:** `mock_sources`, `pipeline`, and `dry_run` provide a deterministic synthetic end-to-end run for contract/integration testing.

Not implemented: the final target mapping, four-target production model, real
specialist models, Final Brain, dynamic trust, ensemble weighting, copulas,
calibration science, news/grid-physics modeling, and the Databricks submission
wrapper. The G3 contracts do not imply those capabilities exist.

## Databricks challenge data

The inspected workspace places challenge data in catalog `edh`, schema
`input`, as 21 managed Delta tables. Coverage is AT/CH/DE/FR/IT.

| Family | Available data |
| --- | --- |
| Network state | `edh.input.net_positions` for AT, CH, DE, FR, IT |
| Directed exchanges | `edh.input.cross_border_exchanges` for AT-CH, AT-IT, CH-DE, CH-IT, DE-AT, FR-CH, FR-DE, FR-IT |
| Capacity | `edh.input.ntc_month` for both CH↔AT, CH↔DE, CH↔FR, and CH↔IT directions |
| Generation | 18 country/technology/total tables, with wind, solar, and total-generation coverage |

The generation tables cover: AT (onshore wind, solar, total); CH (onshore
wind, solar, total); DE (offshore wind, onshore wind, solar, total); FR
(offshore wind, onshore wind, solar, total); and IT (offshore wind, onshore
wind, solar, total).

Technology tables expose `mtu_start`, `mtu_end`, `area`, and value-like
`actual`, `current`, `day_ahead`, and `intraday` fields. Total tables expose
`actual_generation`, `generation_forecast`, and `scheduled_consumption`.
Generation units are documented as MW. The network tables use timestamps and
country/link columns; visible schemas do not declare their units.

PIT warning: the challenge tables do not expose explicit issue/publication
times, revision IDs, or forecast-run versions. Treat net positions/exchanges as
current-only and NTC/generation forecast-like fields as PIT-weak until release
timing is documented. Resolve duplicates/cadence explicitly. Never modify `edh.input`.

## LSEG and external evidence

The retained external package is inventoried at metadata level in
[`artifacts/lseg_data_catalog.csv`](artifacts/lseg_data_catalog.csv) and
compared with Databricks in
[`artifacts/lseg_databricks_gap_matrix.csv`](artifacts/lseg_databricks_gap_matrix.csv).
The high-level picture is:

- historical France/Italy day-ahead price blocks, with CH/DE hourly files present
  but CH/DE 15-minute coverage still incomplete;
- Italy national and seven-zone demand;
- French nuclear outage history, including a publication-aware PIT revision
  block, plus EDF transparency/REMIT messages;
- current/recent weather, residual-load, wind/solar supply, and cost-curve reference exports;
- cross-border screenshots/reports that provide context, not reliable row-level flow data.

This evidence complements Databricks network, capacity, and generation data but
does not create a uniform PIT panel. Remaining gaps include Swiss hydro/storage,
historical PIT-safe weather vintages, load vintages outside Italy, broader
outage/availability data, historical forwards, and release metadata for challenge fields.

Licensed raw files are intentionally local-only:
`data/lseg/raw/` is gitignored and must not be committed, copied into a PR, or
reproduced in documentation. Commit metadata catalogs, gap analyses, and
derived summaries only when licensing permits. Do not commit secrets,
`.env` files, `.databrickscfg`, tokens, keys, or private workspace exports.

## Getting started

```bash
git clone <repository-url>
cd swissgrid-forecaster
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package declares Python `>=3.11` and has no mandatory runtime dependency.
Optional scikit-learn estimators are imported lazily and are not installed by
the project. Run the tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Useful module entrypoints that exist today:

```bash
# Deterministic synthetic integration run
PYTHONPATH=src python3 -m swissgrid_forecaster.dry_run \
  --issue-time 2026-01-10T00:00:00+00:00 --output-dir artifacts/mock_run

# Readiness, real-data baseline, and persisted-artifact acceptance
PYTHONPATH=src python3 -m swissgrid_forecaster.cli readiness --config configs/run.json
PYTHONPATH=src python3 -m swissgrid_forecaster.cli run-champion --config configs/run.json
PYTHONPATH=src python3 -m swissgrid_forecaster.cli acceptance --run artifacts/real_run

# Direct real-data runner, when its explicit config contract is ready
PYTHONPATH=src python3 -m swissgrid_forecaster.run_real_champion \
  --input data/first-drop.csv --config config/real_champion.json \
  --output-dir artifacts/real_run
```

`configs/*.example.json` files are templates with `REPLACE_` values; they are
not ready-to-run challenge configurations. Start with
[`docs/TOMORROW_REAL_DATA_RUNBOOK.md`](docs/TOMORROW_REAL_DATA_RUNBOOK.md) and
[`docs/REAL_DATA_HANDOFF.md`](docs/REAL_DATA_HANDOFF.md) before using real data.

## Databricks workflow and submission

The repository remains a normal installable Python package. Databricks is the
execution/submission environment, not a reason to turn the project into a
notebook-only codebase.

1. Use the Databricks Git folder/workspace checkout and read the challenge
   `src/edh2026/README.md` and `src/edh2026/INPUT_DATA.md`.
2. Read challenge inputs from `edh.input` as read-only.
3. Build the final 168-row table with exactly four `array<int>` columns and
   exactly 300 non-null samples per row/array.
4. Write team-owned intermediate/final output under the assigned group schema
   (the inspected `edh.group_0` was empty at inventory time; confirm the
   team's current schema/table names in Databricks).
5. Run local validation with `edh2026.local_scoring` against an appropriate
   validation table.
6. From Databricks, call
   `edh2026.trigger_submission.submit_prediction_table` for the official
   submission, observing the 30-minute cooldown.

The helper and evaluation job were inspected but not executed during the
workspace inventory. Keep submission code and workspace identifiers in the
Databricks workstream; do not add challenge credentials or raw tables here.

## Independent team workstreams

Contributions can proceed independently when they preserve the contracts:

- data audit and feature investigation;
- external-data gap filling and PIT metadata;
- rolling weekly validation and regime diagnostics;
- submission-table validation, local scoring, and Databricks orchestration;
- visualization and demo material;
- modeling experiments and candidate comparisons.

## Contribution rules

- Work on a branch and keep commits/workstream scope small.
- Keep licensed raw data, secrets, and private workspace exports out of Git.
- Preserve PIT discipline: every feature must be known by its historical issue
  time, with source/revision clocks and lineage retained.
- Use rolling weekly/OOF validation; do not optimize only for a leaderboard
  week or leak the final/random evaluation week.
- Run `edh2026.local_scoring` before an official submission.
- Do not modify `edh.input`.
- When changing a shared contract, config shape, or artifact schema, update its
  tests and the relevant documentation so another workstream can rebase safely.

## Documentation map

- [Databricks workspace inventory](docs/DATABRICKS_WORKSPACE_INVENTORY.md) —
  tables, coverage, PIT limitations, jobs, and workspace findings.
- [Databricks local scoring runbook](docs/DATABRICKS_SCORING_RUNBOOK.md) —
  official local scorer adapter, Delta writer, validation, and notebook commands.
- [LSEG vs Databricks comparison](docs/LSEG_DATABRICKS_DATA_COMPARISON.md) —
  external inventory, overlaps, and acquisition gaps.
- [G0/G1 foundation](docs/G0_G1_FOUNDATION.md) and
  [G1 data foundation](docs/G1_DATA_FOUNDATION.md) — evidence, targets,
  ingestion, and source contracts.
- [G2 Champion foundation](docs/G2_CHAMPION_FOUNDATION.md) — features,
  rolling OOF, candidates, metrics, and holdout rules.
- [G3 specialist contracts](docs/G3_SPECIALIST_CONTRACTS.md) — typed brief,
  registry, runner, ablation, and contribution boundaries.
- [Real-data handoff](docs/REAL_DATA_HANDOFF.md) and
  [real-data runbook](docs/TOMORROW_REAL_DATA_RUNBOOK.md) — preflight and
  baseline execution.
- [Mock end-to-end pipeline](docs/MOCK_END_TO_END_PIPELINE.md) — synthetic
  integration path and artifact files.
- [Backend acceptance gate](docs/BACKEND_ACCEPTANCE_GATE.md) and
  [forecast API contract](docs/API_CONTRACT.md) — persisted artifact checks
  and read-only serving.
