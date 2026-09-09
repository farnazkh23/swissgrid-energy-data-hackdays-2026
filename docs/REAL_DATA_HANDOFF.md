# Real-data handoff

This layer is ready for the first mapped dataset. It does not define the
meaning of any target, choose provider column names, or start G3 specialists.
The caller supplies those decisions in configuration.

## Data contract

`ColumnMapping` maps arbitrary provider columns to canonical fields. Required
forecasting fields are:

```text
source_id, country, domain, unit, value,
event_time, valid_time, known_at,
first_received_at, normalized_at,
revision_id, revision_sequence, source_record_id
```

`publication_time`, `supersedes_revision_id`, and issue-time fields are
preserved when mapped. Static values can be supplied as explicit mapping
constants. No provider-specific column names are assumed.

The default `known_at_policy` is `require`. It refuses rows without
`known_at`, `first_received_at`, and `normalized_at`. A historical dataset may
explicitly select `publication_plus_lag`; when `known_at` is absent, this
requires `publication_time` and reconstructs `known_at` as the maximum of
publication plus the configured lag, receipt, and normalization. An available
historical `known_at` remains authoritative. The audit marks reconstructed
rows explicitly. This is a controlled fallback, not a silent inference.

Source configuration also declares timezone, cadence, publication lag, horizon
range, source metadata, and the dataset identity. The resulting observations
are validated by the existing `Observation`, `SourceRegistry`, and
`DatasetContract` contracts.

For raw-byte evidence, pass an existing `RawStore` and an explicit
`raw_received_at` to `load_real_dataset`. The loader stores the exact file
bytes before adaptation and attaches that content digest to each observation;
it never uses file modification time as a receipt clock.

## Supported files

`load_real_dataset` reads CSV and JSON using the standard library. JSON may be
a list of records or an object containing a `records` or `data` list. Parquet
is supported through an explicitly installed pandas or pyarrow dependency; no
dependency is installed automatically.

## Required preflight

`audit_dataset` runs before any model execution and reports row count, date
range, cadence, missingness, duplicate identities, revision frequency and
lineage, timezone/DST review flags, unit consistency, gaps, suspicious future
availability, stale periods, source coverage, and availability at requested
forecast horizons.

The real runner refuses to proceed when PIT metadata, source forecast metadata,
or the explicit target contract is unresolved. The audit also records whether
the publication-lag fallback was used.

`audit_features` reports Pearson, Spearman, deterministic binned mutual
information, fold-safe univariate OOF MAE, incremental OOF gain, fold
stability, missingness, freshness, and registry leakage risk. Its default
decision is `UNDECIDED`. A caller may provide an explicit multi-signal
`recommendation_policy` to produce KEEP/DROP recommendations; correlation is
never sufficient for either decision.

## Baseline run

Construct a `RealChampionConfig` with:

- an explicit `TargetContract` and target source/record identity;
- explicit historical issue times, with the final issue equal to the target
  contract issue time;
- an explicit `RollingOrigin` plan whose holdout starts at that final issue;
- explicit feature source IDs.

Then run:

```python
from swissgrid_forecaster.real_dataset_loader import load_real_dataset
from swissgrid_forecaster.real_source_adapter import RealSourceConfig
from swissgrid_forecaster.run_real_champion import run_real_champion

dataset = load_real_dataset("data/first-drop.csv", source_config=source_config)
result = run_real_champion(dataset, champion_config, output_dir="artifacts/real_run")
```

The module CLI takes a JSON configuration containing `source_config`,
`target_contract`, `plan`, `issue_times`, `target_source_id`,
`target_source_record_id`, and `feature_source_ids`:

```bash
PYTHONPATH=src python3 -m swissgrid_forecaster.run_real_champion \
  --input data/first-drop.csv \
  --config config/real_champion.json \
  --output-dir artifacts/real_run
```

The runner writes `data_audit.json`, `feature_audit.json`, `scoreboard.json`,
and `diagnostics.json`. Baselines are evaluated with rolling OOF predictions;
calibration and holdout rows are not supplied to candidate models. Candidate
ineligibility and unavailable optional dependencies are recorded rather than
given fabricated scores.

Before tomorrow’s run, confirm the source license, timezone/DST convention,
revision lineage, publication/receipt clocks, unit semantics, target meaning,
feature horizon availability, and the split/holdout dates. Real predictive
validity cannot be assessed until those decisions and data are reviewed.
