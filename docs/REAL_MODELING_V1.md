# Real Modeling V1

V1 extends the real weekly benchmark with a local-only LSEG adapter and a
CPU tree model. It does not modify the historical GOTTI/BELA/DHAM estate.

## Data visibility and licensing

The worktree exposes the licensed package through the ignored symlink
`data/lseg/raw` → `/home/farnaz/swissgrid-forecaster/data/lseg/raw`. A different
location can be supplied with `SWISSGRID_LSEG_RAW` or `--lseg-root`. Raw LSEG
bytes are never copied into Git, derived artifacts contain no raw LSEG rows,
and no raw file is modified.

## Selected LSEG evidence

The loader selects only these machine-readable families:

| Family | Files used | PIT treatment |
| --- | --- | --- |
| FR/IT day-ahead prices | `01_CORE_HISTORICAL/LSEG_PRICES/FR_DAY_AHEAD_HOURLY_AND_15M.xlsx`; `IT_DAY_AHEAD_HOURLY_AND_15M.xlsx` | Hourly RICs; conservative known-at assumption is 12:00 Europe/Paris on the preceding local day. Only lagged values with `known_at <= issue_time` enter features. |
| Italy demand | `01_CORE_HISTORICAL/LSEG_DEMAND/IT_DEMAND_HOURLY_2021_2025.xlsx` | Observed hourly values; only causal lags enter features. |
| French nuclear PIT | `EDF_NUCLEAR_PIT_DAILY_2021_2026.csv`; `EDF_NUCLEAR_OUTAGE_REVISIONS_PIT.csv` | `issue_time_utc` / `publication_time_utc` are explicit PIT clocks. Revision deltas are aggregated only from publications already visible at issue time. |
| EDF/REMIT events | `EDF_TRANSPARENCY_REMIT_MESSAGES_RAW.csv` | `published` is used as known-at. Only event counts are used; free-text start/stop intervals are not interpreted. |

Rejected from model features: current/recent weather, residual-load and
wind/solar reports without historical issue clocks; CH/DE price blocks for
this first panel; raw French outage history without the revision-as-of
contract; cost-curve snapshots; PDFs, screenshots, and cross-border graph
exports. They remain inventory evidence only. The loader records each
selection/rejection in `lseg_inventory.json`.

## PIT and folds

Targets are the four complete UTC hourly means of CH, DE, FR, and IT net
positions. Duplicate quarter-hour observations or missing quarters invalidate
the hour. The canonical run uses the same 12 rolling folds as the baseline:
672 training hours followed by exactly 168 forecast hours, stepping one week.
No future-valid LSEG value is used without a publication clock no later than
the fold issue time.

The Databricks-only and Databricks+LSEG scenarios are both written. Weekly
score is the point MAE proxy because the official scorer namespace is not
available locally; the official scorer is not invoked and no submission slot is
consumed.

## Models and outputs

The benchmark compares seasonal persistence, Ridge, and a dependency-free
histogram gradient booster (24 shallow CPU boosting stages). Each forecast
also receives exactly 300 deterministic integer samples from a past-only
fold residual pool. The scoreboard includes per-target MAE, RMSE, bias, P10/P90
coverage, sharpness, weekly-score mean/median/worst-week/std, and fold count.

Run with the complete Databricks SQL-Statement JSON result and its chunks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m swissgrid_forecaster.real_modeling_v1 \
  --net-json /tmp/swissgrid_net_positions_query.json \
  --net-chunks /tmp/swissgrid_net_positions_chunk_1.json /tmp/swissgrid_net_positions_chunk_2.json \
  --cross-json /tmp/swissgrid_cross_query.json \
  --cross-chunks /tmp/swissgrid_cross_chunk_1.json /tmp/swissgrid_cross_chunk_2.json \
  --ntc-json /tmp/swissgrid_ntc_query.json \
  --lseg-root data/lseg/raw \
  --max-folds 12 --output-dir artifacts/real_modeling_v1
```

Artifacts are derived only: `target_manifest`, `fold_manifest`,
`lseg_inventory`, `scoreboard`, `ablation`, `diagnostics`, and `provenance` in
JSON plus compact CSV score/ablation tables.

## Retention rule

LSEG features are candidates, not presumed improvements. A family is retained
in the production feature set only if it improves matched rolling OOF MAE and
weekly score robustly across folds and does not weaken PIT evidence. The
canonical 12-fold window begins in January 2019, whereas the selected price,
demand, and daily nuclear blocks begin in 2021; therefore those families are
unavailable in the initial window. Any observed family gain in the report is
review-only until an availability-aligned rolling window confirms it.

## Canonical run result

The run used 66,926 complete hourly rows from 2019-01-01 01:00 UTC through
2026-08-20 23:00 UTC and 12 folds. Aggregate point MAE (MW) was:

| Scenario/model | CH | DE | FR | IT |
| --- | ---: | ---: | ---: | ---: |
| Databricks seasonal persistence | 1,066.21 | 3,858.52 | 3,053.65 | 930.69 |
| Databricks Ridge | 1,251.48 | 2,926.30 | 3,161.64 | 1,259.34 |
| Databricks histogram booster | 1,202.33 | 2,973.02 | 3,331.02 | 1,237.69 |
| Databricks + LSEG seasonal persistence | 1,066.21 | 3,858.52 | 3,053.65 | 930.69 |
| Databricks + LSEG Ridge | 1,252.46 | 2,919.86 | 3,153.51 | 1,262.20 |
| Databricks + LSEG histogram booster | 1,202.33 | 2,973.02 | 3,327.72 | 1,237.69 |

Coverage, sharpness, weekly score summaries, fold-level scores, and the exact
ablation values are in `artifacts/real_modeling_v1/scoreboard.csv` and
`ablation.csv`. The initial-window retention decision is to keep the LSEG
loader and ablation machinery, but promote no LSEG feature family into the
default model until a 2021+ availability-aligned OOF window demonstrates a
stable gain.

Family-level MAE gain (positive means lower MAE; CH/DE/FR/IT order) is:

| Family | Ridge gain | Booster gain | Decision |
| --- | --- | --- | --- |
| FR/IT prices | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | No eligible 2019 rows |
| Italy demand | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | No eligible 2019 rows |
| French nuclear PIT/revisions | -0.98 / +6.43 / +8.13 / -2.86 | 0 / 0 / +3.30 / -0.00 | Mixed and too small to promote |
| EDF/REMIT events | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | No eligible 2019 rows |

Seasonal persistence is unaffected by feature families. These values are
availability-resolved on the canonical window; the machine-readable artifact
contains the full per-target rows and weekly-score deltas.
