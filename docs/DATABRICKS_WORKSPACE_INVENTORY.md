# Databricks workspace inventory — Swissgrid EDH 2026

Inventory date: 2026-09-10. Scope is the authenticated user’s visible resources relevant to the Energy Data Hackdays / Swissgrid challenge. This was a read-only inspection. No workspace object, table, schema, volume, cluster, warehouse, permission, GOTTI/BELA/DHAM resource, or data value was modified.

## Workspace overview

The workspace exposes `/github/dp-light-edh` and `/Shared/dp-light-edh` as Git folders. `/Repos` is empty through the Repos API, while the Workspace API exposes these Git folders. The current challenge code is in `/github/dp-light-edh/src/edh2026`.

Relevant challenge documentation and code:

- `README.md` — platform overview.
- `src/edh2026/README.md` — forecast target format: 168 rows, four `array<int>` output columns, 300 samples per array, and official scoring behavior.
- `src/edh2026/INPUT_DATA.md` — describes `edh.input`, the 18 generation tables, net positions, NTC, and cross-border exchanges.
- `src/edh2026/trigger_submission.py` — helper that finds and invokes the official evaluation job. Inspected only; not executed.
- `src/edh2026/local_scoring.py` — local validation scorer.
- `src/edh2026/evaluate_submission` — Python notebook used by the official job.
- `/Shared/.bundle/eval26/edh` — deployed submission-evaluation bundle.

The provided input volume is `edh.input.data`. Its top-level entries are `cross_border_exchanges.csv`, `net_positions.csv`, `ntc_month.csv`, and `generation_forecast/`, with country folders `AT`, `CH`, `DE`, `FR`, and `IT`. The generation tree contained 18 technology/total directories and 144 yearly CSV slices (147 files including the three top-level CSVs). Directory names match the 18 table families, with source spelling `DE_ONSHORE_WIND_GENERATION_FOREACAST` and `FR_OFFSHORE_GENERATION_FORECAST` as exposed. The IT-total directory contains a final `202601010000-202701010000` slice and a visible filename with `(1)`. `edh.group_0.data` was empty at inspection time.

## Unity Catalog and data

Visible catalogs were `edh`, `system`, and `samples`. Challenge data is in `edh.input`; `edh.group_0` exists but contained no tables. No challenge output schema was visible.

All 21 challenge tables are managed Delta tables. The SQL aggregate counts below are observed counts from bounded aggregate queries. Catalog `spark.sql.statistics.numRows` values were stale for most generation tables, so both values are retained in `artifacts/databricks_inventory.json`.

### Dataset summary

| Table | Coverage | Rows | Time range | Cadence summary | Missingness / duplicates | PIT |
|---|---|---:|---|---|---|---|
| `edh.input.cross_border_exchanges` | Directed flows AT-CH, AT-IT, CH-DE, CH-IT, DE-AT, FR-CH, FR-DE, FR-IT | 284,092 | 2019-01-01 to 2026-08-20 23:45 | Median 15 min; max observed gap 75 min | No nulls; 16,380 duplicate timestamp keys | CURRENT_ONLY |
| `edh.input.net_positions` | AT, CH, DE, FR, IT | 271,803 | 2019-01-01 00:15 to 2026-08-20 23:45 | Median 15 min; max observed gap 75 min | No nulls; 4,092 duplicate timestamp keys | CURRENT_ONLY |
| `edh.input.ntc_month` | CH↔AT, CH↔DE, CH↔FR, CH↔IT | 67,103 | 2019-01-01 to 2026-08-27 23:00 | Hourly observed; max gap 2 h | No nulls; 7 duplicate timestamp keys | PIT_WEAK |
| AT wind/solar | AT onshore wind and solar | 268,340 each | 2019-01-01 to 2026-08-27 23:45 | Mostly 15 min; 15 non-15-min steps | `actual`: 673 nulls; `mtu_end`: 15; otherwise complete | PIT_WEAK |
| AT total | AT total | 84,278 | 2019-01-01 to 2026-08-27 23:45 | Mixed; median 1 h, 61,341 non-15-min steps | `actual_generation`: 673; `mtu_end`: 15 | PIT_WEAK |
| CH wind/solar | CH onshore wind and solar | 67,073 each | 2019-01-01 to 2026-08-27 23:00 | Hourly; 14,400 s max gap | CH wind/solar `current` almost/all null; `intraday` heavily null | PIT_WEAK |
| CH total | CH total | 67,073 | 2019-01-01 to 2026-08-27 23:00 | Hourly | `actual_generation`: 168; `scheduled_consumption`: all 67,073 | PIT_WEAK |
| DE wind/solar | DE offshore/onshore wind and solar | 268,340 each | 2019-01-01 to 2026-08-27 23:45 | Mostly 15 min; 15 non-15-min steps | `actual`: 673 each; DE onshore `current`: 192; DE solar `current`: 288 | PIT_WEAK |
| DE total | DE total | 110,552 | 2019-01-01 to 2026-08-27 23:45 | Mixed; 52,587 non-15-min steps | `actual_generation`: 673; `scheduled_consumption`: all null | PIT_WEAK |
| FR wind/solar | FR offshore/onshore wind and solar | 67,073 each | 2019-01-01 to 2026-08-27 23:00 | Hourly | Offshore vintage fields are heavily null; onshore/solar have material missingness | PIT_WEAK |
| FR total | FR total | 84,278 | 2019-01-01 to 2026-08-27 23:45 | Mixed; median 1 h, 61,341 non-15-min steps | `actual_generation`: 786; `generation_forecast`: 218; `scheduled_consumption`: all null | PIT_WEAK |
| IT wind/solar | IT offshore/onshore wind and solar | 110,552 each | 2019-01-01 to 2026-08-27 23:45 | Mixed; median 15 min / p90 1 h | Offshore vintage fields all null; onshore/solar materially incomplete | PIT_WEAK |
| IT total | IT total | 124,740 | 2019-01-01 to 2026-08-27 23:45 | Mixed; one very large gap of 365.5 days; 22,938 duplicate keys | `actual_generation`: 1,346; `generation_forecast`: 364; `scheduled_consumption`: all null | PIT_WEAK |

The detailed per-table schema, null counts, exact cadence percentiles, catalog metadata counts, coverage, and PIT basis are in [`artifacts/databricks_inventory.json`](../artifacts/databricks_inventory.json).

### Full schema inventory

#### Cross-border, positions, and capacity

- `edh.input.cross_border_exchanges`: `Zeitstempel TIMESTAMP`, `AT-CH DOUBLE`, `AT-IT DOUBLE`, `CH-DE DOUBLE`, `CH-IT DOUBLE`, `DE-AT DOUBLE`, `FR-CH DOUBLE`, `FR-DE DOUBLE`, `FR-IT DOUBLE`.
- `edh.input.net_positions`: `Zeitstempel TIMESTAMP`, `AT DOUBLE`, `CH DOUBLE`, `DE DOUBLE`, `FR DOUBLE`, `IT DOUBLE`.
- `edh.input.ntc_month`: `Zeitstempel TIMESTAMP`, `CH_AT__MW_ DOUBLE`, `AT_CH__MW_ DOUBLE`, `CH_DE__MW_ DOUBLE`, `DE_CH__MW_ DOUBLE`, `CH_FR__MW_ DOUBLE`, `FR_CH__MW_ DOUBLE`, `CH_IT__MW_ DOUBLE`, `IT_CH__MW_ DOUBLE`.

All columns in the three tables are nullable according to Unity Catalog metadata. NTC units are indicated by `_MW_` in the column names. The other two tables do not declare units in the visible schema or challenge documentation.

#### Generation technology tables

Each of the following has the same schema: `mtu_start TIMESTAMP`, `mtu_end TIMESTAMP`, `area STRING`, `actual DOUBLE`, `current DOUBLE`, `day_ahead DOUBLE`, `intraday DOUBLE`.

- `edh.input.generation_forecast_at_onshore_wind`
- `edh.input.generation_forecast_at_solar`
- `edh.input.generation_forecast_ch_onshore_wind`
- `edh.input.generation_forecast_ch_solar`
- `edh.input.generation_forecast_de_offshore_wind`
- `edh.input.generation_forecast_de_onshore_wind`
- `edh.input.generation_forecast_de_solar`
- `edh.input.generation_forecast_fr_offshore_wind`
- `edh.input.generation_forecast_fr_onshore_wind`
- `edh.input.generation_forecast_fr_solar`
- `edh.input.generation_forecast_it_offshore_wind`
- `edh.input.generation_forecast_it_onshore_wind`
- `edh.input.generation_forecast_it_solar`

`actual`, `current`, `day_ahead`, and `intraday` are value/vintage-like fields. They are not issue-time fields. The challenge documentation declares generation units as MW.

#### Generation total tables

Each of the following has: `mtu_start TIMESTAMP`, `mtu_end TIMESTAMP`, `area STRING`, `actual_generation DOUBLE`, `generation_forecast DOUBLE`, `scheduled_consumption DOUBLE`.

- `edh.input.generation_forecast_at_total`
- `edh.input.generation_forecast_ch_total`
- `edh.input.generation_forecast_de_total`
- `edh.input.generation_forecast_fr_total`
- `edh.input.generation_forecast_it_total`

The visible documentation does not separately define `scheduled_consumption`; it is fully null in CH, DE, FR, and IT and non-null in AT.

## Temporal coverage and PIT suitability

Generation `mtu_start`/`mtu_end` are treated as candidate valid-time interval fields based on their names and the MTU documentation. `Zeitstempel` is treated only as an observation/valid-time candidate. No table contains an explicit issue time, publication timestamp, revision ID, or forecast-run/version field.

Classification:

- `PIT_STRONG`: none.
- `PIT_RECONSTRUCTABLE`: none.
- `PIT_WEAK`: `ntc_month` and all generation forecast tables. Forecast-like labels or a month-ahead concept exist, but release timing cannot be reconstructed safely.
- `CURRENT_ONLY`: `net_positions` and `cross_border_exchanges`, which expose observed values at timestamps but no vintage dimension.
- `UNKNOWN`: none; the absence of release metadata is reported explicitly rather than inferred away.

Delta table property `version` was visible as storage metadata (version 2); it is not a forecast revision field and is not used as PIT metadata.

## Country and zone coverage

The generation tables cover AT, CH, DE, FR, and IT. Technology coverage is:

- AT: onshore wind, solar, total.
- CH: onshore wind, solar, total.
- DE: offshore wind, onshore wind, solar, total.
- FR: offshore wind, onshore wind, solar, total.
- IT: offshore wind, onshore wind, solar, total.

Net positions cover AT, CH, DE, FR, IT. Cross-border exchanges cover the eight directed columns listed above. NTC covers both directions on CH-AT, CH-DE, CH-FR, and CH-IT.

## Compute resources

Two active clusters were visible; neither was started, resized, or altered.

| Cluster | State | CPU/GPU | Runtime | Node type | Workers | Cores | Memory |
|---|---|---|---|---|---:|---:|---:|
| `group-0-cluster` | RUNNING | CPU | `16.4.x-scala2.12` | `Standard_D8s_v3` | 1 | 16 | 65,536 MB |
| `group-0-gpu-cluster` | RUNNING | GPU | `16.4.x-gpu-ml-scala2.12` | `Standard_NV36ads_A10_v5` | 0, single-node | 36 | 450,560 MB |

No accessible cluster policies or instance pools were returned. Serverless compute is enabled on the SQL warehouse below.

## SQL warehouses

One warehouse was visible:

- `Serverless Starter Warehouse`, state `RUNNING`, size `Small`, type `PRO`, Photon enabled, serverless compute enabled, one cluster, auto-stop 10 minutes, health `HEALTHY`.

The bounded profiling SELECTs used this existing warehouse and did not change its configuration.

## Jobs and pipelines

One job was visible: `edh26_submission_evaluation` (`job_id=1109071846433762`). It is a multi-task bundle job with task `evaluate_submission`, using notebook `/Workspace/github/dp-light-edh/src/edh2026/evaluate_submission.py` and an existing cluster ID `0713-163208-hm6glrk`. No schedule field was exposed and no run history was returned by the inspected list-runs call. The bundle state is under `/Shared/.bundle/eval26/edh`.

No Lakeflow pipelines were returned. The challenge helper `trigger_submission.py` can call the evaluation job and enforce the official submission flow; it was not run. `local_scoring.py` is available for validation-table scoring without official submission.

## MLflow, models, and serving

No MLflow experiments, legacy registered models, or `edh` Unity Catalog registered models were visible. The feature-store API probe did not expose a tables endpoint, and no feature tables appeared in `edh.input`.

Serving inventory exposed Databricks-managed foundation-model endpoints only. No Swissgrid forecasting model endpoint was visible.

## Immediately useful data for Swissgrid forecasting

1. Use `net_positions` and `cross_border_exchanges` as historical network-state and directed-flow features after resolving duplicate timestamps.
2. Use `ntc_month` as capacity context, with explicit handling of its hourly cadence and weak PIT status.
3. Use generation actuals and forecast-like fields for AT/CH/DE/FR/IT renewable and total-generation covariates, with country-specific missingness and cadence handling.
4. Keep CH and FR technology series at their observed hourly resolution unless a documented resampling policy is established.
5. Do not use `current/day_ahead/intraday` as leakage-safe vintages until issue/publication timing is documented.

## Missing data and external data worth acquiring

Missing or weakly represented resources are: a clearly identified Swissgrid target/release series; explicit forecast issue/publication times and revision IDs; load/demand; weather/NWP; hydro; outages/availability; prices; and capacity-outage/release data. External acquisition should prioritize those sources, especially target data with release timestamps, ENTSO-E/Swissgrid forecast vintages, load, weather, outages/availability, hydro, and prices/spreads.

## Recommended ingestion order

1. Preserve raw volume files and add retrieval time, source path, and content hash.
2. Normalize timestamps to UTC while retaining source-timezone assumptions.
3. Resolve duplicate keys with a documented rule; specifically investigate the 22,938 Italy-total duplicates.
4. Acquire or add issue/publication timestamps and revision IDs; fail closed for PIT joins when absent.
5. Build audited features from net positions, exchanges, NTC, and generation actuals/forecast-like fields.
6. Add target/release data, load, weather, outages/availability, hydro, and prices.

## What could and could not be inspected

Inspected successfully: visible workspace paths, Git folders, challenge files and documentation, catalogs and schemas, all visible `edh.input` table schemas, bounded counts/time ranges/cadence/null/duplicate aggregates, volumes and filenames, active clusters, SQL warehouse, policies, pools, jobs, job detail and run history, pipelines, MLflow experiments, model registries, serving endpoints, and bundle state/config files.

Not fully inspected: unrelated/private `/Users` content; full raw table scans; raw data rows; credentials/tokens; inaccessible or non-exposed feature-store endpoints; and any resource outside the authenticated user’s visibility. No commit was made.
