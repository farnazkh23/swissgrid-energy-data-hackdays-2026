# LSEG vs Databricks data comparison

Inventory date: 2026-09-10. Scope: read-only inspection of `/home/farnaz/swissgrid-forecaster/data/lseg/raw` plus the existing `artifacts/databricks_inventory.json`. No raw LSEG file, Databricks object, compute resource, model, or job was modified; no model was trained.

## Executive summary

LSEG supplies the strongest external blocks for France/Italy day-ahead prices, Italy national and seven-zone demand, French nuclear outage revisions with publication history, and EDF transparency/REMIT messages. It also contains current/recent weather, residual-load, wind/solar supply, and cost-curve reference exports. Databricks supplies the model-ready challenge panel for generation actual/forecast fields, net positions, directed cross-border exchanges, and CH-border NTC. The datasets are complementary by information family, but neither side provides a uniform point-in-time panel across all target countries.

## Workspace/file overview

- Raw files inspected: **200** (100 data/document/image files and 100 `Zone.Identifier` sidecars).
- Exact duplicate content groups: **13**; duplicate paths remain catalogued but are not counted as new information.
- Existing Databricks inventory: **21** managed Delta tables in `edh.input`, covering generation, net positions, exchanges and NTC.
- The raw tree currently contains `CH_DA_RAW.xlsx` and `DE_DA_RAW.xlsx`, despite stale recovery notes saying they were absent; `DE_15M_RAW.xlsx` is still not present.

## Information-family gap matrix

See the complete machine-readable matrix in [`artifacts/lseg_databricks_gap_matrix.csv`](../artifacts/lseg_databricks_gap_matrix.csv).

| Family | Classification | Practical conclusion |
|---|---|---|
| weather | **LSEG_ONLY** | Historical PIT-safe NWP/weather vintages remain missing |
| load_demand | **LSEG_ONLY** | CH/DE/FR/AT historical load, especially forecast vintages, remain missing |
| residual_load | **LSEG_ONLY** | Historical PIT residual-load vintages for CH/DE/FR/IT remain missing |
| hydro_storage | **STILL_MISSING** | Swiss hydro reservoir, inflow, pumped-storage and availability series |
| outages_availability | **LSEG_ONLY** | Non-French fleet, cross-border and renewable availability remain missing |
| prices_day_ahead | **LSEG_ONLY** | Historical DE/CH 15-minute price panel and vintages remain incomplete |
| prices_forwards | **STILL_MISSING** | Historical power forwards/short-dated curves with as-of timestamps |
| generation | **BOTH** | Historical PIT generation forecast vintages remain missing |
| wind | **COMPLEMENTARY** | PIT-safe historical regional wind forecast vintages |
| solar | **COMPLEMENTARY** | PIT-safe historical solar forecast vintages |
| net_positions | **DATABRICKS_ONLY** | Independent historical release/vintage metadata for net positions |
| cross_border_flows | **DATABRICKS_ONLY** | Reliable PIT historical schedules/flows for CH links and publication times |
| NTC_ATC_capacity | **COMPLEMENTARY** | Reliable historical ATC/NTC vintages for all relevant borders |
| forecast_vintages_revisions | **COMPLEMENTARY** | Release-time metadata for DB forecasts and historical LSEG weather/residual vintages |
| news_events | **LSEG_ONLY** | Broader market/news/events and structured event taxonomy |
| country_zone_coverage | **COMPLEMENTARY** | Uniform CH/DE/FR/IT/AT historical load, weather, outages, prices and vintages |

## LSEG file inventory

The complete per-file catalog is [`artifacts/lseg_data_catalog.csv`](../artifacts/lseg_data_catalog.csv), with a JSON version at [`artifacts/lseg_inventory.json`](../artifacts/lseg_inventory.json). It includes filename, format, topic, geography, columns/variables, units, timestamps, date ranges, resolution, actual/forecast classification, PIT fields, missingness, overlap, unique information, hashes and duplicate groups.

### Key tabular LSEG blocks

| Block | Files/coverage | Observed profile | PIT suitability |
|---|---|---|---|
| FR/IT day-ahead prices | FR and IT workbooks; CH/DE hourly files also present at raw root | Timestamped hourly and/or 15-minute RIC panels; price units likely EUR/MWh but not explicitly declared | PIT_RECONSTRUCTABLE; publication schedule is not stored in workbooks |
| Italy demand | National hourly and seven zones at 15 minutes | Timestamped historical demand RIC panels; units likely MW but not explicitly declared | PIT_RECONSTRUCTABLE; no issue timestamp |
| French nuclear | Revision/event CSV, daily PIT feature CSV, raw EDF history, REMIT/transparency CSV | Publication, version, outage interval and availability fields in PIT block; daily horizons h0/h24/h48/h72/h168 | PIT_STRONG for revision/daily PIT block; raw disclosures vary |
| Weather/residual/supply/cost curves | XLS/PDF reference exports and reports | Current/recent views with operator/ensemble labels and retrieval/update context; not a historical row-level panel | CURRENT_ONLY or reference-only |
| Cross-border reference | PNG/PDF graphs | Visual selectors for schedules, capacities and net imports; no reliable row-level values | PIT_WEAK/reference-only |

## Databricks comparison and PIT suitability

The existing Databricks inventory reports:

- `edh.input.net_positions`: AT/CH/DE/FR/IT, valid timestamp only, no issue/publication/revision fields; **CURRENT_ONLY**.
- `edh.input.cross_border_exchanges`: directed exchange columns including CH links, valid timestamp only, no release metadata; **CURRENT_ONLY**.
- `edh.input.ntc_month`: CH-border capacity columns in MW-named fields, no release metadata; **PIT_WEAK**.
- Generation tables: actual plus `current`, `day_ahead`, `intraday` or total-generation forecast fields, but no forecast issue/publication timestamps; **PIT_WEAK**.

LSEG provides the strongest explicit PIT metadata through EDF `publication_time_utc`, `version`, `is_first_publication`, prior-version and revision fields. LSEG historical price/demand workbooks have delivery timestamps but no publication timestamp, so their PIT class is **PIT_RECONSTRUCTABLE** only when exchange publication schedules are defensible. Current weather, residual-load and supply exports are **CURRENT_ONLY**; screenshots/reports are not promoted to row-level model data.

## What LSEG gives us

1. France and Italy day-ahead price histories at hourly/15-minute granularity, plus currently present CH/DE hourly workbooks.
2. Italy national hourly and seven-zone 15-minute demand.
3. French nuclear outage availability, event revisions and publication-aware daily PIT features.
4. EDF transparency/REMIT message history.
5. Current/recent temperature, precipitation, cloud cover, residual load, wind/solar supply, weather-regime and cost-curve context.

## Duplicates and complementary information

### Duplicates / overlaps

There is no substantive tabular duplication between LSEG and the Databricks challenge tables. Databricks has generation, net positions, exchanges and NTC; LSEG has no comparable row-level net-position/flow panel. The LSEG wind/solar supply references overlap conceptually with Databricks generation forecasts, but are current/reference exports rather than the same historical tables. Repeated root/intermediate LSEG paths are exact-content duplicates and are catalogued as such.

### Complementary

The most useful combined feature set is Databricks net position/exchange/NTC/generation joined with LSEG prices, Italy demand, French nuclear PIT revisions, current weather and residual-load context. The combination is still limited by the absence of common issue/publication timestamps.

## Truly missing / external gaps

The retained LSEG files do not close these gaps: Swiss hydro reservoirs/inflows/pumped storage; historical PIT-safe weather/NWP vintages; historical load and load-forecast vintages outside Italy (especially CH/DE/FR/AT); reliable historical CH bilateral schedules/flows and ATC/NTC vintages; historical forwards/short-dated curves; and broader non-French fleet availability.

## Five highest-priority remaining external data acquisitions

1. Swiss hydro, reservoir, inflow and pumped-storage time series with publication/revision metadata.
2. Historical PIT-safe NWP/weather forecast vintages with issue and valid times for load, wind and solar.
3. Historical load and load-forecast vintages for CH/DE/FR/AT and regional FR/DE/IT coverage.
4. Reliable historical CH-DE/FR/IT/AT schedules, flows and ATC/NTC with release timestamps.
5. Historical day-ahead, forward and short-dated power curves with as-of timestamps.

## Limitations and handling

- Legacy `.xls` files were identified and semantically mapped where documentation/visible labels allowed, but their full cell schema was not asserted because a reliable legacy-XLS parser was unavailable in the local environment.
- PDFs and PNGs were treated as non-tabular evidence; no row-level date range, missingness, or forecast-vintage semantics were inferred from images.
- Workbook units are marked as likely/contextual where not declared in the workbook layout.
- No expensive full-table scans were run against Databricks, and no raw files were modified or deleted.
- No commit was created.
