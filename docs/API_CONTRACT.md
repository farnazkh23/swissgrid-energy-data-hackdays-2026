# Forecast serving API contract

This is the versioned output contract for the future frontend. It reads the
immutable JSON artifacts written by the forecast pipeline. Serving does not
fit, recalibrate, sample, or otherwise recompute a model. If a required value
is absent from an artifact, the API returns `null` or an unavailable artifact
error; it does not invent a replacement forecast value.

The current schema version is `forecast-api.v1`. JSON serialization is stable:
keys are sorted, separators are compact, Unicode is preserved, and non-finite
numbers are rejected.

## Endpoints

| Method and path | Response |
| --- | --- |
| `GET /health` | repository availability and latest forecast identity |
| `GET /forecast/latest` | complete latest forecast response |
| `GET /forecast/{forecast_id}` | complete response for the persisted forecast ID |
| `GET /scoreboard` | persisted candidate metrics, capabilities, and selection |
| `GET /histogram/latest` | persisted histogram for the latest forecast |
| `GET /provenance/{forecast_id}` | preserved provenance artifact for that forecast |

FastAPI support is optional. `create_app(...)` is available when FastAPI is
already installed; importing the module or using `ForecastService` does not
require it.

## Forecast response

`GET /forecast/latest` and `GET /forecast/{forecast_id}` return:

```json
{
  "schema_version": "forecast-api.v1",
  "forecast": {
    "forecast_id": "...",
    "issue_time": "2026-01-10T00:00:00+00:00",
    "target_time": "2026-01-10T01:00:00+00:00",
    "horizon": "1:00:00",
    "target": {"name": "net_position", "entity": "AT"},
    "unit": "MW",
    "sign_convention": "positive=import; negative=export"
  },
  "probability": {
    "p10": -54.9, "p25": -52.8, "p50": -47.9,
    "p75": -46.3, "p90": -43.4,
    "probability_import": 0.0, "probability_export": 1.0
  },
  "visualization": {
    "histogram": {
      "bin_edges": [], "bin_centers": [], "probabilities": [],
      "total_probability": 1.0
    },
    "central_50_interval": [-52.8, -46.3],
    "central_90_interval": [-55.6, -42.5],
    "median": -47.9
  },
  "model": {
    "selected_champion": "seasonal_persistence",
    "selected_model": "seasonal_persistence",
    "model_version": "...",
    "oof_metrics": {"mae": 3.5},
    "fit_cutoff": "2026-01-10T00:00:00+00:00"
  },
  "evidence": {
    "dataset_manifest": "...",
    "feature_manifest": "...",
    "sources": ["..."],
    "freshness": {},
    "warnings": []
  },
  "state": "NORMAL"
}
```

The probability fields are empirical values already persisted in
`forecast.json`; import and export probabilities sum to one when present.
Histogram bins and all interval values are read from `histogram.json`, whose
probabilities must total one. OOF metrics come from the selected candidate in
`scoreboard.json` and are `null` if that artifact has no metrics.

The four scored forecast-output targets are exactly AT, DE, FR, IT in that
canonical order (see `swissgrid_forecaster.target_contract.TARGETS`). CH is an
input/context country only and must never appear as a served forecast-output
entity; any Switzerland aggregate shown by a frontend is derived context, not a
scored target.

`state` is one of `NORMAL`, `FALLBACK`, `ABSTAIN`, or `DEGRADED`. Explicit
abstention takes precedence over fallback. Fallback takes precedence over
degradation. A stale forecast or persisted warning produces `DEGRADED` when
neither fallback nor abstention applies. The service default staleness window
is 24 hours and can be supplied explicitly by the embedding application.

## Other response envelopes

Scoreboard returns `{ "schema_version": ..., "scoreboard": <persisted object> }`.
Histogram returns `{ "schema_version": ..., "forecast_id": ..., "histogram":
<persisted object> }`. Provenance returns `{ "schema_version": ...,
"forecast_id": ..., "provenance": <persisted object> }`; the nested object is
preserved, including fields added by future artifact writers.
