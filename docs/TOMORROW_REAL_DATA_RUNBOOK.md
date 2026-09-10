# Tomorrow's real-data runbook

This is the pre-G3 handoff for a provider dataset. It does not activate
specialists, Final Brain, trust, copulas, news, or grid physics.

1. Copy the example JSON files and replace every `REPLACE_` value. Confirm the
   target meaning, unit, sign convention, issue schedule, and label rule with
   the data owner. Do not infer a target meaning from a column name.
2. Map provider columns in `column_mapping.json`. Every forecasting row needs
   event time, valid time, source record identity, revision identity/sequence,
   unit, and complete point-in-time clocks (`known_at`, `first_received_at`,
   and `normalized_at`).
3. Use `publication_rules.json` only when the provider has documented,
   versioned publication rules. Set reconstruction permission explicitly;
   otherwise the adapter fails closed.
4. Set the real file in `run.json`, then run:

```bash
python3 -m swissgrid_forecaster.cli readiness --config configs/run.json
python3 -m swissgrid_forecaster.cli run-champion --config configs/run.json
python3 -m swissgrid_forecaster.cli acceptance --run artifacts/real_run
```

Readiness must pass mandatory checks before the Champion runner is allowed to
fit. Review warnings about gaps, stale observations, revisions, and feature
freshness. The runner uses rolling OOF only and writes diagnostics and a
content-addressed run manifest. A failed acceptance gate is not a forecast.

Keep the original provider file immutable and retain its receipt metadata.
Never write to historical GOTTI/BELA/DHAM paths. Specialist activation remains
disabled until separate coverage and evaluation evidence exists.
