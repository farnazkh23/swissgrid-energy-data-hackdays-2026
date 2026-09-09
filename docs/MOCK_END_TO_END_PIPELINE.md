# Mock end-to-end pipeline

This document describes the deterministic integration path added for the next
forecasting milestone. It is an integration fixture, not a claim that the
synthetic data predicts the Swiss grid.

## Path

The dry run executes the existing contracts in this order:

1. deterministic hourly mock receipts for net position, load, wind, solar,
   hydro, and neighbour-flow-like series;
2. exact-byte ingestion into a temporary raw evidence store;
3. point-in-time revision resolution and dataset membership filtering;
4. an issue-time feature build with explicit missing and stale flags;
5. rolling-origin splits with purge, embargo, and an isolated holdout;
6. persistence, seasonal persistence, historical conditional empirical, and
   optional Ridge candidates;
7. rolling OOF predictions and an OOF-only scoreboard;
8. a selected baseline forecast, empirical samples, quantiles, and histogram.

All clocks are supplied by the caller. The fixed seed controls only synthetic
noise. Feature construction excludes observations whose `event_time` is after
the issue time, even if those observations were known early. Revisions are
selected by the existing `known_at` and revision-sequence rules.

## Output contract

`forecast.json` contains identity, model and data manifests, fit cutoff,
quantiles, deterministic empirical samples, sign-derived import/export
probabilities, provenance, warnings, and explicit fallback/abstention state.
The sign convention is `positive=import; negative=export`; import is the
empirical probability of a nonnegative sample and export is the probability of
a negative sample, so defined probabilities sum to one.

`histogram.json` contains equal-width bin edges, centers, bin probabilities,
their validated total, median, and central 50%/90% empirical intervals.
`scoreboard.json` identifies unavailable optional capabilities without putting
invented metric values in the table. `provenance.json` records source, data,
feature, model, OOF, and quality identities. `run_manifest.json` records the
explicit issue time, seed, artifact names, and holdout policy.

Run it with:

```bash
PYTHONPATH=src python3 -m swissgrid_forecaster.dry_run \
  --issue-time 2026-01-10T00:00:00+00:00 \
  --output-dir artifacts/mock_run
```

The command writes exactly these forecast artifacts:

```text
artifacts/mock_run/forecast.json
artifacts/mock_run/histogram.json
artifacts/mock_run/scoreboard.json
artifacts/mock_run/provenance.json
artifacts/mock_run/run_manifest.json
```

This milestone intentionally does not start specialist models, a Final Brain,
trust or copula machinery, news, or physical grid modeling. Real-source
calibration, operational relevance, and any production forecast claim remain
impossible until real historical data and approved source contracts are
available.
