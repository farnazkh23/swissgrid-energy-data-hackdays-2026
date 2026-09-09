# Swissgrid Forecaster

G0 target contracts and the minimal G1 point-in-time evidence foundation. Python
3.11+; no runtime dependencies or imports from the historical estate.

Run only this project's tests from this directory:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The package uses immutable dataclasses with runtime validation. Aware timestamps
are converted to UTC; naive or absent required clocks raise `ValueError`.
`resolve_as_of` returns only observations known by the issue time, choosing the
highest eligible revision sequence for each `(source_id, source_record_id)`.

`RawStore.retrieve` receives complete bytes before sampling its receipt clock.
`RawStore.put` accepts already-retrieved immutable bytes and an attested receipt
time. Raw identity is SHA-256 of the exact supplied bytes; manifests preserve
individual receipts even when multiple receipts share one payload.

See [the implementation note](docs/G0_G1_FOUNDATION.md) for contracts, assumptions,
test coverage, and persistence guarantees. Target semantics remain configurable
pending G0 confirmation. No models, specialists, Final Brain, calibration,
scenario generation, or Task 2 implementation are included.
