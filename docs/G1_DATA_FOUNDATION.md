# G1 data/domain ingestion foundation

This branch adds provider-neutral ingestion and dataset contracts. It reuses
`Observation`, `RawStore`, `earliest_known_at`, `utc`, `resolve_as_of`, and
`TargetContract` without changing them. No external calls, production adapters,
model training, specialists, or Final Brain are implemented.

## Files and responsibilities

| Module in `src/swissgrid_forecaster/` | Responsibility |
| --- | --- |
| `source_contracts.py` | Immutable source metadata, domain enum, explicit unit declarations and horizon range |
| `source_registry.py` | Unique source registration, lookup and domain filtering |
| `ingestion.py` | Received bytes, adapter protocol, raw persistence and provenance validation |
| `dataset_contracts.py` | Source membership, metadata validation and as-of selection |
| `target_config.py` | Bind an existing explicit target contract to source and record identity |
| `data_quality.py` | Configurable diagnostics and pre-normalization timestamp flags |
| `mock_sources.py` | Deterministic synthetic fixtures exclusively for tests |

Each module has a matching `tests/test_<module>.py` file.

## Source and adapter contracts

`SourceContract` requires source identity, country/scope text, one of the 14
requested domains, provider, IANA timezone, permitted unit symbols, update cadence,
publication lag, revision policy, horizon availability, and license/usage notes.
Country is explicit scope text, not an enforced ISO code (weather and market
sources may cover areas beyond one country). No real source is registered by
default. Unit validation is exact membership in the configured source's units;
there is no inferred dimensional conversion or domain-specific unit default.

Unknown cadence, lag, and horizons may be represented as `None` for archival
onboarding. Forecast ingestion and dataset selection reject any such missing
metadata. Cadence is positive; lag is nonnegative. Horizon bounds are inclusive
`valid_time - issue_time` offsets and may include negative offsets for historical
inputs. Descriptive publication lag never substitutes for actual receipt or
normalization evidence. Unknown publication time remains valid when receipt and
normalization clocks are present, as specified by the existing availability
contract. Revision and usage policy text records declarations; it is not an
executable policy engine.

Future Swissgrid, ENTSO-E, weather, generation/outage and market/schedule adapters
implement `SourceAdapter.normalize(received, normalized_at=...) -> Observation`.
The ingestion entry point accepts already-received bytes and performs no fetching.
Its current envelope represents one independently revisable scalar record;
batch transport parsing and stable scalar IDs must be supplied by future adapters.
Callers attest completed receipt and normalization times. All present clocks must
be aware; UTC conversion follows the existing helper. Civil-time ambiguity must
be resolved by the adapter before construction; source timezone metadata is not
used to silently localize naive times.

`ingest` validates source registration and required forecast metadata, persists
exact raw bytes, invokes the adapter, and verifies source/record/revision identity,
receipt and normalization clocks, declared units and raw digest. Invalid parsing
or units leave raw evidence available for diagnosis but return no normalized
record. Observations preserve event, valid, publication (if known), receipt,
normalization and knowledge times, revision sequence and predecessor identity,
source identity and raw SHA-256. Existing availability lower bounds remain in
force. Full predecessor-chain validation is adapter-specific and is not invented
for partial histories. Normalized persistence is left to the caller.

## Dataset and target use

`DatasetContract.as_of` validates all declared sources, including those with no
rows, and checks every row's membership and unit. It delegates revision eligibility,
conflict rejection and deterministic selection to `resolve_as_of`, then restricts
selected records to the source's configured horizon range. Missing clocks cannot
enter an `Observation`; missing source availability fails before selection.
Future-known records are excluded and exact issue-time equality is eligible.
The selection API does not certify dataset completeness or label maturity.

`TargetConfig` requires an existing `TargetContract` plus explicit source and
record IDs. Its `validate` method checks registration, units and source metadata.
It does not force future labels to be observable at issue time, interpret target
sign, execute aggregation, or choose a target for Switzerland.

## Quality diagnostics

All seven flags are supported: `missing`, `stale`, `revised`, `duplicate`,
`out_of_range`, `timezone_issue`, `unavailable_at_issue`.

`assess` returns immutable flag sets aligned with input rows. Missing means a
null value. Staleness compares event age at issue against an explicitly configured
maximum; bounds apply to numeric values only. Revised means a declared predecessor
exists, since a sequence starting at one does not alone prove revision. Duplicate
marks every repeated source/record/revision identity; conflicting duplicates still
fail in the as-of resolver. Availability uses the existing helper. Quality flags
are diagnostics and do not silently filter records or alter evidence.

`timestamp_flags` diagnoses required raw clocks before normalization: missing
clocks flag missing/unavailable, invalid or naive clocks flag timezone issues.
Omit unknown optional publication time from this helper. Invalid timestamps still
raise during normalized construction; flags never relax validation. Missing
expected intervals require a future explicitly configured sampling grid; they
cannot be inferred from absent rows without target resolution semantics.

Mock payloads and normalized records depend only on explicit inputs. All domains
are supported and marked synthetic/test-only. Raw receipt UUIDs retain the existing
store behavior and are intentionally not deterministic.

## Verification

`PYTHONPATH=src python3 -m unittest discover -s tests -v`

**61 tests passed**, including 31 new tests covering metadata, every domain,
units, clocks, provenance tampering, raw retention, repeat ingestion, missing
availability, revision leakage/conflicts, horizons, target binding, all quality
flags and deterministic fixtures. No dependencies were installed.

## Unresolved domain decisions

- Final physical target, geography/border and authoritative label source.
- Sign and direction conventions, units/conversions and aggregation rules.
- Resolution, interval boundaries, issue schedule, horizons and DST handling.
- Actual provider cadence, publication lag, access/embargo rules and licensing.
- Provider scalar identities, revision ordering, lineage, maturity and finality.
- Per-series ranges, staleness policy and expected-grid completeness rules.
- Production transport/batch mapping and normalized dataset persistence.

These decisions remain explicit configuration or future adapter work. No G2
modeling/evaluation files or historical GOTTI/BELA/DHAM files were modified.
