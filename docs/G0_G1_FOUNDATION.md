# G0 + minimal G1 foundation

Implemented 2026-09-09 in `/home/farnaz/swissgrid-forecaster` against the frozen
`/home/farnaz/GOTTI_WORKING_COPY/SWISSGRID_PLAN_CODE_RECONCILIATION.md`.
The historical GOTTI/BELA/DHAM snapshot was read only. No historical module was
imported, executed, or modified. BELA collector/core/schema informed raw byte
identity and receipts; GOTTI collection and snapshot patterns informed the
negative cases. This is independently implemented code.

## Scope

Only the target contract, observation schema, availability calculation, as-of
resolver, raw store, manifest, and their tests are implemented. G0 semantics are
representable but are not scientifically locked by this implementation. The rest
of G1, including feature registry and fold-local transformations, is deferred.
No models, specialists, Final Brain, calibration, or scenario engine exist.
Scenario support is a contract boolean only. Task 2 has not started.

## Target contract

`TargetContract` records target name, entity, optional country and ordered border,
unit, explicit sign convention, positive resolution, issue and valid times,
horizon, aggregation rule, label availability rule, forecast type, quantile
levels, optional scenario support, and evaluation metric names.

Valid time must strictly follow issue time and horizon must equal their elapsed
UTC difference. Quantiles must be finite, unique, strictly increasing, and inside
(0, 1). Quantile forecasts require at least one level. Aware offset datetimes are
normalized to UTC before elapsed-time calculations. No sign convention or target
interpretation is supplied by default. Rule strings and metric names declare
configuration; this foundation does not execute aggregation, label release
policies, or metric calculations.

## Observation and availability

`Observation` is a frozen scalar schema, supporting numeric, string, boolean and
missing values. Mutable nested values are deliberately excluded. It preserves
`event_time`, `valid_time`, optional `publication_time`, `first_received_at`,
`normalized_at`, `known_at`, `revision_id`, `source_id`, and `source_record_id`.
Optional raw digest and predecessor revision ID link provenance. All present
clocks must be timezone-aware and are stored in UTC.

For this live-pipeline foundation, the conservative lower bound for knowledge is:

```
max(first_received_at, normalized_at, publication_time if known,
    permitted_at if supplied to earliest_known_at)
```

Normalization cannot precede completed receipt. Observation validation rejects
knowledge earlier than receipt, normalization, or known publication. A later
`known_at` can represent an embargo or other access restriction. The caller must
supply the correct policy-derived time; the schema cannot independently establish
legal access. Neither event nor valid time is constrained to precede knowledge.
This permits planned outages, schedules, and weather forecasts of future periods.

Missing or invalid knowledge fails closed with `ValueError`. Future knowledge is
valid evidence but is excluded from an earlier issue. Equality at the issue
boundary is eligible. No clock is read by availability or resolution.

## Revision and source resolution

Logical identity is `(source_id, source_record_id)`. Adapters must give each
independently revisable scalar a stable record ID (including variable, location,
run and valid interval as appropriate), and assign a nonnegative monotonic
`revision_sequence` according to provider revision order. Opaque revision IDs are
preserved but never lexically ranked. Late receipt of an older version does not
make it the newest revision.

The resolver filters by knowledge first and then selects the maximum eligible
sequence, returning a tuple sorted by logical identity. Equal observations dedup
idempotently; conflicting eligible content for an ID or sequence raises instead
of depending on input order. Future versions cannot change earlier selection or
introduce eligible-revision conflicts. Invalid input schemas are always rejected.
`resolve_sources_as_of` also checks source collection keys against each record.

Frozen versions preserve their own clocks and values. The resolver never changes
the input or reads a mutable latest-state object. `supersedes_revision_id`
preserves declared lineage; full predecessor-chain integrity is not validated
because a query may receive only a subset of source history. Source adapters must
validate provider lineage. Normalized observation persistence and deletions /
tombstones are outside this minimal foundation.

## Raw evidence and manifest persistence

Canonical raw representation means the exact immutable retrieved `bytes`.
There is no JSON parse/serialize cycle, whitespace rewriting, added newline, or
mutable companion reread. Semantically equivalent JSON with different bytes is
intentionally different raw evidence. SHA-256 is computed exactly once on this
representation before persistence. Existing blobs are compared directly with
those bytes; reads independently recompute the digest for verification.

`RawStore.retrieve(fetch, ...)` calls the supplied retrieval function before the
receipt clock. The function must finish reading and closing the external response
before returning bytes. Failed retrieval records neither timestamp nor manifest.
Source metadata is copied before retrieval, avoiding a later mutable metadata
reread. `put` is the lower-level interface for already-completed receipts: its
caller attests the receipt timestamp. It cannot prove that caller's historical
claim. No external adapter or network retrieval is implemented here.

Store layout:

```
raw/<sha256>                 exact payload, shared across receipts
manifests/<receipt_id>.json  immutable schema-versioned receipt
```

Each receipt records SHA-256, length, source/record/revision IDs, UTC receipt time,
and immutable string metadata. Duplicate payloads reuse the blob and get distinct
receipt manifests, preserving source and receipt lineage. Receipt IDs use UUID4;
raw identity alone is content-addressed. Manifest enumeration has deterministic
ordering for persisted entries. The manifest is provenance, not a signed audit
ledger; payload verification cannot authenticate externally modified metadata.

Writes use a temporary file in the destination directory, flush/fsync, read-only
mode, then atomic hard-link publication with no replacement. Directory entries are
fsynced after cleanup. Concurrent publishers cannot overwrite the winner. Existing
corrupt blobs raise `IntegrityError` and are never repaired silently. Reads check
the digest, and receipt reads also check length. This implementation targets local
POSIX filesystems supporting hard links and directory fsync; it is not an object
storage or distributed transaction implementation. Read-only mode prevents casual
writes but is not protection against an owner or administrator changing files.

Blob publication precedes manifest publication. A crash in between can leave an
unreferenced complete blob; a retry verifies and reuses it. Publication is atomic
per file, not a two-file transaction. Automatic orphan deletion and receipt retry
idempotency are deferred; retrying a receipt intentionally creates another receipt.

## Verification

Executed only this project's test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Result: **30 tests passed** on Python 3.14.4. No runtime or test dependency install
was needed. Tests cover all 13 requested adversarial cases, plus exact issue-time
boundaries, embargo calculation, immutable lineage, UTC offset normalization,
late-arriving older revisions, multi-source isolation, conflicting versions,
concurrent duplicate publication, atomic publication failure, hash call count,
manifest round-trip, receipt length verification, and failed retrieval behavior.
No historical tests or runtime commands were run. Python 3.11 compatibility is
declared but was not separately executed in this environment.

## G0 decisions still required

- Exact physical target: actual physical flow, scheduled exchange, net balance,
  or another precisely specified quantity; country and border scope.
- Sign orientation, border direction, unit, and aggregation semantics.
- Resolution, valid-interval convention, weekly issue schedule, horizon set, and
  any civil-time scheduling / daylight-saving rules before UTC conversion.
- Authoritative label source, release latency, embargo, revision maturity and
  finality; whether a historically reconstructed research policy will differ from
  this conservative live-pipeline receipt/normalization policy.
- Forecast representation, quantile levels, evaluation metrics, and evaluation
  aggregation. No metric set is silently selected for production.
- Provider-specific logical record identity, revision sequence and predecessor
  mapping, plus evidence supporting historical first receipt and normalization.

These are configuration and source-policy decisions, not assumed Swissgrid facts.
