"""Deterministic mock end-to-end orchestration for integration tests only."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import random
import tempfile
import math
from pathlib import Path

from .asof_resolver import resolve_as_of
from .availability import utc
from .baseline_runner import BaselineRun, run_baselines
from .champion import HistoricalConditional, OptionalEstimator, Persistence, SeasonalPersistence
from .dataset_contracts import DatasetContract
from .feature_registry import FeatureDefinition, FeatureRegistry, identity
from .forecast_output import ForecastOutput
from .histogram import HistogramPayload
from .ingestion import ReceivedRecord, ingest
from .model_contracts import FitContext, PredictContext
from .mock_sources import MockAdapter, mock_contract, mock_record
from .raw_store import RawStore
from .source_contracts import Domain
from .source_registry import SourceRegistry
from .splits import RollingOrigin, Sample, data_manifest


MOCK_DOMAINS = (Domain.NET_POSITION, Domain.LOAD, Domain.WIND, Domain.SOLAR,
                Domain.HYDRO, Domain.CROSS_BORDER_FLOW)
TARGET_DOMAIN = Domain.NET_POSITION
STALE_AFTER = timedelta(hours=3)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    issue_time: datetime
    seed: int = 20260909
    horizon: timedelta = timedelta(hours=1)
    history_hours: int = 216
    first_origin_hours: int = 72
    train_window_hours: int = 48
    validation_window_hours: int = 12
    calibration_window_hours: int = 6
    step_hours: int = 24
    include_ridge: bool = True

    def __post_init__(self):
        object.__setattr__(self, "issue_time", utc(self.issue_time, "issue_time"))
        if type(self.seed) is not int or self.history_hours <= self.first_origin_hours:
            raise ValueError("invalid deterministic mock configuration")
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("positive horizon required")
        if any(type(value) is not int or value <= 0 for value in
               (self.history_hours, self.first_origin_hours, self.train_window_hours,
                self.validation_window_hours, self.calibration_window_hours, self.step_hours)):
            raise ValueError("window sizes must be positive integers")


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    samples: tuple[Sample, ...]
    registry: FeatureRegistry
    flags_by_issue: dict
    used_observation_ids: dict


@dataclass(frozen=True, slots=True)
class MockEvidence:
    observations: tuple
    source_ids: tuple[str, ...]
    registry: SourceRegistry
    dataset: DatasetContract
    raw_manifest_count: int


@dataclass(frozen=True, slots=True)
class PipelineResult:
    config: PipelineConfig
    evidence: MockEvidence
    features: FeatureBuildResult
    plan: RollingOrigin
    baselines: BaselineRun
    forecast: ForecastOutput
    histogram: HistogramPayload
    provenance: dict
    run_manifest: dict


def _hour_range(start: datetime, end: datetime):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(hours=1)


def _source_id(domain: Domain) -> str:
    return mock_contract(domain).source_id


def _received(domain, at, value, *, revision_sequence=0, first_received_at=None):
    received = mock_record(at=at, domain=domain, value=value, revision_sequence=revision_sequence)
    received = replace(received, source_record_id=f"{_source_id(domain)}:{at.isoformat()}")
    if first_received_at is not None:
        received = replace(received, first_received_at=first_received_at)
    return received


def generate_mock_received_records(config: PipelineConfig) -> tuple[ReceivedRecord, ...]:
    """Generate realistic-but-non-predictive scalar source receipts."""
    rng = random.Random(config.seed)
    start = config.issue_time - timedelta(hours=config.history_hours + 2)
    end = config.issue_time + config.horizon + timedelta(hours=2)
    records = []
    for index, at in enumerate(_hour_range(start, end)):
        elapsed = (at - start).total_seconds() / 3600
        hour = at.hour + at.minute / 60
        daily = math.sin(2 * math.pi * hour / 24)
        weekly = math.sin(2 * math.pi * elapsed / 168)
        load = 92 + 18 * daily + 3 * weekly + rng.gauss(0, 1.2)
        wind = 27 + 9 * math.sin(2 * math.pi * (hour + 4) / 24) + rng.gauss(0, 2)
        solar = max(0, 20 * math.sin(math.pi * (hour - 6) / 12)) + rng.gauss(0, .8)
        hydro = 16 + 2 * weekly + rng.gauss(0, .4)
        neighbour = 4 * math.sin(2 * math.pi * (hour - 2) / 24) + rng.gauss(0, 1)
        target = wind + solar + hydro + neighbour - load + rng.gauss(0, 1.5)
        values = {Domain.NET_POSITION: target, Domain.LOAD: load, Domain.WIND: wind,
                  Domain.SOLAR: solar, Domain.HYDRO: hydro, Domain.CROSS_BORDER_FLOW: neighbour}
        for domain in MOCK_DOMAINS:
            # A gap near the issue time makes stale state observable. A None
            # observation makes missingness observable without dropping the row.
            if domain == Domain.WIND and config.issue_time - timedelta(hours=6) < at <= config.issue_time:
                continue
            value = None if domain == Domain.SOLAR and index % 29 == 0 else round(values[domain], 6)
            delayed = timedelta(hours=3) if domain == Domain.LOAD and index % 13 == 0 else timedelta(0)
            receipt = at + delayed
            base = _received(domain, at, value, first_received_at=receipt)
            records.append(base)
            if domain == Domain.LOAD and index % 17 == 0:
                records.append(_received(domain, at, None if value is None else value + 1.25,
                                         revision_sequence=1, first_received_at=at + timedelta(hours=2)))
    return tuple(sorted(records, key=lambda record: (record.first_received_at,
                                                     record.source_id, record.source_record_id,
                                                     record.revision_id)))


def _feature_registry() -> FeatureRegistry:
    definitions = []
    for domain in MOCK_DOMAINS:
        source = _source_id(domain)
        for name, units in (("value", "MW"), ("missing", "flag"),
                            ("stale", "flag"), ("freshness_hours", "hours")):
            definitions.append(FeatureDefinition(
                f"{source}-{name}", "mock-source", f"{domain.value}_{name}", ("CH",), source,
                "known_at <= issue_time and event_time <= issue_time", (timedelta(hours=1),),
                "latest_as_of_issue", "1", (), units, "medium", "propagate"))
    definitions.extend([
        FeatureDefinition("calendar-hour", "calendar", "utc_hour", ("UTC",), "derived",
                          "issue_time", (timedelta(hours=1),), "utc_hour", "1", (), "hour", "low", "propagate"),
        FeatureDefinition("calendar-weekday", "calendar", "utc_weekday", ("UTC",), "derived",
                          "issue_time", (timedelta(hours=1),), "utc_weekday", "1", (), "day", "low", "propagate"),
    ])
    return FeatureRegistry(tuple(definitions))


def ingest_mock_evidence(config: PipelineConfig, raw_root: str | Path | None = None) -> MockEvidence:
    sources = tuple(mock_contract(domain) for domain in MOCK_DOMAINS)
    registry = SourceRegistry(sources)
    dataset = DatasetContract("mock-multivariate-hourly-v1", tuple(source.source_id for source in sources))
    records = generate_mock_received_records(config)
    adapter = MockAdapter()
    if raw_root is None:
        with tempfile.TemporaryDirectory(prefix="swissgrid-mock-raw-") as directory:
            return _ingest(records, adapter, registry, dataset, directory)
    return _ingest(records, adapter, registry, dataset, raw_root)


def _ingest(records, adapter, registry, dataset, raw_root):
    store = RawStore(raw_root)
    observations = []
    for received in records:
        result = ingest(received, adapter, registry, store,
                        normalized_at=received.first_received_at, for_forecasting=True)
        observations.append(result.observation)
    return MockEvidence(tuple(observations), dataset.source_ids, registry, dataset,
                        len(store.manifests()))


def _latest_by_source(observations, issue_time):
    issue = utc(issue_time)
    selected = resolve_as_of(observations, issue)
    latest = {}
    for observation in selected:
        if observation.event_time <= issue and observation.valid_time <= issue:
            current = latest.get(observation.source_id)
            if current is None or (observation.event_time, observation.revision_sequence) > (current.event_time, current.revision_sequence):
                latest[observation.source_id] = observation
    return latest


def _feature_vector(observations, source_ids, issue_time):
    issue = utc(issue_time)
    latest = _latest_by_source(observations, issue)
    values, flags, used_ids = [], {}, []
    for source_id in source_ids:
        observation = latest.get(source_id)
        missing = observation is None or not isinstance(observation.value, (int, float)) or isinstance(observation.value, bool)
        freshness = (issue - observation.event_time).total_seconds() / 3600 if observation else None
        stale = missing or freshness > STALE_AFTER.total_seconds() / 3600
        value = 0.0 if missing else float(observation.value)
        values.extend((value, float(missing), float(stale), float(freshness if freshness is not None else 9999)))
        flags[source_id] = {"missing": bool(missing), "stale": bool(stale),
                            "freshness_hours": freshness, "observation_id": observation.revision_id if observation else None}
        if observation is not None:
            used_ids.append(f"{observation.source_id}:{observation.source_record_id}:{observation.revision_id}")
    values.extend((float(issue.hour), float(issue.weekday())))
    known = max([issue] + [latest[source_id].known_at for source_id in latest])
    return tuple(values), known, flags, tuple(sorted(used_ids))


def build_feature_samples(config: PipelineConfig, evidence: MockEvidence) -> FeatureBuildResult:
    registry = _feature_registry()
    start = config.issue_time - timedelta(hours=config.history_hours)
    samples = []
    flags_by_issue, used_by_issue = {}, {}
    target_source = _source_id(TARGET_DOMAIN)
    for issue in _hour_range(start, config.issue_time):
        features, feature_known_at, flags, used_ids = _feature_vector(evidence.observations,
                                                                       evidence.source_ids, issue)
        target_time = issue + config.horizon
        target_rows = [row for row in evidence.dataset.as_of(evidence.observations,
                                                              evidence.registry, target_time)
                       if row.source_id == target_source and row.valid_time == target_time]
        target = target_rows[0].value if target_rows and isinstance(target_rows[0].value, (int, float)) else None
        label_known_at = target_rows[0].known_at if target_rows else target_time
        row_id = f"mock:{issue.isoformat()}"
        samples.append(Sample(row_id, issue, target_time, feature_known_at, label_known_at,
                              features, None if target is None else float(target)))
        flags_by_issue[issue.isoformat()] = flags
        used_by_issue[issue.isoformat()] = used_ids
    return FeatureBuildResult(tuple(samples), registry, flags_by_issue, used_by_issue)


def _model_factory(model_id):
    if model_id == "persistence":
        return Persistence()
    if model_id == "seasonal_persistence":
        return SeasonalPersistence(timedelta(hours=24))
    if model_id == "historical_conditional":
        return HistoricalConditional(())
    if model_id == "ridge":
        return OptionalEstimator("ridge")
    raise ValueError(f"unknown selected model: {model_id}")


def run_mock_pipeline(config: PipelineConfig, *, raw_root: str | Path | None = None) -> PipelineResult:
    evidence = ingest_mock_evidence(config, raw_root)
    features = build_feature_samples(config, evidence)
    plan = RollingOrigin(
        config.issue_time - timedelta(hours=config.first_origin_hours), config.issue_time,
        timedelta(hours=config.train_window_hours), timedelta(hours=config.validation_window_hours),
        timedelta(hours=config.calibration_window_hours), timedelta(hours=config.step_hours),
        purge=config.horizon, embargo=config.horizon, label_delay=timedelta(0))
    baselines = run_baselines(plan, features.samples, features.registry.manifest_hash,
                              include_ridge=config.include_ridge)
    selected_id, selected_version = baselines.selected_model
    train = tuple(replace(row, partition="train") for row in features.samples
                  if row.issue_time < config.issue_time and row.target_time <= config.issue_time
                  and row.label_known_at <= config.issue_time and row.feature_known_at <= row.issue_time
                  and row.target is not None)
    if not train:
        raise ValueError("mock final fit has no point-in-time training rows")
    fit_context = FitContext(train, config.issue_time, features.registry.manifest_hash, data_manifest(train))
    model = _model_factory(selected_id)
    model.fit(tuple(row.features for row in train), tuple(row.target for row in train), fit_context)
    final_sample = next(row for row in features.samples if row.issue_time == config.issue_time)
    hidden_final = replace(final_sample, target=None, label_known_at=final_sample.target_time)
    prediction = model.predict((hidden_final.features,), PredictContext((hidden_final,), features.registry.manifest_hash))[0]
    if prediction.prediction_type == "distribution":
        forecast_samples = prediction.values
    else:
        residuals = tuple(baselines.oof.truth(record.fold_id, record.row_id) - record.prediction.values[0]
                          for record in baselines.oof.records if (record.model_id, record.version) == baselines.selected_model)
        point = prediction.values[0]
        forecast_samples = tuple(point + residual for residual in residuals) or (point,)
    all_manifest = identity([(row.source_id, row.source_record_id, row.revision_id,
                              row.known_at.isoformat(), row.raw_sha256, row.value)
                             for row in sorted(evidence.observations, key=lambda row: (
                                 row.source_id, row.source_record_id, row.revision_sequence))])
    forecast_id = identity([config.issue_time.isoformat(), config.seed, selected_id,
                            selected_version, all_manifest, features.registry.manifest_hash])
    final_flags = features.flags_by_issue[config.issue_time.isoformat()]
    warnings = ["synthetic mock data; results have no predictive relevance claim"]
    if any(value["missing"] for value in final_flags.values()):
        warnings.append("missing source observations were imputed to zero and flagged")
    if any(value["stale"] for value in final_flags.values()):
        warnings.append("stale source observations were retained and flagged")
    forecast = ForecastOutput.from_samples(
        forecast_id=forecast_id, issue_time=config.issue_time, horizon=config.horizon,
        target_name="net_position", target_entity="AT", unit="MW",
        selected_model_id=selected_id, selected_model_version=selected_version,
        feature_manifest_hash=features.registry.manifest_hash,
        training_data_manifest_hash=data_manifest(train), fit_cutoff=config.issue_time,
        samples=forecast_samples, source_ids=evidence.source_ids,
        dataset_manifest_id=all_manifest, feature_manifest_id=features.registry.manifest_hash,
        fold_evaluation_summary={"folds": len(baselines.oof.folds), "oof_rows": len(baselines.oof.records),
                                 "selection_metric": "mae", "holdout_used": False},
        warnings=warnings, fallback_abstention_state={"fallback_used": False, "abstained": False,
                                                       "reason": None, "policy": "mock_baselines_only"})
    histogram = HistogramPayload.from_samples(forecast.samples)
    provenance = {
        "source_ids": list(evidence.source_ids), "dataset_manifest_id": all_manifest,
        "feature_manifest_id": features.registry.manifest_hash,
        "model_id": selected_id, "model_version": selected_version,
        "training_data_manifest_hash": data_manifest(train), "fit_cutoff": config.issue_time.isoformat(),
        "raw_evidence_receipts": evidence.raw_manifest_count,
        "as_of_policy": "known_at <= issue_time; revisions resolved by highest eligible sequence; event_time <= issue_time for features",
        "feature_quality_flags": final_flags,
        "oof": forecast.fold_evaluation_summary,
        "fallback_abstention_state": forecast.fallback_abstention_state,
    }
    run_manifest = {"schema_version": "mock-run.v1", "seed": config.seed,
                    "issue_time": config.issue_time.isoformat(), "horizon": str(config.horizon),
                    "artifacts": ["forecast.json", "histogram.json", "scoreboard.json",
                                  "provenance.json", "run_manifest.json"],
                    "source_ids": list(evidence.source_ids), "holdout_used": False,
                    "historical_paths_written": []}
    return PipelineResult(config, evidence, features, plan, baselines, forecast, histogram,
                          provenance, run_manifest)
