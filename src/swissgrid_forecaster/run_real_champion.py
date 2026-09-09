"""Real-data handoff runner: audit, fold-safe snapshots, OOF baselines, diagnostics."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import argparse
import json
from pathlib import Path

from .availability import utc
from .baseline_runner import BaselineRun, run_baselines
from .data_audit import DataAuditReport, audit_dataset
from .dataset_contracts import DatasetContract
from .feature_audit import FeatureAuditReport, audit_features, univariate_oof_score
from .feature_registry import FeatureDefinition, FeatureRegistry
from .real_dataset_loader import RealDataset, load_real_dataset
from .real_source_adapter import RealSourceConfig
from .source_contracts import Domain
from .splits import RollingOrigin, Sample, data_manifest
from .target_config import TargetConfig
from .target_contract import ForecastType, TargetContract


class RealChampionRefusal(ValueError):
    """A required target or PIT contract is unresolved, so no model may run."""


@dataclass(frozen=True, slots=True)
class RealChampionConfig:
    target_contract: TargetContract
    target_source_id: str
    target_source_record_id: str
    issue_times: tuple[datetime, ...]
    plan: RollingOrigin
    feature_source_ids: tuple[str, ...] = ()
    include_ridge: bool = True
    feature_registry: FeatureRegistry | None = None
    feature_recommendation_policy: dict | None = None

    def __post_init__(self):
        if not isinstance(self.target_contract, TargetContract):
            raise ValueError("explicit TargetContract is required")
        if not self.issue_times:
            raise ValueError("explicit issue times are required")
        issues = tuple(sorted({utc(value, "issue_time") for value in self.issue_times}))
        if issues[-1] != self.target_contract.forecast_issue_time:
            raise ValueError("last issue time must match TargetContract.forecast_issue_time")
        object.__setattr__(self, "issue_times", issues)
        if self.plan.holdout_start != self.target_contract.forecast_issue_time:
            raise ValueError("rolling holdout must begin at target forecast issue time")
        if not self.feature_source_ids:
            raise ValueError("feature source IDs must be explicit")
        object.__setattr__(self, "feature_source_ids", tuple(self.feature_source_ids))
        if self.feature_registry is not None and not isinstance(self.feature_registry, FeatureRegistry):
            raise ValueError("feature_registry must be a FeatureRegistry")


@dataclass(frozen=True, slots=True)
class RealChampionResult:
    dataset: RealDataset
    data_audit: DataAuditReport
    feature_audit: FeatureAuditReport
    features: FeatureRegistry
    samples: tuple[Sample, ...]
    plan: RollingOrigin
    baselines: BaselineRun
    diagnostics: dict


def _feature_registry(source_ids, unit):
    return FeatureRegistry(tuple(
        FeatureDefinition(source, "real-source", f"{source}_value", ("configured",), source,
                          "known_at <= issue_time and event_time <= issue_time", (timedelta(hours=1),),
                          "latest_as_of_issue", "1", (), unit, "review", "propagate")
        for source in source_ids))


def _resolved_latest(dataset, issue_time, source_ids):
    selected = dataset.contract.as_of(dataset.observations, dataset.registry, issue_time)
    latest = {}
    for row in selected:
        if row.source_id not in source_ids or row.event_time > issue_time or row.valid_time > issue_time:
            continue
        previous = latest.get(row.source_id)
        if previous is None or (row.event_time, row.revision_sequence) > (previous.event_time, previous.revision_sequence):
            latest[row.source_id] = row
    return latest


def build_real_samples(dataset: RealDataset, config: RealChampionConfig) -> tuple[Sample, ...]:
    target_source = config.target_source_id
    samples = []
    for issue in config.issue_times:
        latest = _resolved_latest(dataset, issue, config.feature_source_ids)
        values, known = [], issue
        for source in config.feature_source_ids:
            row = latest.get(source)
            missing = row is None or row.value is None
            values.extend((0.0 if missing else float(row.value), float(missing)))
            if row is not None:
                known = max(known, row.known_at)
        target_time = issue + config.target_contract.horizon
        record_selector = (lambda value: value.startswith(config.target_source_record_id[:-1])) \
            if config.target_source_record_id.endswith("*") else \
            (lambda value: value == config.target_source_record_id)
        selected = [row for row in dataset.contract.as_of(dataset.observations, dataset.registry, target_time)
                    if row.source_id == target_source and record_selector(row.source_record_id)
                    and row.valid_time == target_time]
        target_row = selected[0] if selected else None
        target = None if target_row is None or target_row.value is None else float(target_row.value)
        label_known = target_row.known_at if target_row is not None else target_time
        samples.append(Sample(f"real:{issue.isoformat()}", issue, target_time, known, label_known,
                              tuple(values), target))
    return tuple(samples)


def _feature_audit(dataset, config, features, samples, *, incremental_oof_gains=None):
    feature_values = {}
    freshness = {}
    for index, source in enumerate(config.feature_source_ids):
        feature_values[source] = tuple(row.features[index * 2] if row.features[index * 2 + 1] == 0 else None
                                       for row in samples)
        freshness[source] = []
        for issue in config.issue_times:
            latest = _resolved_latest(dataset, issue, (source,)).get(source)
            freshness[source].append((issue - latest.event_time).total_seconds() / 3600 if latest else None)
    valid_target = tuple(row.target for row in samples)
    univariate = {}
    for index, source in enumerate(config.feature_source_ids):
        univariate[source] = univariate_oof_score(config.plan, samples, index * 2)
    return audit_features(feature_values, valid_target, evaluation_id="real-feature-audit-v1",
                          feature_registry=features, freshness_by_feature=freshness,
                          univariate_scores=univariate,
                          incremental_oof_gains=incremental_oof_gains,
                          recommendation_policy=config.feature_recommendation_policy)


def run_real_champion(dataset: RealDataset, config: RealChampionConfig, *, output_dir=None) -> RealChampionResult:
    """Refuse unresolved PIT/target contracts before constructing model evidence."""
    data_audit = audit_dataset(dataset, issue_time=config.target_contract.forecast_issue_time,
                               forecast_horizons=(config.target_contract.horizon,))
    if not data_audit.ready_for_forecast:
        raise RealChampionRefusal("data audit refused forecast run: " + "; ".join(data_audit.errors))
    try:
        TargetConfig(config.target_contract, config.target_source_id,
                     config.target_source_record_id).validate(dataset.registry)
    except ValueError as exc:
        raise RealChampionRefusal("target contract refused: " + str(exc)) from exc
    features = config.feature_registry or _feature_registry(config.feature_source_ids, config.target_contract.unit)
    missing_feature_definitions = set(config.feature_source_ids) - {
        definition.feature_id for definition in features.definitions}
    if missing_feature_definitions:
        raise RealChampionRefusal("feature registry is missing configured sources: " +
                                  ", ".join(sorted(missing_feature_definitions)))
    samples = build_real_samples(dataset, config)
    if any(row.target is None for row in samples):
        raise RealChampionRefusal("target rows are missing at one or more OOF horizons")
    baselines = run_baselines(config.plan, samples, features.manifest_hash,
                              include_ridge=config.include_ridge)
    persistence_score = baselines.scoreboard.get("persistence", {}).get("metrics", {}).get("mae")
    gains = None
    if persistence_score is not None:
        # Compare each feature's fold-safe univariate OOF score with the
        # persistence OOF score. This is evidence for review, not selection.
        provisional = _feature_audit(dataset, config, features, samples)
        gains = {record.feature_id: persistence_score - record.univariate_oof_score
                 for record in provisional.records if record.univariate_oof_score is not None}
    feature_audit = _feature_audit(dataset, config, features, samples,
                                   incremental_oof_gains=gains)
    diagnostics = {
        "data_audit": data_audit.to_dict(), "feature_audit": feature_audit.to_dict(),
        "target_contract": {"target_name": config.target_contract.target_name,
                            "target_entity": config.target_contract.target_entity,
                            "unit": config.target_contract.unit,
                            "sign_convention": config.target_contract.sign_convention,
                            "horizon_seconds": config.target_contract.horizon.total_seconds()},
        "feature_manifest_hash": features.manifest_hash,
        "training_row_manifest_candidates": [data_manifest(fold.train) for fold in baselines.oof.folds],
        "selected_model": list(baselines.selected_model), "holdout_used": False,
        "refused": False,
    }
    result = RealChampionResult(dataset, data_audit, feature_audit, features, samples,
                                config.plan, baselines, diagnostics)
    if output_dir is not None:
        write_real_champion_artifacts(result, output_dir)
    return result


def write_real_champion_artifacts(result: RealChampionResult, output_dir: str | Path):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "scoreboard.json": {"capabilities": result.baselines.capabilities,
                            "selected_model": list(result.baselines.selected_model),
                            "candidates": result.baselines.scoreboard},
        "data_audit.json": result.data_audit.to_dict(),
        "feature_audit.json": result.feature_audit.to_dict(),
        "diagnostics.json": result.diagnostics,
    }
    for name, payload in payloads.items():
        (directory / name).write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    return tuple(directory / name for name in payloads)


def _duration(value):
    return timedelta(seconds=float(value))


def _target_contract(payload):
    return TargetContract(payload["target_name"], payload["target_entity"], payload["unit"], payload["sign_convention"],
                          _duration(payload["resolution_seconds"]), datetime.fromisoformat(payload["forecast_issue_time"]),
                          datetime.fromisoformat(payload["target_valid_time"]), _duration(payload["horizon_seconds"]),
                          payload["aggregation_rule"], payload["label_availability_rule"],
                          ForecastType(payload["forecast_type"]), tuple(payload["evaluation_metrics"]),
                          payload.get("country"), tuple(payload["border"]) if payload.get("border") else None,
                          tuple(payload.get("quantile_levels", ())), payload.get("scenario_support", False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit and run real-data baseline Champion evaluation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True, help="JSON configuration containing source/target/plan mappings")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = RealSourceConfig.from_dict(payload["source_config"])
    target = _target_contract(payload["target_contract"])
    plan_payload = payload["plan"]
    plan = RollingOrigin(datetime.fromisoformat(plan_payload["first_origin"]),
                         datetime.fromisoformat(plan_payload["holdout_start"]),
                         _duration(plan_payload["train_window_seconds"]), _duration(plan_payload["validation_window_seconds"]),
                         _duration(plan_payload["calibration_window_seconds"]), _duration(plan_payload["step_seconds"]),
                         _duration(plan_payload.get("purge_seconds", 0)), _duration(plan_payload.get("embargo_seconds", 0)),
                         _duration(plan_payload.get("label_delay_seconds", 0)))
    config = RealChampionConfig(target, payload["target_source_id"], payload["target_source_record_id"],
                                tuple(datetime.fromisoformat(value) for value in payload["issue_times"]), plan,
                                tuple(payload["feature_source_ids"]), payload.get("include_ridge", True),
                                feature_recommendation_policy=payload.get("feature_recommendation_policy"))
    dataset = load_real_dataset(args.input, source_config=source)
    result = run_real_champion(dataset, config, output_dir=args.output_dir)
    print("real champion baseline run complete")
    for path in (Path(args.output_dir) / "data_audit.json", Path(args.output_dir) / "feature_audit.json",
                 Path(args.output_dir) / "scoreboard.json", Path(args.output_dir) / "diagnostics.json"):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
