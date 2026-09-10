"""One-command readiness, Champion, and acceptance orchestration."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

from .acceptance import evaluate_acceptance
from .readiness import evaluate_readiness
from .real_dataset_loader import load_real_dataset
from .real_source_adapter import RealSourceConfig
from .run_manifest import build_run_manifest, canonical_hash, file_hash, git_commit, write_run_manifest
from .run_real_champion import RealChampionConfig, _target_contract, run_real_champion
from .splits import RollingOrigin


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ref(value, base):
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        if path.is_file():
            return _json(path)
    return value


def _seconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _target(payload):
    value = dict(payload or {})
    aliases = {"resolution_seconds": "resolution", "forecast_issue_time": "issue_time",
               "target_valid_time": "valid_time", "horizon_seconds": "horizon",
               "aggregation_rule": "aggregation", "label_availability_rule": "label_availability"}
    for destination, source in aliases.items():
        if destination not in value and source in value:
            value[destination] = value[source]
    for field in ("resolution_seconds", "horizon_seconds"):
        number = _seconds(value.get(field))
        if number is not None:
            value[field] = number
    value.setdefault("forecast_type", "point")
    value.setdefault("evaluation_metrics", ["mae", "rmse", "bias"])
    return value


def normalize_config(path: str | Path) -> dict:
    path = Path(path)
    payload = _json(path)
    base = path.parent
    target = _target(_ref(payload.get("target_config_path", payload.get("target_config", payload.get("target_path", payload.get("target", {})))), base))
    mapping = _ref(payload.get("mapping_path", payload.get("column_mapping", payload.get("mapping", {}))), base)
    sources = _ref(payload.get("sources_path", payload.get("source_config", payload.get("sources", {}))), base)
    rules = _ref(payload.get("publication_rules_path", payload.get("publication_rules", {})), base)
    if isinstance(sources, dict) and isinstance(sources.get("source_config"), dict):
        source = dict(sources["source_config"])
    else:
        source = dict(sources or {})
    if isinstance(sources, dict) and isinstance(sources.get("sources"), list):
        source.setdefault("source_id", None)
        source["source_manifests"] = sources["sources"]
    source.setdefault("mapping", mapping)
    if isinstance(mapping, dict) and "columns" not in mapping and isinstance(mapping.get("mapping"), dict):
        mapping = mapping["mapping"]
        source["mapping"] = mapping
    normalized = dict(payload)
    normalized.update({"target_contract": target, "column_mapping": mapping,
                       "source_config": source, "publication_rules": rules})
    normalized["output_directory"] = str((base / normalized.get("output_directory", "artifacts/real_run")).resolve()) \
        if not Path(str(normalized.get("output_directory", "artifacts/real_run"))).is_absolute() else normalized.get("output_directory")
    if "dataset_path" in normalized:
        dataset = Path(normalized["dataset_path"])
        normalized["dataset_path"] = str((base / dataset).resolve()) if not dataset.is_absolute() else str(dataset)
    normalized["target_contract"] = _target(normalized["target_contract"])
    normalized["issue_times"] = tuple(normalized.get("issue_times", [normalized["target_contract"].get("forecast_issue_time")]))
    return normalized


def _plan(payload):
    value = payload
    def duration(name, aliases=()):
        raw = value.get(name)
        if raw is None:
            for alias in aliases:
                raw = value.get(alias)
                if raw is not None:
                    break
        return timedelta(seconds=float(raw or 0))
    return RollingOrigin(datetime.fromisoformat(value["first_origin"]), datetime.fromisoformat(value["holdout_start"]),
                         duration("train_window", ("train_window_seconds",)),
                         duration("validation_window", ("validation_window_seconds",)),
                         duration("calibration_window", ("calibration_window_seconds",)),
                         duration("step", ("step_seconds",)), duration("purge", ("purge_seconds",)),
                         duration("embargo", ("embargo_seconds",)), duration("label_delay", ("label_delay_seconds",)))


def _load_dataset(config):
    path = config.get("dataset_path")
    if not path or not Path(path).is_file():
        return None
    return load_real_dataset(path, source_config=RealSourceConfig.from_dict(config["source_config"]),
                             format=config.get("input_format"))


def _print_report(report, as_json=False):
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2) if as_json else report.human_text())


def readiness_command(config_path, as_json=False):
    config = normalize_config(config_path)
    run_dir = config.get("readiness_dataset_path")
    if run_dir:
        candidate = Path(config_path).parent / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)
        if candidate.is_dir() and (candidate / "forecast.json").exists():
            report = evaluate_readiness(config=config, run_dir=candidate)
            _print_report(report, as_json)
            return 0 if report.ready else 1
    try:
        dataset = _load_dataset(config)
    except Exception as exc:
        print(f"READINESS: FAIL\ndataset_load [mandatory]: {exc}")
        return 1
    report = evaluate_readiness(dataset, config=config)
    _print_report(report, as_json)
    return 0 if report.ready else 1


def run_champion_command(config_path):
    config = normalize_config(config_path)
    dataset = _load_dataset(config)
    if dataset is None:
        print("RUN: FAIL\ndataset_path is missing or does not identify a file")
        return 1
    readiness = evaluate_readiness(dataset, config=config)
    if not readiness.ready:
        print(readiness.human_text())
        return 1
    target = _target_contract(config["target_contract"])
    champion = RealChampionConfig(target, config["target_source_id"], config["target_source_record_id"],
                                  tuple(datetime.fromisoformat(value) for value in config["issue_times"]),
                                  _plan(config["fold_config"]), tuple(config["feature_source_ids"]),
                                  config.get("include_ridge", True),
                                  feature_recommendation_policy=config.get("feature_recommendation_policy"))
    output = Path(config["output_directory"])
    result = run_real_champion(dataset, champion, output_dir=output)
    artifact_hashes = {path.name: file_hash(path) for path in output.glob("*.json") if path.name != "run_manifest.json"}
    source_manifests = tuple(config.get("source_config", {}).get("source_manifests", ()))
    manifest = build_run_manifest(
        created_at=datetime.fromisoformat(config["created_at"]), git_revision=git_commit(),
        config_hashes={"run": canonical_hash(config), **{key: canonical_hash(config.get(key)) for key in ("target_contract", "column_mapping", "source_config", "publication_rules", "fold_config")}},
        dataset_hash=dataset.raw_receipt.manifest.sha256 if dataset.raw_receipt else canonical_hash(dataset.raw_rows),
        source_manifests=source_manifests, target_contract_hash=canonical_hash(config["target_contract"]),
        feature_manifest_hash=result.features.manifest_hash,
        model_identities=tuple({"model_id": key, "model_version": value.get("model_version")}
                               for key, value in result.baselines.scoreboard.items() if isinstance(value, dict) and value.get("model_version")),
        fit_cutoffs=tuple({"fold_id": fold.fold_id, "fit_cutoff": fold.fit_cutoff.isoformat()} for fold in result.baselines.oof.folds),
        fold_ids=tuple(fold.fold_id for fold in result.baselines.oof.folds), random_seed=int(config["random_seed"]),
        artifact_hashes=artifact_hashes, warnings=tuple(result.data_audit.warnings), readiness_result=readiness.to_dict())
    write_run_manifest(manifest, output / "run_manifest.json")
    print("real Champion run complete")
    print(json.dumps({"run_id": manifest.run_id, "output_directory": str(output), "artifacts": sorted(artifact_hashes) + ["run_manifest.json"]}, sort_keys=True))
    return 0


def acceptance_command(run_dir, allow_fallback=False, as_json=False):
    report = evaluate_acceptance(run_dir, allow_fallback=allow_fallback)
    _print_report(report, as_json)
    return 0 if report.passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="swissgrid_forecaster.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    ready = sub.add_parser("readiness"); ready.add_argument("--config", required=True); ready.add_argument("--json", action="store_true")
    run = sub.add_parser("run-champion"); run.add_argument("--config", required=True)
    accept = sub.add_parser("acceptance"); accept.add_argument("--run", required=True); accept.add_argument("--allow-fallback", action="store_true"); accept.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "readiness": return readiness_command(args.config, args.json)
    if args.command == "run-champion": return run_champion_command(args.config)
    return acceptance_command(args.run, args.allow_fallback, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
