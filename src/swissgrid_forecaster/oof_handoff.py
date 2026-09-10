"""Out-of-fold residual handoff to the probabilistic distribution teammate.

Builds the challenge OOF handoff with one row per (fold, issue time) and the
four-dimensional residual vector ordered exactly as

    [AT_residual, DE_residual, FR_residual, IT_residual]

from true out-of-fold predictions only. If the best Champion differs by
target, the per-target Champion predictions are used. The written artifacts
are ``oof_predictions.csv`` and ``oof_residual_summary.json``.
"""
import csv
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean

from .oof import OOFResult
from .target_contract import InvalidOutputTargets, TARGETS, validate_output_targets

OOF_HANDOFF_SCHEMA = "oof-handoff.v1"
OOF_HANDOFF_COLUMNS = ("timestamp", "fold_id", "horizon", "issue_time",
                       "AT_actual", "AT_pred", "AT_residual",
                       "DE_actual", "DE_pred", "DE_residual",
                       "FR_actual", "FR_pred", "FR_residual",
                       "IT_actual", "IT_pred", "IT_residual")
RESIDUAL_VECTOR_ORDER: tuple[str, ...] = TARGETS
RESIDUAL_DEFINITION = "actual - prediction"


def _champion_records(oof: OOFResult, champion):
    if not isinstance(champion, tuple) or len(champion) != 2:
        raise ValueError("champion must be a (model_id, model_version) tuple")
    if champion not in oof.candidate_identities:
        raise ValueError(f"champion {champion[0]} is not an OOF candidate for this target")
    return tuple(record for record in oof.records
                 if (record.model_id, record.version) == champion)


def _point_prediction(record):
    values = tuple(record.prediction.values)
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not isfinite(v) for v in values):
        raise ValueError(f"non-finite OOF prediction for {record.model_id} at {record.issue_time}")
    return mean(values)


def build_oof_handoff(oof_by_target, champions_by_target):
    """Build handoff rows from per-target OOF results and per-target Champions.

    ``oof_by_target`` maps each scored entity (exactly AT, DE, FR, IT) to a
    validated ``OOFResult``; ``champions_by_target`` maps the same entities to
    their per-target Champion ``(model_id, model_version)``.
    """
    if not isinstance(oof_by_target, dict) or not isinstance(champions_by_target, dict):
        raise ValueError("oof_by_target and champions_by_target must be mappings keyed by target")
    if "CH" in oof_by_target or "CH" in champions_by_target:
        raise InvalidOutputTargets("CH is an input/context country and has no OOF forecast output")
    if set(oof_by_target) != set(TARGETS) or set(champions_by_target) != set(TARGETS):
        raise InvalidOutputTargets(
            "OOF handoff requires exactly the four scored targets " + ", ".join(TARGETS))
    validate_output_targets(TARGETS)
    per_target = {}
    for entity in TARGETS:
        oof = oof_by_target[entity]
        if not isinstance(oof, OOFResult):
            raise ValueError(f"{entity} OOF evidence must be an OOFResult")
        records = _champion_records(oof, tuple(champions_by_target[entity]))
        if not records:
            raise ValueError(f"{entity} has no OOF records for its champion")
        entries = {}
        for record in records:
            actual = oof.truth(record.fold_id, record.row_id)
            if actual is None or not isfinite(actual):
                raise ValueError(f"{entity} OOF truth missing at {record.issue_time}")
            prediction = _point_prediction(record)
            key = (record.fold_id, record.row_id)
            if key in entries:
                raise ValueError(f"duplicate {entity} OOF prediction for {key}")
            entries[key] = (record, actual, prediction)
        per_target[entity] = entries
    reference = {key for key in per_target[TARGETS[0]]}
    for entity in TARGETS[1:]:
        if {key for key in per_target[entity]} != reference:
            raise ValueError("OOF rows must align across all four targets (same folds and issue times)")
    rows = []
    for fold_id, row_id in sorted(reference, key=lambda key: (per_target[TARGETS[0]][key][0].issue_time, key[0])):
        first = per_target[TARGETS[0]][(fold_id, row_id)][0]
        horizon_seconds = (first.target_time - first.issue_time).total_seconds()
        if horizon_seconds <= 0:
            raise ValueError("OOF horizon must be positive")
        # Persist horizon in hours; ResidualPanel stores the same duration as
        # timedelta, so the CSV has one explicit, unambiguous unit. Fractional
        # hours remain valid for non-hourly generic OOF contracts.
        horizon = horizon_seconds / 3600
        if horizon.is_integer():
            horizon = int(horizon)
        row = {"timestamp": first.target_time.isoformat(), "fold_id": fold_id,
               "horizon": horizon, "issue_time": first.issue_time.isoformat()}
        for entity in TARGETS:
            record, actual, prediction = per_target[entity][(fold_id, row_id)]
            if (record.issue_time, record.target_time) != (first.issue_time, first.target_time):
                raise ValueError("OOF issue/target times must agree across targets")
            row[f"{entity}_actual"] = actual
            row[f"{entity}_pred"] = prediction
            row[f"{entity}_residual"] = actual - prediction
        rows.append(row)
    return tuple(rows)


def oof_handoff_column_order(rows):
    """Assert the persisted handoff preserves the exact AT/DE/FR/IT columns."""
    if not rows:
        raise ValueError("empty OOF handoff")
    for row in rows:
        if tuple(row) != OOF_HANDOFF_COLUMNS:
            raise ValueError("OOF handoff columns must be exactly " + ", ".join(OOF_HANDOFF_COLUMNS))
    return OOF_HANDOFF_COLUMNS


def residual_summary(rows, champions_by_target):
    """Per-target residual statistics plus the documented target order."""
    if "CH" in champions_by_target:
        raise InvalidOutputTargets("CH must not appear as an OOF handoff target")
    if set(champions_by_target) != set(TARGETS):
        raise InvalidOutputTargets("residual summary requires champions for exactly " + ", ".join(TARGETS))
    columns = oof_handoff_column_order(rows)
    summary = {"schema_version": OOF_HANDOFF_SCHEMA,
               "target_order": list(TARGETS),
               "residual_vector_order": list(RESIDUAL_VECTOR_ORDER),
               "residual_definition": RESIDUAL_DEFINITION,
               "horizon_unit": "hours",
               "handoff_columns": list(columns),
               "rows": len(rows),
               "folds": sorted({row["fold_id"] for row in rows}),
               "date_range": [min(row["timestamp"] for row in rows),
                              max(row["timestamp"] for row in rows)],
               "champions": {entity: {"model_id": champions_by_target[entity][0],
                                      "model_version": champions_by_target[entity][1]}
                             for entity in TARGETS},
               "per_target": {}}
    for entity in TARGETS:
        actuals = tuple(row[f"{entity}_actual"] for row in rows)
        predictions = tuple(row[f"{entity}_pred"] for row in rows)
        residuals = tuple(row[f"{entity}_residual"] for row in rows)
        count = len(residuals)
        residual_mean = sum(residuals) / count
        summary["per_target"][entity] = {
            "count": count,
            "mae": sum(abs(r) for r in residuals) / count,
            "rmse": sqrt(sum(r * r for r in residuals) / count),
            "bias": sum(p - a for a, p in zip(actuals, predictions)) / count,
            "residual_mean": residual_mean,
            "residual_std": sqrt(sum((r - residual_mean) ** 2 for r in residuals) / count),
        }
    return summary


def write_oof_handoff(rows, champions_by_target, output_dir):
    """Write oof_predictions.csv and oof_residual_summary.json atomically-ish."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    columns = oof_handoff_column_order(rows)
    predictions_path = directory / "oof_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = residual_summary(rows, champions_by_target)
    summary_path = directory / "oof_residual_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    return predictions_path, summary_path
