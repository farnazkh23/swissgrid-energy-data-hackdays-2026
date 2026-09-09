"""Feature diagnostics; correlation is evidence, never an automatic selector."""
from dataclasses import dataclass
from math import log, sqrt, isfinite
from statistics import mean, pstdev
from collections import Counter
from collections.abc import Mapping

from .feature_registry import FeatureAudit as RegistryFeatureAudit
from .metrics import mae
from .splits import Sample


@dataclass(frozen=True, slots=True)
class FeatureAuditRecord:
    feature_id: str
    pearson: float | None
    spearman: float | None
    mutual_information: float | None
    univariate_oof_score: float | None
    incremental_oof_gain: float | None
    stability_across_folds: float | None
    missingness: float | None
    freshness: dict | None
    leakage_risk: str
    keep_drop: str = "UNDECIDED"

    def __post_init__(self):
        if not self.feature_id or self.keep_drop not in ("KEEP", "DROP", "UNDECIDED"):
            raise ValueError("invalid feature audit identity or decision")
        for value in (self.pearson, self.spearman, self.mutual_information,
                      self.univariate_oof_score, self.incremental_oof_gain,
                      self.stability_across_folds, self.missingness):
            if value is not None and not isfinite(value):
                raise ValueError("feature audit values must be finite")
        if self.missingness is not None and not 0 <= self.missingness <= 1:
            raise ValueError("missingness must be a fraction")

    def to_dict(self):
        return {"feature_id": self.feature_id, "pearson": self.pearson,
                "spearman": self.spearman, "mutual_information": self.mutual_information,
                "univariate_oof_score": self.univariate_oof_score,
                "incremental_oof_gain": self.incremental_oof_gain,
                "stability_across_folds": self.stability_across_folds,
                "missingness": self.missingness, "freshness": self.freshness,
                "leakage_risk": self.leakage_risk, "keep_drop": self.keep_drop}


@dataclass(frozen=True, slots=True)
class FeatureAuditReport:
    evaluation_id: str
    records: tuple[FeatureAuditRecord, ...]
    selection_policy: str
    warnings: tuple[str, ...]

    def to_dict(self):
        return {"evaluation_id": self.evaluation_id, "records": [record.to_dict() for record in self.records],
                "selection_policy": self.selection_policy, "warnings": list(self.warnings)}


def _pairs(values, target):
    pairs = [(float(x), float(y)) for x, y in zip(values, target)
             if x is not None and y is not None and isfinite(float(x)) and isfinite(float(y))]
    return tuple(pairs)


def _pearson(pairs):
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _mutual_information(pairs, bins=5):
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    def bucket(values, value):
        low, high = min(values), max(values)
        if low == high:
            return 0
        return min(bins - 1, int((value - low) / ((high - low) / bins)))
    joint = Counter((bucket(xs, x), bucket(ys, y)) for x, y in pairs)
    x_count = Counter(key[0] for key in joint for _ in range(joint[key]))
    y_count = Counter(key[1] for key in joint for _ in range(joint[key]))
    total = len(pairs)
    return sum((count / total) * log((count * total) / (x_count[x] * y_count[y]))
               for (x, y), count in joint.items())


def _recommend(record, policy):
    """Make a review recommendation only from explicitly requested evidence."""
    if policy is None:
        return "UNDECIDED"
    if not isinstance(policy, Mapping):
        raise ValueError("recommendation_policy must be an object")
    if record.leakage_risk.lower() in set(policy.get("drop_leakage_risk", ("high", "critical"))):
        return "DROP"
    definition = policy.get("enabled")
    if definition is False:
        return "DROP"
    required = (record.incremental_oof_gain, record.stability_across_folds,
                record.univariate_oof_score)
    if policy.get("require_oof_evidence", True) and any(value is None for value in required):
        return "UNDECIDED"
    if record.missingness is None or record.missingness > float(policy.get("max_missingness", 1.0)):
        return "DROP"
    if (record.incremental_oof_gain is not None and
            record.incremental_oof_gain < float(policy.get("min_incremental_oof_gain", 0.0))):
        return "DROP"
    if (record.stability_across_folds is not None and
            record.stability_across_folds < float(policy.get("min_fold_stability", 0.0))):
        return "DROP"
    if (record.freshness is not None and policy.get("max_freshness_hours") is not None and
            record.freshness.get("max", float("inf")) > float(policy["max_freshness_hours"])):
        return "DROP"
    return "KEEP"


def univariate_oof_score(plan, rows: tuple[Sample, ...], feature_index: int) -> float:
    """Fold-safe conditional-mean OOF MAE for one feature column."""
    folds = plan.split(rows)
    truth, predictions = [], []
    for fold in folds:
        train = [row for row in fold.train if row.target is not None]
        global_mean = mean(row.target for row in train)
        groups = {}
        for row in train:
            groups.setdefault(row.features[feature_index], []).append(row.target)
        for row in fold.validation:
            values = groups.get(row.features[feature_index])
            predictions.append(mean(values) if values else global_mean)
            truth.append(row.target)
    return mae(truth, predictions)


def audit_features(feature_values: dict[str, tuple], target_values: tuple, *, evaluation_id: str,
                   feature_registry=None, freshness_by_feature=None, fold_scores=None,
                   univariate_scores=None, baseline_oof_score=None, full_oof_score=None,
                   incremental_oof_gains=None, recommendation_policy=None) -> FeatureAuditReport:
    if not feature_values or not target_values or not evaluation_id:
        raise ValueError("feature values, target values, and evaluation identity are required")
    target_values = tuple(target_values)
    definitions = {definition.feature_id: definition for definition in feature_registry.definitions} if feature_registry else {}
    records = []
    for feature_id in sorted(feature_values):
        values = tuple(feature_values[feature_id])
        if len(values) != len(target_values):
            raise ValueError("feature and target lengths must match")
        pairs = _pairs(values, target_values)
        pearson = _pearson(pairs)
        spearman = _pearson(tuple(zip(_rank([pair[0] for pair in pairs]), _rank([pair[1] for pair in pairs])))) if len(pairs) >= 2 else None
        definition = definitions.get(feature_id)
        fold_stability = None
        if fold_scores and feature_id in fold_scores:
            scores = tuple(float(score) for score in fold_scores[feature_id])
            fold_stability = 1.0 / (1.0 + pstdev(scores)) if scores else None
        freshness = None if freshness_by_feature is None else freshness_by_feature.get(feature_id)
        if freshness is not None:
            freshness_values = tuple(float(value) for value in freshness
                                     if value is not None and isfinite(float(value)))
            freshness = {"mean": mean(freshness_values), "max": max(freshness_values),
                         "min": min(freshness_values)} if freshness_values else None
        record = FeatureAuditRecord(
            feature_id, pearson, spearman, _mutual_information(pairs),
            None if univariate_scores is None else univariate_scores.get(feature_id),
            (None if incremental_oof_gains is None else incremental_oof_gains.get(feature_id))
            if incremental_oof_gains is not None else
            (None if baseline_oof_score is None or full_oof_score is None else baseline_oof_score - full_oof_score),
            fold_stability, sum(value is None for value in values) / len(values), freshness,
            definition.leakage_risk if definition else "UNASSESSED",
            "UNDECIDED")
        records.append(FeatureAuditRecord(**{**record.to_dict(),
                                             "keep_drop": _recommend(record, recommendation_policy)}))
    policy_text = "Correlation is diagnostic only; feature decisions require fold-safe OOF evidence and stability."
    if recommendation_policy is not None:
        policy_text += " KEEP/DROP uses the explicit recommendation policy supplied by the caller."
    return FeatureAuditReport(evaluation_id, tuple(records),
                              policy_text,
                              ("Do not select features solely from correlation.",))


def registry_audits(report: FeatureAuditReport) -> tuple[RegistryFeatureAudit, ...]:
    """Convert diagnostics to the existing registry audit contract."""
    return tuple(RegistryFeatureAudit(record.feature_id, report.evaluation_id,
                                      pearson=record.pearson, spearman=record.spearman,
                                      mutual_information=record.mutual_information,
                                      univariate_score=record.univariate_oof_score,
                                      oof_gain=record.incremental_oof_gain,
                                      stability=record.stability_across_folds,
                                      keep_drop=record.keep_drop)
                 for record in report.records)
