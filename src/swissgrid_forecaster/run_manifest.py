"""Immutable, content-addressed metadata for future real-data runs."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import subprocess
from pathlib import Path

from .availability import nonempty, utc
from .manifest import validate_digest


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()
    return sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: str | Path = ".") -> str:
    try:
        result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repo, check=True,
                                capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("git commit could not be resolved") from exc
    value = result.stdout.strip()
    if not value:
        raise ValueError("git commit could not be resolved")
    return value


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    created_at: datetime
    git_commit: str
    config_hashes: dict
    dataset_hash: str
    source_manifests: tuple[dict, ...]
    target_contract_hash: str
    feature_manifest_hash: str
    model_identities: tuple[dict, ...]
    fit_cutoffs: tuple[dict, ...]
    fold_ids: tuple[str, ...]
    calibration_version: str | None
    random_seed: int
    artifact_hashes: dict
    warnings: tuple[str, ...]
    readiness_result: dict
    manifest_hash: str

    def __post_init__(self):
        nonempty(self.run_id, "run_id")
        object.__setattr__(self, "created_at", utc(self.created_at, "created_at"))
        nonempty(self.git_commit, "git_commit")
        validate_digest(self.dataset_hash)
        validate_digest(self.target_contract_hash)
        validate_digest(self.feature_manifest_hash)
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        for digest in tuple(self.config_hashes.values()) + tuple(self.artifact_hashes.values()):
            validate_digest(digest)
        object.__setattr__(self, "source_manifests", tuple(dict(item) for item in self.source_manifests))
        object.__setattr__(self, "model_identities", tuple(dict(item) for item in self.model_identities))
        object.__setattr__(self, "fit_cutoffs", tuple(dict(item) for item in self.fit_cutoffs))
        object.__setattr__(self, "fold_ids", tuple(self.fold_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if not isinstance(self.readiness_result, dict):
            raise ValueError("readiness_result must be an object")
        validate_digest(self.manifest_hash)
        if self.manifest_hash != canonical_hash(self._identity_dict()):
            raise ValueError("manifest hash mismatch")

    def _identity_dict(self):
        return {"run_id": self.run_id, "git_commit": self.git_commit,
                "config_hashes": self.config_hashes, "dataset_hash": self.dataset_hash,
                "source_manifests": list(self.source_manifests),
                "target_contract_hash": self.target_contract_hash,
                "feature_manifest_hash": self.feature_manifest_hash,
                "model_identities": list(self.model_identities), "fit_cutoffs": list(self.fit_cutoffs),
                "fold_ids": list(self.fold_ids), "calibration_version": self.calibration_version,
                "random_seed": self.random_seed, "artifact_hashes": self.artifact_hashes,
                "warnings": list(self.warnings), "readiness_result": self.readiness_result}

    def to_dict(self):
        return {**self._identity_dict(), "created_at": self.created_at.isoformat(),
                "manifest_hash": self.manifest_hash}

    @classmethod
    def from_dict(cls, payload: dict) -> "RunManifest":
        identity = dict(payload)
        identity.pop("created_at", None)
        identity.pop("manifest_hash", None)
        return cls(payload["run_id"], datetime.fromisoformat(payload["created_at"]), payload["git_commit"],
                   payload["config_hashes"], payload["dataset_hash"], tuple(payload["source_manifests"]),
                   payload["target_contract_hash"], payload["feature_manifest_hash"],
                   tuple(payload["model_identities"]), tuple(payload["fit_cutoffs"]), tuple(payload["fold_ids"]),
                   payload.get("calibration_version"), payload["random_seed"], payload["artifact_hashes"],
                   tuple(payload["warnings"]), payload["readiness_result"], payload["manifest_hash"])


def build_run_manifest(*, created_at: datetime, git_revision: str, config_hashes: dict,
                       dataset_hash: str, source_manifests=(), target_contract_hash: str,
                       feature_manifest_hash: str, model_identities=(), fit_cutoffs=(), fold_ids=(),
                       calibration_version=None, random_seed: int, artifact_hashes: dict,
                       warnings=(), readiness_result: dict) -> RunManifest:
    identity = {"git_commit": git_revision, "config_hashes": config_hashes,
                "dataset_hash": dataset_hash, "source_manifests": list(source_manifests),
                "target_contract_hash": target_contract_hash, "feature_manifest_hash": feature_manifest_hash,
                "model_identities": list(model_identities), "fit_cutoffs": list(fit_cutoffs),
                "fold_ids": list(fold_ids), "calibration_version": calibration_version,
                "random_seed": random_seed, "artifact_hashes": artifact_hashes,
                "warnings": list(warnings), "readiness_result": readiness_result}
    run_id = canonical_hash(identity)
    return RunManifest(run_id, created_at, git_revision, config_hashes, dataset_hash,
                       tuple(source_manifests), target_contract_hash, feature_manifest_hash,
                       tuple(model_identities), tuple(fit_cutoffs), tuple(fold_ids), calibration_version,
                       random_seed, artifact_hashes, tuple(warnings), readiness_result,
                       canonical_hash({"run_id": run_id, **identity}))


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
