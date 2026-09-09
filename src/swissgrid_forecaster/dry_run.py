"""CLI for one deterministic mock forecast run."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from .api_schema import build_api_payload
from .pipeline import PipelineConfig, run_mock_pipeline


def write_mock_artifacts(result, output_dir: str | Path) -> tuple[Path, ...]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "forecast.json": result.forecast.to_dict(),
        "histogram.json": result.histogram.to_dict(),
        "scoreboard.json": {"capabilities": result.baselines.capabilities,
                            "selected_model": list(result.baselines.selected_model),
                            "candidates": result.baselines.scoreboard},
        "provenance.json": result.provenance,
        "run_manifest.json": result.run_manifest,
    }
    paths = []
    for name, payload in payloads.items():
        path = directory / name
        path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(path)
    # Validate the exact transport contract before handing artifacts to a caller.
    build_api_payload(result.forecast, result.histogram)
    return tuple(paths)


def run_and_write(*, output_dir, issue_time, seed=20260909, include_ridge=True):
    config = PipelineConfig(issue_time=issue_time, seed=seed, include_ridge=include_ridge)
    result = run_mock_pipeline(config)
    write_mock_artifacts(result, output_dir)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the deterministic mock forecast pipeline")
    parser.add_argument("--issue-time", required=True, help="aware ISO-8601 issue time")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output-dir", default="artifacts/mock_run")
    parser.add_argument("--no-ridge", action="store_true")
    args = parser.parse_args(argv)
    issue_time = datetime.fromisoformat(args.issue_time)
    result = run_and_write(output_dir=args.output_dir, issue_time=issue_time,
                           seed=args.seed, include_ridge=not args.no_ridge)
    print("mock dry run complete")
    for name in result.run_manifest["artifacts"]:
        print(f"{Path(args.output_dir) / name}")
    print(f"selected_model={result.baselines.selected_model[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
