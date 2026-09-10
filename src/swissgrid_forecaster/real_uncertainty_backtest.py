"""CLI: fit/evaluate the G4 uncertainty layer on real rolling OOF residuals.

Loads `artifacts/real_backtest/oof_predictions.csv` — already-validated,
genuine out-of-fold residuals from the persistence baseline (see
docs/PROBABILITY_HANDOFF.md). Only the OOF residual columns in that file are
ever used to fit or calibrate; nothing here reads training-fold residuals,
and evaluation always holds out later weeks from earlier ones.
"""
import argparse
from csv import DictReader
from datetime import datetime, timedelta
import json
from pathlib import Path

from .sampler_evaluation import compare_samplers, evaluate_sampler, weekly_folds
from .uncertainty_model import ResidualPanel, calibrate_scale, fit_uncertainty_model, sample_distribution

TARGET_NAMES = ("CH", "DE", "FR", "IT")
METHODS = {
    "independent_gaussian": {},
    "correlated_gaussian": {},
    "student_t": {"degrees_of_freedom": 6.0},
    "empirical_bootstrap": {},
}


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_residual_panel(path, targets=TARGET_NAMES) -> ResidualPanel:
    """Build a ResidualPanel directly from the CSV's `*_residual` columns."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for record in DictReader(handle):
            issue_time = _parse_time(record["issue_time"])
            target_time = _parse_time(record["timestamp"])
            horizon = timedelta(hours=float(record["horizon"]))
            residuals = tuple(float(record[f"{name}_residual"]) for name in targets)
            rows.append((issue_time, target_time, horizon, residuals))
    return ResidualPanel(tuple(targets), tuple(rows))


def load_point_forecasts(path, targets=TARGET_NAMES) -> dict:
    """timestamp -> {target: point forecast}, for demo sampling around real mu."""
    forecasts = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for record in DictReader(handle):
            target_time = _parse_time(record["timestamp"])
            forecasts[target_time] = {name: float(record[f"{name}_pred"]) for name in targets}
    return forecasts


def compare_methods(panel: ResidualPanel, *, seed: int = 2026, n_samples: int = 300,
                    week_hours: int = 168, baseline: str = "independent_gaussian",
                    methods: dict = METHODS) -> tuple:
    """Walk-forward comparison of every requested method; see SamplerEvaluation.to_row()."""
    evaluations = {name: evaluate_sampler(panel, name, seed=seed, n_samples=n_samples,
                                          week_hours=week_hours, **kwargs)
                  for name, kwargs in methods.items()}
    return compare_samplers(evaluations, baseline=baseline)


def fit_calibrated_champion(panel: ResidualPanel, method: str, *, week_hours: int = 168,
                            fit_weeks: int, calibration_weeks: int, degrees_of_freedom=None,
                            target_coverage: float = 0.8, seed: int = 2026, n_samples: int = 300):
    """Nested split: fit on the earliest weeks, calibrate scale on the next, leave the rest for sampling.

    Never reuses fit rows for calibration, or calibration rows for the demo sample.
    """
    folds = weekly_folds(panel, week_hours=week_hours)
    if fit_weeks + calibration_weeks >= len(folds):
        raise ValueError("not enough weekly folds for the requested fit/calibration/demo split")
    fit_rows = tuple(row for fold in folds[:fit_weeks] for row in fold)
    calibration_rows = tuple(row for fold in folds[fit_weeks:fit_weeks + calibration_weeks] for row in fold)
    demo_folds = folds[fit_weeks + calibration_weeks:]
    fit_panel = ResidualPanel(panel.target_names, fit_rows)
    model = fit_uncertainty_model(fit_panel, method=method, fit_cutoff=calibration_rows[0][0],
                                  degrees_of_freedom=degrees_of_freedom)
    calibration_panel = ResidualPanel(panel.target_names, calibration_rows)
    calibrated = calibrate_scale(model, calibration_panel, target_coverage=target_coverage,
                                 seed=seed, n_samples=n_samples)
    return calibrated, demo_folds


def sample_folds(model, demo_folds, point_forecasts: dict, *, seed: int = 2026, n_samples: int = 300) -> tuple:
    """One entry per demo timestamp: `{timestamp, <target>_samples: [n_samples ints]}`."""
    results = []
    index = 0
    for fold in demo_folds:
        for _, target_time, _, _ in fold:
            if target_time not in point_forecasts:
                raise ValueError("missing point forecast for demo timestamp")
            draws = sample_distribution(model, point_forecasts[target_time], seed=seed + index, n_samples=n_samples)
            entry = {"timestamp": target_time.isoformat()}
            entry.update({f"{name}_samples": list(values) for name, values in draws.items()})
            results.append(entry)
            index += 1
    return tuple(results)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit/evaluate the G4 uncertainty layer on real OOF residuals")
    parser.add_argument("--input", default="artifacts/real_backtest/oof_predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/real_backtest")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--student-t-dof", type=float, default=6.0)
    parser.add_argument("--fit-weeks", type=int, default=8)
    parser.add_argument("--calibration-weeks", type=int, default=2)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--champion-method", default="correlated_gaussian")
    args = parser.parse_args(argv)

    panel = load_residual_panel(args.input)
    methods = dict(METHODS)
    methods["student_t"] = {"degrees_of_freedom": args.student_t_dof}
    ranked = compare_methods(panel, seed=args.seed, n_samples=args.n_samples, methods=methods)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "uncertainty_method_comparison.json"
    comparison_path.write_text(json.dumps([row.to_row() for row in ranked], indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")

    champion_kwargs = methods[args.champion_method]
    calibrated, demo_folds = fit_calibrated_champion(
        panel, args.champion_method, fit_weeks=args.fit_weeks, calibration_weeks=args.calibration_weeks,
        target_coverage=args.target_coverage, seed=args.seed, n_samples=args.n_samples, **champion_kwargs)
    point_forecasts = load_point_forecasts(args.input)
    samples = sample_folds(calibrated, demo_folds, point_forecasts, seed=args.seed, n_samples=args.n_samples)
    samples_path = output_dir / "uncertainty_champion_samples.json"
    samples_path.write_text(json.dumps(samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("method comparison (best first):")
    for row in ranked:
        print(f"  {row.method}: mean_score={row.mean_score:.3f} capture={row.capture_rate:.3f} "
              f"sharpness={row.mean_sharpness:.1f} keep_drop={row.keep_drop}")
    print(f"champion={args.champion_method} written to {comparison_path} and {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
