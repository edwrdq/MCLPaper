"""Run JAX adaptive-noise experiments end-to-end and compute Wilcoxon tests.

This script orchestrates:
1) Regular JAX suite run
2) Adaptive-noise ablation run
3) Adaptive-noise sensitivity sweep run

Then it performs paired Wilcoxon signed-rank tests on per-trial metrics by
comparing adaptive-noise-related conditions against baseline in each scenario.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    scipy_wilcoxon = None


@dataclass
class SuiteResult:
    suite_name: str
    run_dir: Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive-noise JAX studies + Wilcoxon analysis")
    parser.add_argument("--output-root", type=str, default="research_runs_jax_pipeline")
    parser.add_argument("--trials-per-condition", type=int, default=20)
    parser.add_argument("--trial-batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--particles", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--room-size", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--ray-beams", type=int, default=9)
    parser.add_argument("--weight-beam-stride", type=int, default=1)
    parser.add_argument("--safety-margin", type=float, default=0.5)
    parser.add_argument("--init-radius", type=float, default=2.0)
    parser.add_argument(
        "--regular-study-mode",
        choices=["standard", "collision_ablation"],
        default="collision_ablation",
        help="Which existing suite to use for the regular run.",
    )

    parser.add_argument("--adaptive-noise-alpha", type=float, default=0.5)
    parser.add_argument("--adaptive-noise-beta", type=float, default=0.2)
    parser.add_argument("--adaptive-noise-smoothing", action="store_true")
    parser.add_argument("--adaptive-noise-damping", type=float, default=0.0)

    parser.add_argument("--ablation-regime", choices=["no_collision", "with_collision", "both"], default="both")
    parser.add_argument("--ablation-include-smoothing-toggle", action="store_true")

    parser.add_argument(
        "--single-stage-sensitivity",
        action="store_true",
        help="Disable two-stage sensitivity workflow and run one full grid sweep only.",
    )
    parser.add_argument(
        "--sensitivity-regime",
        choices=["no_collision", "with_collision", "both"],
        default="both",
        help="Confirmatory sensitivity regime(s).",
    )
    parser.add_argument(
        "--sensitivity-stage1-regime",
        choices=["no_collision", "with_collision", "both"],
        default="with_collision",
        help="Exploratory stage-1 sensitivity regime(s) used to rank top-K configs.",
    )
    parser.add_argument("--sensitivity-stage1-trials", type=int, default=8)
    parser.add_argument("--sensitivity-stage1-steps", type=int, default=2_000)
    parser.add_argument("--sensitivity-top-k", type=int, default=8)
    parser.add_argument(
        "--sensitivity-stage1-ranking-metric",
        choices=["auto", "mean_collision_peak_rmse", "mean_of_mean_rmse", "mean_final_rmse"],
        default="auto",
    )
    parser.add_argument("--sensitivity-alpha-grid", type=str, default="0.02,0.05,0.1,0.2")
    parser.add_argument("--sensitivity-beta-grid", type=str, default="0.005,0.01,0.05,0.1")
    parser.add_argument("--sensitivity-gamma-grid", type=str, default="0.0,0.6,0.8")
    parser.add_argument("--sensitivity-include-baseline", action="store_true")

    parser.add_argument("--collision-probability", type=float, default=0.01)
    parser.add_argument("--collision-speed-loss-min", type=float, default=0.3)
    parser.add_argument("--collision-speed-loss-max", type=float, default=0.8)
    parser.add_argument("--collision-ang-kick-max", type=float, default=0.8)
    parser.add_argument("--collision-backstep-max", type=float, default=0.2)
    parser.add_argument("--collision-peak-window", type=int, default=50)
    parser.add_argument("--collision-recovery-window", type=int, default=200)
    parser.add_argument("--collision-recovery-pre-window", type=int, default=20)
    parser.add_argument("--collision-recovery-hold", type=int, default=5)
    parser.add_argument("--collision-recovery-threshold-abs", type=float, default=0.03)
    parser.add_argument("--collision-recovery-threshold-factor", type=float, default=2.0)

    parser.add_argument(
        "--independent-condition-seeds",
        action="store_true",
        help="Disable paired condition seeds (not recommended for paired Wilcoxon).",
    )
    parser.add_argument("--no-rich", action="store_true")
    return parser.parse_args()


def discover_new_run_dir(parent: Path, before: set[Path]) -> Path:
    after = {p for p in parent.glob("aggressive_suite_*") if p.is_dir()}
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if len(new_dirs) == 1:
        return new_dirs[0]
    if len(new_dirs) > 1:
        return new_dirs[-1]
    existing = sorted(after, key=lambda p: p.stat().st_mtime)
    if not existing:
        raise RuntimeError(f"No aggressive_suite_* output directory found under {parent}")
    return existing[-1]


def run_suite(
    suite_name: str,
    repo_root: Path,
    output_parent: Path,
    shared_args: list[str],
    mode_args: list[str],
) -> SuiteResult:
    output_parent.mkdir(parents=True, exist_ok=True)
    before = {p for p in output_parent.glob("aggressive_suite_*") if p.is_dir()}
    cmd = [
        sys.executable,
        str(repo_root / "jax_benchmark" / "jax_research_benchmark.py"),
        *shared_args,
        *mode_args,
        "--output-dir",
        str(output_parent),
    ]
    print(f"[run] {suite_name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    run_dir = discover_new_run_dir(output_parent, before)
    print(f"[done] {suite_name}: {run_dir}")
    return SuiteResult(suite_name=suite_name, run_dir=run_dir, command=cmd)


def read_metrics_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    rows.sort(key=lambda r: int(r.get("trial", 0.0)))
    return rows


def collect_condition_rows(run_dir: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    mapping: dict[tuple[str, str], list[dict[str, float]]] = {}
    for csv_path in run_dir.rglob("metrics.csv"):
        parts = csv_path.relative_to(run_dir).parts
        if len(parts) == 2:
            scenario = "standard"
            condition = parts[0]
        elif len(parts) == 3:
            scenario = parts[0]
            condition = parts[1]
        else:
            continue
        mapping[(scenario, condition)] = read_metrics_csv(csv_path)
    return mapping


def _maybe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _format_config_triplet(alpha: float, beta: float, gamma: float) -> str:
    return f"{alpha:.10g}:{beta:.10g}:{gamma:.10g}"


def select_top_sensitivity_configs(
    sensitivity_explore_run_dir: Path,
    *,
    stage1_regime: str,
    ranking_metric: str,
    top_k: int,
) -> tuple[list[tuple[float, float, float]], list[dict[str, str]]]:
    summary_rows = read_summary_rows(sensitivity_explore_run_dir / "summary.csv")
    if not summary_rows:
        raise RuntimeError("Exploratory sensitivity summary.csv is empty.")

    candidates: list[dict[str, str]] = []
    for row in summary_rows:
        scenario = row.get("scenario", "standard").strip() or "standard"
        if stage1_regime == "no_collision" and scenario != "no_collision":
            continue
        if stage1_regime == "with_collision" and scenario != "with_collision":
            continue
        condition = row.get("condition", "").strip()
        alpha = _maybe_float(row.get("adaptive_noise_alpha"))
        beta = _maybe_float(row.get("adaptive_noise_beta"))
        gamma = _maybe_float(row.get("adaptive_noise_gamma"))
        if condition == "baseline":
            continue
        if not np.isfinite(alpha) or not np.isfinite(beta) or not np.isfinite(gamma):
            continue
        if not condition.startswith("noise_a"):
            continue
        candidates.append(row)

    if not candidates:
        raise RuntimeError("No exploratory sensitivity candidates found to rank.")

    metric = ranking_metric
    if ranking_metric == "auto":
        has_collision_peak = any(np.isfinite(_maybe_float(r.get("mean_collision_peak_rmse"))) for r in candidates)
        metric = "mean_collision_peak_rmse" if has_collision_peak else "mean_of_mean_rmse"

    def score(row: dict[str, str]) -> tuple[float, float]:
        primary = _maybe_float(row.get(metric))
        fallback = _maybe_float(row.get("mean_of_mean_rmse"))
        if not np.isfinite(primary):
            primary = float("inf")
        if not np.isfinite(fallback):
            fallback = float("inf")
        return (primary, fallback)

    ranked = sorted(candidates, key=score)
    top_rows = ranked[: max(1, int(top_k))]
    configs: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for row in top_rows:
        alpha = float(_maybe_float(row.get("adaptive_noise_alpha")))
        beta = float(_maybe_float(row.get("adaptive_noise_beta")))
        gamma = float(_maybe_float(row.get("adaptive_noise_gamma")))
        cfg = (alpha, beta, gamma)
        if cfg in seen:
            continue
        seen.add(cfg)
        configs.append(cfg)
    if not configs:
        raise RuntimeError("Top-K ranking produced no valid sensitivity configs.")
    return configs, top_rows


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(values.shape[0], dtype=float)
    sorted_vals = values[order]
    n = sorted_vals.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int, float, str]:
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shape for paired Wilcoxon.")
    diffs = np.asarray(x - y, dtype=float)
    finite = np.isfinite(diffs)
    diffs = diffs[finite]
    diffs = diffs[diffs != 0.0]
    n = int(diffs.shape[0])
    if n == 0:
        return 0.0, 1.0, 0, 0.0, "degenerate_all_zero"

    if scipy_wilcoxon is not None:
        result = scipy_wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
        ranks = rankdata_average(np.abs(diffs))
        w_plus = float(np.sum(ranks[diffs > 0.0]))
        w_minus = float(np.sum(ranks[diffs < 0.0]))
        denom = n * (n + 1) / 2.0
        rbc = 0.0 if denom <= 0 else (w_plus - w_minus) / denom
        return float(result.statistic), float(result.pvalue), n, float(rbc), "scipy"

    abs_diffs = np.abs(diffs)
    ranks = rankdata_average(abs_diffs)
    w_plus = float(np.sum(ranks[diffs > 0.0]))
    w_minus = float(np.sum(ranks[diffs < 0.0]))
    w_stat = float(min(w_plus, w_minus))

    counts = np.unique(abs_diffs, return_counts=True)[1]
    tie_term = float(np.sum([c * (c + 1) * (2 * c + 1) for c in counts if c > 1]))
    mean_w_plus = n * (n + 1) / 4.0
    var_w_plus = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_term) / 24.0
    if var_w_plus <= 0.0:
        p_two = 1.0
    else:
        z = (w_plus - mean_w_plus - 0.5 * np.sign(w_plus - mean_w_plus)) / math.sqrt(var_w_plus)
        p_two = math.erfc(abs(z) / math.sqrt(2.0))
    denom = n * (n + 1) / 2.0
    rbc = 0.0 if denom <= 0 else (w_plus - w_minus) / denom
    return w_stat, float(min(max(p_two, 0.0), 1.0)), n, float(rbc), "normal_approx"


def paired_metric_arrays(
    rows_a: list[dict[str, float]],
    rows_b: list[dict[str, float]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_trial_a = {int(r["trial"]): float(r[metric]) for r in rows_a if "trial" in r and metric in r}
    by_trial_b = {int(r["trial"]): float(r[metric]) for r in rows_b if "trial" in r and metric in r}
    common = sorted(set(by_trial_a).intersection(by_trial_b))
    if not common:
        return np.array([], dtype=float), np.array([], dtype=float)
    a = np.array([by_trial_a[t] for t in common], dtype=float)
    b = np.array([by_trial_b[t] for t in common], dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    return a[finite], b[finite]


def collect_wilcoxon_rows(
    suite_name: str,
    run_dir: Path,
    metric_names: Iterable[str],
) -> list[dict[str, float | str]]:
    rows = collect_condition_rows(run_dir)
    out: list[dict[str, float | str]] = []

    scenarios = sorted({scenario for scenario, _ in rows.keys()})
    for scenario in scenarios:
        baseline_key = (scenario, "baseline")
        if baseline_key not in rows:
            continue
        baseline_rows = rows[baseline_key]
        for (_, condition), cond_rows in sorted(rows.items()):
            if _ != scenario or condition == "baseline":
                continue
            if ("noise" not in condition) and (condition not in {"adaptive_noise", "adaptive_both"}):
                continue
            for metric in metric_names:
                x, y = paired_metric_arrays(cond_rows, baseline_rows, metric)
                if x.size < 2:
                    continue
                stat, p_value, n, rbc, method = wilcoxon_signed_rank(x, y)
                out.append(
                    {
                        "suite": suite_name,
                        "run_dir": str(run_dir),
                        "scenario": scenario,
                        "comparison": f"{condition}_vs_baseline",
                        "condition_a": condition,
                        "condition_b": "baseline",
                        "metric": metric,
                        "n_pairs": float(n),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p_value),
                        "rank_biserial_corr": float(rbc),
                        "mean_diff_a_minus_b": float(np.mean(x - y)),
                        "median_diff_a_minus_b": float(np.median(x - y)),
                        "method": method,
                    }
                )

        smoothed_key = (scenario, "noise_alpha_jerk_smoothed")
        raw_key = (scenario, "noise_alpha_jerk_raw")
        if smoothed_key in rows and raw_key in rows:
            for metric in metric_names:
                x, y = paired_metric_arrays(rows[smoothed_key], rows[raw_key], metric)
                if x.size < 2:
                    continue
                stat, p_value, n, rbc, method = wilcoxon_signed_rank(x, y)
                out.append(
                    {
                        "suite": suite_name,
                        "run_dir": str(run_dir),
                        "scenario": scenario,
                        "comparison": "noise_alpha_jerk_smoothed_vs_noise_alpha_jerk_raw",
                        "condition_a": "noise_alpha_jerk_smoothed",
                        "condition_b": "noise_alpha_jerk_raw",
                        "metric": metric,
                        "n_pairs": float(n),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p_value),
                        "rank_biserial_corr": float(rbc),
                        "mean_diff_a_minus_b": float(np.mean(x - y)),
                        "median_diff_a_minus_b": float(np.median(x - y)),
                        "method": method,
                    }
                )
    return out


def holm_bonferroni(rows: list[dict[str, float | str]], alpha: float = 0.05) -> None:
    if not rows:
        return
    indexed = sorted(
        [(idx, float(row["p_value"])) for idx, row in enumerate(rows)],
        key=lambda t: t[1],
    )
    m = len(indexed)
    adj_sorted: list[float] = []
    for rank, (_, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        adj_sorted.append(adj)
    for i in range(1, len(adj_sorted)):
        if adj_sorted[i] < adj_sorted[i - 1]:
            adj_sorted[i] = adj_sorted[i - 1]
    for (idx, _), adj in zip(indexed, adj_sorted):
        rows[idx]["p_value_holm"] = float(adj)
        rows[idx]["significant_0p05_holm"] = "true" if adj < alpha else "false"


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_root = (repo_root / args.output_root / f"adaptive_noise_pipeline_{run_stamp}").resolve()
    pipeline_root.mkdir(parents=True, exist_ok=True)

    def build_shared_args(*, trials: int, steps: int) -> list[str]:
        out = [
            "--trials-per-condition",
            str(int(trials)),
            "--trial-batch",
            str(int(args.trial_batch)),
            "--steps",
            str(int(steps)),
            "--particles",
            str(int(args.particles)),
            "--seed",
            str(int(args.seed)),
            "--room-size",
            str(float(args.room_size)),
            "--dt",
            str(float(args.dt)),
            "--ray-beams",
            str(int(args.ray_beams)),
            "--weight-beam-stride",
            str(int(args.weight_beam_stride)),
            "--safety-margin",
            str(float(args.safety_margin)),
            "--init-radius",
            str(float(args.init_radius)),
            "--collision-probability",
            str(float(args.collision_probability)),
            "--collision-speed-loss-min",
            str(float(args.collision_speed_loss_min)),
            "--collision-speed-loss-max",
            str(float(args.collision_speed_loss_max)),
            "--collision-ang-kick-max",
            str(float(args.collision_ang_kick_max)),
            "--collision-backstep-max",
            str(float(args.collision_backstep_max)),
            "--collision-peak-window",
            str(int(args.collision_peak_window)),
            "--collision-recovery-window",
            str(int(args.collision_recovery_window)),
            "--collision-recovery-pre-window",
            str(int(args.collision_recovery_pre_window)),
            "--collision-recovery-hold",
            str(int(args.collision_recovery_hold)),
            "--collision-recovery-threshold-abs",
            str(float(args.collision_recovery_threshold_abs)),
            "--collision-recovery-threshold-factor",
            str(float(args.collision_recovery_threshold_factor)),
            "--adaptive-noise-alpha",
            str(float(args.adaptive_noise_alpha)),
            "--adaptive-noise-beta",
            str(float(args.adaptive_noise_beta)),
            "--adaptive-noise-damping",
            str(float(args.adaptive_noise_damping)),
        ]
        if args.no_rich:
            out.append("--no-rich")
        if args.adaptive_noise_smoothing:
            out.append("--adaptive-noise-smoothing")
        if not args.independent_condition_seeds:
            out.append("--paired-condition-seeds")
        return out

    full_shared_args = build_shared_args(
        trials=int(args.trials_per_condition),
        steps=int(args.steps),
    )
    stage1_shared_args = build_shared_args(
        trials=max(1, int(args.sensitivity_stage1_trials)),
        steps=max(1, int(args.sensitivity_stage1_steps)),
    )

    suite_results_all: list[SuiteResult] = []
    suite_results_for_stats: list[SuiteResult] = []

    regular_result = run_suite(
        "regular",
        repo_root,
        pipeline_root / "regular",
        full_shared_args,
        ["--study-mode", args.regular_study_mode],
    )
    suite_results_all.append(regular_result)
    suite_results_for_stats.append(regular_result)

    ablation_mode_args = [
        "--study-mode",
        "adaptive_noise_ablation",
        "--ablation-regime",
        args.ablation_regime,
    ]
    if args.ablation_include_smoothing_toggle:
        ablation_mode_args.append("--ablation-include-smoothing-toggle")
    ablation_result = run_suite(
        "ablation",
        repo_root,
        pipeline_root / "ablation",
        full_shared_args,
        ablation_mode_args,
    )
    suite_results_all.append(ablation_result)
    suite_results_for_stats.append(ablation_result)

    if args.single_stage_sensitivity:
        sensitivity_mode_args = [
            "--study-mode",
            "adaptive_noise_sensitivity",
            "--sensitivity-regime",
            args.sensitivity_regime,
            "--sensitivity-alpha-grid",
            args.sensitivity_alpha_grid,
            "--sensitivity-beta-grid",
            args.sensitivity_beta_grid,
            "--sensitivity-gamma-grid",
            args.sensitivity_gamma_grid,
        ]
        if args.sensitivity_include_baseline:
            sensitivity_mode_args.append("--sensitivity-include-baseline")
        sensitivity_result = run_suite(
            "sensitivity",
            repo_root,
            pipeline_root / "sensitivity",
            full_shared_args,
            sensitivity_mode_args,
        )
        suite_results_all.append(sensitivity_result)
        suite_results_for_stats.append(sensitivity_result)
    else:
        sensitivity_explore_mode_args = [
            "--study-mode",
            "adaptive_noise_sensitivity",
            "--sensitivity-regime",
            args.sensitivity_stage1_regime,
            "--sensitivity-alpha-grid",
            args.sensitivity_alpha_grid,
            "--sensitivity-beta-grid",
            args.sensitivity_beta_grid,
            "--sensitivity-gamma-grid",
            args.sensitivity_gamma_grid,
            "--sensitivity-include-baseline",
        ]
        sensitivity_explore_result = run_suite(
            "sensitivity_explore",
            repo_root,
            pipeline_root / "sensitivity_explore",
            stage1_shared_args,
            sensitivity_explore_mode_args,
        )
        suite_results_all.append(sensitivity_explore_result)

        top_configs, top_rows = select_top_sensitivity_configs(
            sensitivity_explore_result.run_dir,
            stage1_regime=args.sensitivity_stage1_regime,
            ranking_metric=args.sensitivity_stage1_ranking_metric,
            top_k=max(1, int(args.sensitivity_top_k)),
        )
        top_config_rows = [
            {"rank": str(i + 1), "alpha": f"{cfg[0]:.10g}", "beta": f"{cfg[1]:.10g}", "gamma": f"{cfg[2]:.10g}"}
            for i, cfg in enumerate(top_configs)
        ]
        write_csv(pipeline_root / "sensitivity_topk_configs.csv", top_config_rows)
        write_csv(pipeline_root / "sensitivity_stage1_ranked_rows.csv", top_rows)

        config_text = ",".join(_format_config_triplet(*cfg) for cfg in top_configs)
        sensitivity_confirm_mode_args = [
            "--study-mode",
            "adaptive_noise_sensitivity",
            "--sensitivity-regime",
            args.sensitivity_regime,
            "--sensitivity-configs",
            config_text,
            "--sensitivity-include-baseline",
        ]
        sensitivity_confirm_result = run_suite(
            "sensitivity",
            repo_root,
            pipeline_root / "sensitivity",
            full_shared_args,
            sensitivity_confirm_mode_args,
        )
        suite_results_all.append(sensitivity_confirm_result)
        suite_results_for_stats.append(sensitivity_confirm_result)

    metric_names = [
        "mean_rmse",
        "final_step_rmse",
        "collision_peak_rmse_mean",
        "collision_recovery_time_mean_steps",
    ]
    wilcoxon_rows: list[dict[str, float | str]] = []
    for suite in suite_results_for_stats:
        wilcoxon_rows.extend(collect_wilcoxon_rows(suite.suite_name, suite.run_dir, metric_names))
    holm_bonferroni(wilcoxon_rows, alpha=0.05)

    write_csv(pipeline_root / "wilcoxon_results.csv", wilcoxon_rows)

    manifest_rows = [
        {
            "suite": suite.suite_name,
            "run_dir": str(suite.run_dir),
            "command": " ".join(suite.command),
        }
        for suite in suite_results_all
    ]
    write_csv(pipeline_root / "run_manifest.csv", manifest_rows)
    print(f"[done] pipeline_root={pipeline_root}")
    print(f"[done] wilcoxon_results={pipeline_root / 'wilcoxon_results.csv'}")
    print(f"[done] run_manifest={pipeline_root / 'run_manifest.csv'}")


if __name__ == "__main__":
    main()
