#!/usr/bin/env python3
"""Combine per-condition JAX metrics.npz files into one long-form RMSE CSV.

Output columns:
  - seed
  - method
  - timestep
  - rmse

Each input condition directory must contain `metrics.npz` with at least:
  - seed: shape (N,)
  - rmse_history: shape (N, T)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine condition metrics.npz files into seed/method/timestep/rmse CSV."
    )
    parser.add_argument(
        "--condition-dir",
        action="append",
        required=True,
        help="Condition directory containing metrics.npz. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to combined output CSV.",
    )
    return parser.parse_args()


def load_condition_rows(condition_dir: Path) -> list[tuple[int, str, int, float]]:
    npz_path = condition_dir / "metrics.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing metrics.npz: {npz_path}")

    method = condition_dir.name
    with np.load(npz_path) as data:
        if "seed" not in data.files:
            raise ValueError(f"{npz_path} is missing 'seed' array.")
        if "rmse_history" not in data.files:
            raise ValueError(
                f"{npz_path} is missing 'rmse_history' array. "
                "This file does not contain per-step RMSE."
            )

        seeds = np.asarray(data["seed"], dtype=int).reshape(-1)
        rmse_history = np.asarray(data["rmse_history"], dtype=float)

    if rmse_history.ndim != 2:
        raise ValueError(
            f"Expected rmse_history to be 2D (N,T), got shape {rmse_history.shape} in {npz_path}"
        )
    if rmse_history.shape[0] != seeds.shape[0]:
        raise ValueError(
            f"Seed count ({seeds.shape[0]}) does not match rmse_history rows ({rmse_history.shape[0]}) "
            f"in {npz_path}"
        )

    rows: list[tuple[int, str, int, float]] = []
    n_trials, n_steps = rmse_history.shape
    for i in range(n_trials):
        seed = int(seeds[i])
        for t in range(n_steps):
            rows.append((seed, method, t, float(rmse_history[i, t])))
    return rows


def main() -> None:
    args = parse_args()
    condition_dirs = [Path(p).resolve() for p in args.condition_dir]
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[int, str, int, float]] = []
    for condition_dir in condition_dirs:
        all_rows.extend(load_condition_rows(condition_dir))

    all_rows.sort(key=lambda r: (r[1], r[0], r[2]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "method", "timestep", "rmse"])
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to: {out_path}")


if __name__ == "__main__":
    main()

