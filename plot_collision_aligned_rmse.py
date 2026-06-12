#!/usr/bin/env python3
"""Generate collision-aligned RMSE plots from trial CSV data.

Expected CSV columns:
  - seed
  - method
  - timestep
  - rmse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean collision-aligned RMSE over time with ±1 std error bands."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input CSV with columns: seed, method, timestep, rmse.",
    )
    parser.add_argument(
        "--output-png",
        default="collision_aligned_rmse.png",
        help="Output PNG path (default: collision_aligned_rmse.png).",
    )
    parser.add_argument(
        "--title",
        default="Collision-Aligned RMSE by Method",
        help="Plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--collision-timestep",
        type=float,
        default=0.0,
        help="Collision timestep in the raw input data (default: 0.0).",
    )
    parser.add_argument(
        "--start-at-collision",
        action="store_true",
        default=True,
        help="Show only post-collision timesteps (default: enabled).",
    )
    parser.add_argument(
        "--no-start-at-collision",
        action="store_false",
        dest="start_at_collision",
        help="Keep pre-collision timesteps in the plot.",
    )
    parser.add_argument(
        "--max-steps-after",
        type=float,
        default=None,
        help="Optional x-axis window after collision in relative timesteps.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Base font size for plot text (default: 14).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required_cols = {"seed", "method", "timestep", "rmse"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.dropna(subset=["method", "timestep", "rmse"])
    collision_t = float(args.collision_timestep)
    # Align to collision so t=0 is always the event.
    df["timestep_rel"] = df["timestep"] - collision_t

    if args.start_at_collision:
        df = df[df["timestep_rel"] >= 0.0].copy()
    if args.max_steps_after is not None:
        df = df[df["timestep_rel"] <= float(args.max_steps_after)].copy()
    if df.empty:
        raise ValueError("No data left after collision-window filtering. Adjust collision/window args.")

    summary = (
        df.groupby(["method", "timestep_rel"], as_index=False)["rmse"]
        .agg(mean_rmse="mean", std_rmse="std")
        .fillna({"std_rmse": 0.0})
    )

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
        palette = sns.color_palette("tab10", n_colors=summary["method"].nunique())
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        palette = plt.cm.tab10.colors

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size,
            "axes.labelsize": args.font_size,
            "legend.fontsize": args.font_size,
            "legend.title_fontsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
        }
    )

    line_styles = ["-", "--", "-.", ":"]
    methods = sorted(summary["method"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, method in enumerate(methods):
        method_df = summary[summary["method"] == method].sort_values("timestep_rel")
        x = method_df["timestep_rel"].to_numpy()
        mean = method_df["mean_rmse"].to_numpy()
        std = method_df["std_rmse"].to_numpy()

        color = palette[idx % len(palette)]
        line_style = line_styles[idx % len(line_styles)]
        label = str(method).replace("_", " ").title()

        ax.plot(
            x,
            mean,
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=2.2,
        )
        ax.fill_between(
            x,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=color,
            alpha=0.22,
            linewidth=0,
        )

    collision_x = 0.0
    ax.axvline(
        collision_x,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=None,
    )
    ymax = float(summary["mean_rmse"].max() + summary["std_rmse"].max())
    ax.text(
        collision_x,
        ymax * 0.98 if ymax > 0 else 0.02,
        "Collision",
        rotation=90,
        va="top",
        ha="right",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Add a small margin so the collision line is clearly visible at boundaries.
    x_min = float(summary["timestep_rel"].min())
    x_max = float(summary["timestep_rel"].max())
    pad = max(1.0, 0.02 * (x_max - x_min + 1.0))
    if args.start_at_collision:
        ax.set_xlim(min(0.0, x_min), x_max + pad)
    else:
        ax.set_xlim(x_min - pad, x_max + pad)

    ax.set_title(args.title)
    ax.set_xlabel("Time relative to collision")
    ax.set_ylabel("RMSE (m)")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=True, title=None)

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
