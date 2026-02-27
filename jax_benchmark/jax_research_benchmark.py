"""JAX benchmark runner for the 2D MCL research suite.

Benchmark-only port (no visualization) intended to mirror the existing
`research_aggressive_benchmark.py` experiment structure and outputs while using
JAX for batched trial execution on GPU/CPU.

Design goals:
- Keep the existing NumPy implementation as the reference path.
- Mirror condition/scenario outputs (`metrics.csv`, `metrics.npz`, `summary.csv`).
- Support adaptive noise, adaptive particles, and collision ablation.
- Use fixed-shape padded particle arrays so adaptive particle count remains JIT-friendly.

Notes:
- JAX RNG and FP32 math will not match NumPy bit-for-bit. Reproducibility is
  preserved within this JAX runner via deterministic seeds.
- This script targets benchmark mode only.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import jax
    import jax.numpy as jnp
    from jax import lax, random
except Exception as exc:  # pragma: no cover - import error reported at runtime
    raise RuntimeError(
        "JAX is required for jax_benchmark/jax_research_benchmark.py. "
        "Install JAX/JAXLIB for your CUDA setup before running this script."
    ) from exc


# Allow importing the existing config/param helpers from repo root when executed
# as `python jax_benchmark/jax_research_benchmark.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import Params, apply_aggression_preset, resolve_base_particles, resolve_max_particles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggressive MCL research benchmarks with JAX")
    parser.add_argument(
        "--study-mode",
        choices=["standard", "collision_ablation"],
        default="standard",
        help="Standard suite or collision ablation (with/without spontaneous collisions).",
    )
    parser.add_argument("--trials-per-condition", type=int, default=20, help="Trials for each condition")
    parser.add_argument("--trial-batch", type=int, default=8, help="JAX lockstep trial batch size")
    parser.add_argument("--steps", type=int, default=10_000, help="Simulation steps per trial")
    parser.add_argument("--particles", type=int, default=2_000, help="Base particle count")
    parser.add_argument("--seed", type=int, default=123, help="Base seed")
    parser.add_argument("--room-size", type=float, default=20.0, help="Square room size in meters")
    parser.add_argument("--start-x", type=float, default=None, help="True start x (default: off-center heuristic)")
    parser.add_argument("--start-y", type=float, default=None, help="True start y (default: off-center heuristic)")
    parser.add_argument("--start-theta", type=float, default=None, help="True start heading in radians")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--ray-beams", type=int, default=9, help="Total ray beams for sensing")
    parser.add_argument("--weight-beam-stride", type=int, default=1, help="Use every Nth beam for particle likelihoods")
    parser.add_argument("--safety-margin", type=float, default=0.5, help="Inset safety margin from walls")
    parser.add_argument("--init-radius", type=float, default=2.0, help="Particle spawn radius around start pose (m)")
    parser.add_argument("--output-dir", type=str, default="research_runs_jax", help="Parent directory for results")
    parser.add_argument("--collision-probability", type=float, default=0.01, help="Per-step spontaneous collision probability (collision ablation)")
    parser.add_argument("--collision-speed-loss-min", type=float, default=0.3)
    parser.add_argument("--collision-speed-loss-max", type=float, default=0.8)
    parser.add_argument("--collision-ang-kick-max", type=float, default=0.8)
    parser.add_argument("--collision-backstep-max", type=float, default=0.20)
    parser.add_argument("--collision-peak-window", type=int, default=50, help="Steps after a collision to search for post-collision peak RMSE")
    parser.add_argument("--collision-recovery-window", type=int, default=200, help="Max steps to search for recovery after collision")
    parser.add_argument("--collision-recovery-pre-window", type=int, default=20, help="Steps before collision used to estimate pre-collision baseline RMSE")
    parser.add_argument("--collision-recovery-hold", type=int, default=5, help="Require threshold to hold for N steps to count as recovered")
    parser.add_argument("--collision-recovery-threshold-abs", type=float, default=0.03, help="Absolute RMSE recovery threshold floor (meters)")
    parser.add_argument("--collision-recovery-threshold-factor", type=float, default=2.0, help="Relative recovery threshold multiplier on pre-collision RMSE")

    parser.add_argument("--adaptive-noise-alpha", type=float, default=0.5)
    parser.add_argument("--adaptive-noise-beta", type=float, default=0.2)
    parser.add_argument("--adaptive-noise-smoothing", action="store_true")
    parser.add_argument("--adaptive-noise-damping", type=float, default=0.0)

    parser.add_argument("--ess-threshold", type=float, default=0.4, help="ESS threshold (ratio if <=1)")
    parser.add_argument("--ess-high-ratio", type=float, default=0.85)
    parser.add_argument("--particle-growth-factor", type=float, default=1.5)
    parser.add_argument("--particle-shrink-step", type=int, default=50)
    parser.add_argument("--max-particles", type=int, default=2_000)

    parser.add_argument("--no-rich", action="store_true", help="Disable Rich terminal output")
    return parser.parse_args()


def make_base_params(args: argparse.Namespace) -> Params:
    params = Params()
    params.mode = "benchmark"
    params.seed = args.seed
    params.num_steps = args.steps
    params.num_particles = args.particles
    params.base_particles = args.particles
    params.max_particles = args.max_particles
    params.dt = args.dt

    room = float(args.room_size)
    params.world = np.array([[0.0, room], [0.0, room]], dtype=float)

    start_x = 0.35 * room if args.start_x is None else float(args.start_x)
    start_y = 0.40 * room if args.start_y is None else float(args.start_y)
    start_theta = np.pi / 6.0 if args.start_theta is None else float(args.start_theta)
    params.x0_true = np.array([start_x, start_y, start_theta], dtype=float)

    params.ray_num_beams = args.ray_beams
    params.weight_beam_stride = max(1, args.weight_beam_stride)
    params.safety_margin = max(0.0, args.safety_margin)

    params.particle_init_mode = "local_radius"
    params.init_radius_m = max(0.05, args.init_radius)

    params.spontaneous_collisions = False
    params.collision_probability = float(np.clip(args.collision_probability, 0.0, 1.0))
    params.collision_speed_loss_range = np.array(
        [
            float(np.clip(args.collision_speed_loss_min, 0.0, 1.0)),
            float(np.clip(args.collision_speed_loss_max, 0.0, 1.0)),
        ],
        dtype=float,
    )
    if params.collision_speed_loss_range[1] < params.collision_speed_loss_range[0]:
        params.collision_speed_loss_range[1] = params.collision_speed_loss_range[0]
    params.collision_ang_vel_kick_max = max(0.0, float(args.collision_ang_kick_max))
    params.collision_backstep_max = max(0.0, float(args.collision_backstep_max))
    # Analysis-only collision response metrics (do not affect simulation).
    params.collision_peak_window = max(1, int(args.collision_peak_window))
    params.collision_recovery_window = max(1, int(args.collision_recovery_window))
    params.collision_recovery_pre_window = max(1, int(args.collision_recovery_pre_window))
    params.collision_recovery_hold = max(1, int(args.collision_recovery_hold))
    params.collision_recovery_threshold_abs = max(0.0, float(args.collision_recovery_threshold_abs))
    params.collision_recovery_threshold_factor = max(1.0, float(args.collision_recovery_threshold_factor))

    apply_aggression_preset(params, "high")

    params.adaptive_noise = False
    params.adaptive_particles = False
    params.adaptive_noise_alpha = args.adaptive_noise_alpha
    params.adaptive_noise_beta = args.adaptive_noise_beta
    params.adaptive_noise_smoothing = bool(args.adaptive_noise_smoothing)
    params.adaptive_noise_damping = float(np.clip(args.adaptive_noise_damping, 0.0, 1.0))
    params.ess_threshold = args.ess_threshold
    params.ess_high_ratio = args.ess_high_ratio
    params.particle_growth_factor = args.particle_growth_factor
    params.particle_shrink_step = args.particle_shrink_step
    return params


def condition_params(base: Params, name: str) -> Params:
    p = Params(**asdict(base))
    p.world = np.array(base.world, dtype=float)
    p.x0_true = np.array(base.x0_true, dtype=float)
    p.motion_noise_std = np.array(base.motion_noise_std, dtype=float)
    p.jerk_std = np.array(base.jerk_std, dtype=float)
    p.accel_limits = np.array(base.accel_limits, dtype=float)
    p.velocity_limits = np.array(base.velocity_limits, dtype=float)
    p.ang_velocity_limits = np.array(base.ang_velocity_limits, dtype=float)
    p.velocity_reversion_gain = np.array(base.velocity_reversion_gain, dtype=float)
    p.collision_speed_loss_range = np.array(base.collision_speed_loss_range, dtype=float)
    p.adaptive_noise = name in {"adaptive_noise", "adaptive_both"}
    p.adaptive_particles = name in {"adaptive_particles", "adaptive_both"}
    # Copy analysis-only dynamic attributes (not part of Params dataclass fields).
    for attr in [
        "collision_peak_window",
        "collision_recovery_window",
        "collision_recovery_pre_window",
        "collision_recovery_hold",
        "collision_recovery_threshold_abs",
        "collision_recovery_threshold_factor",
    ]:
        if hasattr(base, attr):
            setattr(p, attr, getattr(base, attr))
    return p


def write_condition_outputs(
    out_dir: Path,
    params: Params,
    rows: list[dict[str, float | int]],
    histories: dict[str, list[np.ndarray]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        out_dir / "metrics.npz",
        trial=np.array([r["trial"] for r in rows], dtype=int),
        seed=np.array([r["seed"] for r in rows], dtype=int),
        final_step_rmse=np.array([r["final_step_rmse"] for r in rows], dtype=float),
        mean_rmse=np.array([r["mean_rmse"] for r in rows], dtype=float),
        min_rmse=np.array([r["min_rmse"] for r in rows], dtype=float),
        max_rmse=np.array([r["max_rmse"] for r in rows], dtype=float),
        runtime_s=np.array([r["runtime_s"] for r in rows], dtype=float),
        rmse_history=np.stack(histories["rmse_history"], axis=0),
        sigma_t_history=np.stack(histories["sigma_t_history"], axis=0),
        ess_history=np.stack(histories["ess_history"], axis=0),
        particle_count_history=np.stack(histories["particle_count_history"], axis=0),
        accel_history=np.stack(histories["accel_history"], axis=0),
        jerk_history=np.stack(histories["jerk_history"], axis=0),
        accel_mag_history=np.stack(histories["accel_mag_history"], axis=0),
        jerk_mag_history=np.stack(histories["jerk_mag_history"], axis=0),
        collision_history=np.stack(histories["collision_history"], axis=0),
    )

    config_lines = [
        "backend=jax",
        f"jax_devices={[str(d) for d in jax.devices()]}",
        f"num_steps={params.num_steps}",
        f"num_particles={params.num_particles}",
        f"base_particles={resolve_base_particles(params)}",
        f"max_particles={resolve_max_particles(params)}",
        f"dt={params.dt}",
        f"world={params.world.tolist()}",
        f"x0_true={params.x0_true.tolist()}",
        f"aggression={params.aggression}",
        f"particle_init_mode={params.particle_init_mode}",
        f"init_radius_m={params.init_radius_m}",
        f"ray_num_beams={params.ray_num_beams}",
        f"weight_beam_stride={params.weight_beam_stride}",
        f"safety_margin={params.safety_margin}",
        f"spontaneous_collisions={params.spontaneous_collisions}",
        f"collision_probability={params.collision_probability}",
        f"collision_speed_loss_range={params.collision_speed_loss_range.tolist()}",
        f"collision_ang_vel_kick_max={params.collision_ang_vel_kick_max}",
        f"collision_backstep_max={params.collision_backstep_max}",
        f"collision_peak_window={getattr(params, 'collision_peak_window', 50)}",
        f"collision_recovery_window={getattr(params, 'collision_recovery_window', 200)}",
        f"collision_recovery_pre_window={getattr(params, 'collision_recovery_pre_window', 20)}",
        f"collision_recovery_hold={getattr(params, 'collision_recovery_hold', 5)}",
        f"collision_recovery_threshold_abs={getattr(params, 'collision_recovery_threshold_abs', 0.03)}",
        f"collision_recovery_threshold_factor={getattr(params, 'collision_recovery_threshold_factor', 2.0)}",
        f"adaptive_noise={params.adaptive_noise}",
        f"adaptive_noise_alpha={params.adaptive_noise_alpha}",
        f"adaptive_noise_beta={params.adaptive_noise_beta}",
        f"adaptive_noise_smoothing={params.adaptive_noise_smoothing}",
        f"adaptive_noise_damping={params.adaptive_noise_damping}",
        f"adaptive_particles={params.adaptive_particles}",
        f"ess_threshold={params.ess_threshold}",
        f"ess_high_ratio={params.ess_high_ratio}",
        f"particle_growth_factor={params.particle_growth_factor}",
        f"particle_shrink_step={params.particle_shrink_step}",
    ]
    (out_dir / "config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")


def _config_from_params(params: Params) -> dict[str, Any]:
    """Extract immutable JAX-friendly config values from Params."""
    if params.ray_num_beams <= 1:
        beam_angles = np.array([0.0], dtype=np.float32)
    else:
        beam_angles = np.linspace(
            -0.5 * np.deg2rad(params.ray_fov_deg),
            0.5 * np.deg2rad(params.ray_fov_deg),
            params.ray_num_beams,
            dtype=np.float32,
        )
    weight_beam_angles = beam_angles[:: max(1, params.weight_beam_stride)]
    base_n = int(resolve_base_particles(params))
    max_n = int(resolve_max_particles(params)) if params.adaptive_particles else int(params.num_particles)
    max_n = max(max_n, int(params.num_particles))
    return {
        "dt": np.float32(params.dt),
        "num_steps": int(params.num_steps),
        "num_particles": int(params.num_particles),
        "base_particles": int(base_n),
        "max_particles": int(max_n),
        "adaptive_noise": bool(params.adaptive_noise),
        "adaptive_noise_alpha": np.float32(params.adaptive_noise_alpha),
        "adaptive_noise_beta": np.float32(params.adaptive_noise_beta),
        "adaptive_noise_smoothing": bool(params.adaptive_noise_smoothing),
        "adaptive_noise_damping": np.float32(np.clip(params.adaptive_noise_damping, 0.0, 1.0)),
        "adaptive_particles": bool(params.adaptive_particles),
        "ess_threshold": np.float32(params.ess_threshold),
        "ess_high_ratio": np.float32(params.ess_high_ratio),
        "particle_growth_factor": np.float32(params.particle_growth_factor),
        "particle_shrink_step": int(params.particle_shrink_step),
        "eps_weight": np.float32(params.eps_weight),
        "world": np.asarray(params.world, dtype=np.float32),
        "safety_margin": np.float32(params.safety_margin),
        "x0_true": np.asarray(params.x0_true, dtype=np.float32),
        "motion_noise_std": np.asarray(params.motion_noise_std, dtype=np.float32),
        "sensor_noise_std": np.float32(params.sensor_noise_std),
        "ray_max_range": np.float32(params.ray_max_range),
        "beam_angles": beam_angles.astype(np.float32),
        "weight_beam_angles": weight_beam_angles.astype(np.float32),
        "weight_beam_stride": int(max(1, params.weight_beam_stride)),
        "init_radius_m": np.float32(params.init_radius_m),
        "init_pos_std": np.float32(params.init_pos_std),
        "init_theta_std": np.float32(params.init_theta_std),
        "particle_init_mode": str(params.particle_init_mode),
        "v0": np.float32(params.v0),
        "w0": np.float32(params.w0),
        "a_v0": np.float32(params.a_v0),
        "a_w0": np.float32(params.a_w0),
        "jerk_std": np.asarray(params.jerk_std, dtype=np.float32),
        "accel_limits": np.asarray(params.accel_limits, dtype=np.float32),
        "velocity_limits": np.asarray(params.velocity_limits, dtype=np.float32),
        "ang_velocity_limits": np.asarray(params.ang_velocity_limits, dtype=np.float32),
        "jerk_distribution": str(params.jerk_distribution),
        "velocity_reversion_gain": np.asarray(params.velocity_reversion_gain, dtype=np.float32),
        "wall_margin": np.float32(params.wall_margin),
        "wall_turn_gain": np.float32(params.wall_turn_gain),
        "wall_speed_scale": np.float32(params.wall_speed_scale),
        "spontaneous_collisions": bool(params.spontaneous_collisions),
        "collision_probability": np.float32(params.collision_probability),
        "collision_speed_loss_range": np.asarray(params.collision_speed_loss_range, dtype=np.float32),
        "collision_ang_vel_kick_max": np.float32(params.collision_ang_vel_kick_max),
        "collision_backstep_max": np.float32(params.collision_backstep_max),
    }


def _wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _safe_world(world: jnp.ndarray, safety_margin: jnp.ndarray) -> jnp.ndarray:
    x_min, x_max = world[0]
    y_min, y_max = world[1]
    margin = jnp.maximum(jnp.float32(0.0), safety_margin)
    margin_x = jnp.minimum(margin, jnp.float32(0.49) * (x_max - x_min))
    margin_y = jnp.minimum(margin, jnp.float32(0.49) * (y_max - y_min))
    return jnp.array([[x_min + margin_x, x_max - margin_x], [y_min + margin_y, y_max - margin_y]], dtype=world.dtype)


def _raycast_distances(poses: jnp.ndarray, beam_angles: jnp.ndarray, world: jnp.ndarray, max_range: jnp.ndarray) -> jnp.ndarray:
    """JAX raycast to axis-aligned room walls; supports [...,3] poses -> [...,B]."""
    x = poses[..., 0:1]
    y = poses[..., 1:2]
    theta = poses[..., 2:3]
    angles = theta + beam_angles.reshape((1,) * (poses.ndim - 1) + (-1,))
    c = jnp.cos(angles)
    s = jnp.sin(angles)
    eps = jnp.float32(1e-6)
    c_safe = jnp.where(jnp.abs(c) < eps, jnp.where(c >= 0, eps, -eps), c)
    s_safe = jnp.where(jnp.abs(s) < eps, jnp.where(s >= 0, eps, -eps), s)

    x_min, x_max = world[0]
    y_min, y_max = world[1]

    tx_min = (x_min - x) / c_safe
    y_at_x_min = y + tx_min * s
    valid_tx_min = (tx_min > 0.0) & (y_at_x_min >= y_min) & (y_at_x_min <= y_max)
    tx_max = (x_max - x) / c_safe
    y_at_x_max = y + tx_max * s
    valid_tx_max = (tx_max > 0.0) & (y_at_x_max >= y_min) & (y_at_x_max <= y_max)

    ty_min = (y_min - y) / s_safe
    x_at_y_min = x + ty_min * c
    valid_ty_min = (ty_min > 0.0) & (x_at_y_min >= x_min) & (x_at_y_min <= x_max)
    ty_max = (y_max - y) / s_safe
    x_at_y_max = x + ty_max * c
    valid_ty_max = (ty_max > 0.0) & (x_at_y_max >= x_min) & (x_at_y_max <= x_max)

    inf = jnp.float32(jnp.inf)
    d_tx_min = jnp.where(valid_tx_min, tx_min, inf)
    d_tx_max = jnp.where(valid_tx_max, tx_max, inf)
    d_ty_min = jnp.where(valid_ty_min, ty_min, inf)
    d_ty_max = jnp.where(valid_ty_max, ty_max, inf)
    d = jnp.minimum(jnp.minimum(d_tx_min, d_tx_max), jnp.minimum(d_ty_min, d_ty_max))
    return jnp.clip(d, 0.0, max_range)


def _estimate_state(particles: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    x = jnp.sum(weights * particles[..., 0], axis=1)
    y = jnp.sum(weights * particles[..., 1], axis=1)
    c = jnp.sum(weights * jnp.cos(particles[..., 2]), axis=1)
    s = jnp.sum(weights * jnp.sin(particles[..., 2]), axis=1)
    theta = jnp.arctan2(s, c)
    return jnp.stack([x, y, theta], axis=1)


def _compute_rmse(true_xy: jnp.ndarray, est_xy: jnp.ndarray) -> jnp.ndarray:
    diff = true_xy - est_xy
    return jnp.sqrt(jnp.mean(diff * diff, axis=1))


def _split_keys(keys: jnp.ndarray, n: int) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    splits = jax.vmap(lambda k: random.split(k, n + 1))(keys)
    new_keys = splits[:, 0, :]
    subkeys = [splits[:, i + 1, :] for i in range(n)]
    return new_keys, subkeys


def _rand_uniform_batched(
    keys: jnp.ndarray,
    shape_tail: tuple[int, ...],
    *,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> jnp.ndarray:
    return jax.vmap(
        lambda k: random.uniform(k, shape_tail, minval=minval, maxval=maxval, dtype=jnp.float32)
    )(keys)


def _rand_normal_batched(keys: jnp.ndarray, shape_tail: tuple[int, ...]) -> jnp.ndarray:
    return jax.vmap(lambda k: random.normal(k, shape_tail, dtype=jnp.float32))(keys)


def _sample_particles_init(keys: jnp.ndarray, cfg: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize padded particles and weights for a batch of trials."""
    B = keys.shape[0]
    P = int(cfg["max_particles"])
    N0 = int(cfg["num_particles"])
    safe_world = _safe_world(jnp.asarray(cfg["world"]), jnp.asarray(cfg["safety_margin"]))
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]
    x0 = jnp.asarray(cfg["x0_true"])
    mode = cfg["particle_init_mode"]

    keys, subs = _split_keys(keys, 4)
    k1, k2, k3, k4 = subs
    if mode == "local_radius":
        r = jnp.asarray(cfg["init_radius_m"]) * jnp.sqrt(_rand_uniform_batched(k1, (P,)))
        phi = _rand_uniform_batched(k2, (P,), minval=0.0, maxval=jnp.float32(2.0 * np.pi))
        x = x0[0] + r * jnp.cos(phi)
        y = x0[1] + r * jnp.sin(phi)
        theta = _wrap_to_pi(x0[2] + _rand_normal_batched(k3, (P,)) * jnp.asarray(cfg["init_theta_std"]))
    elif mode == "global":
        x = _rand_uniform_batched(k1, (P,), minval=x_min, maxval=x_max)
        y = _rand_uniform_batched(k2, (P,), minval=y_min, maxval=y_max)
        theta = _rand_uniform_batched(k3, (P,), minval=-jnp.pi, maxval=jnp.pi)
    else:
        # Local Gaussian fallback for parity with NumPy implementation.
        x = x0[0] + _rand_normal_batched(k1, (P,)) * jnp.asarray(cfg["init_pos_std"])
        y = x0[1] + _rand_normal_batched(k2, (P,)) * jnp.asarray(cfg["init_pos_std"])
        theta = _wrap_to_pi(x0[2] + _rand_normal_batched(k3, (P,)) * jnp.asarray(cfg["init_theta_std"]))

    x = jnp.clip(x, x_min, x_max)
    y = jnp.clip(y, y_min, y_max)
    particles = jnp.stack([x, y, theta], axis=-1)

    idx = jnp.arange(P)[None, :]
    active_count = jnp.full((B,), N0, dtype=jnp.int32)
    active_mask = idx < active_count[:, None]
    particles = jnp.where(active_mask[..., None], particles, jnp.zeros_like(particles))
    weights = jnp.where(active_mask, 1.0 / jnp.float32(N0), 0.0)
    return keys, particles, weights


def _motion_model(keys: jnp.ndarray, particles: jnp.ndarray, control: jnp.ndarray, dt: jnp.ndarray, sigma_t: jnp.ndarray, active_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched particle motion model with per-trial sigma_t and active mask."""
    B, P, _ = particles.shape
    keys, subs = _split_keys(keys, 2)
    kv, kw = subs
    v = control[:, 0:1]
    w = control[:, 1:2]
    sigma_v = sigma_t[:, 0:1]
    sigma_w = sigma_t[:, 1:2]

    v_noisy = v + _rand_normal_batched(kv, (P,)) * sigma_v
    w_noisy = w + _rand_normal_batched(kw, (P,)) * sigma_w
    theta = particles[..., 2]

    x_new = particles[..., 0] + v_noisy * dt * jnp.cos(theta)
    y_new = particles[..., 1] + v_noisy * dt * jnp.sin(theta)
    th_new = _wrap_to_pi(theta + w_noisy * dt)
    updated = jnp.stack([x_new, y_new, th_new], axis=-1)
    updated = jnp.where(active_mask[..., None], updated, particles)
    return keys, updated


def _clamp_particles_to_world(particles: jnp.ndarray, world: jnp.ndarray, safety_margin: jnp.ndarray, active_mask: jnp.ndarray) -> jnp.ndarray:
    safe_world = _safe_world(world, safety_margin)
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]

    x = particles[..., 0]
    y = particles[..., 1]
    th = particles[..., 2]

    hit_x = active_mask & ((x < x_min) | (x > x_max))
    hit_y = active_mask & ((y < y_min) | (y > y_max))
    x = jnp.clip(x, x_min, x_max)
    y = jnp.clip(y, y_min, y_max)
    th = jnp.where(hit_x, _wrap_to_pi(jnp.pi - th), th)
    th = jnp.where(hit_y, _wrap_to_pi(-th), th)

    out = jnp.stack([x, y, th], axis=-1)
    return jnp.where(active_mask[..., None], out, particles)


def _step_control_dynamics_batch(keys: jnp.ndarray, control_state: jnp.ndarray, cfg: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized control dynamics over trials.

    control_state columns: [v, w, a_v, a_w]
    """
    B = control_state.shape[0]
    dt = jnp.asarray(cfg["dt"])
    jerk_std = jnp.asarray(cfg["jerk_std"])
    accel_limits = jnp.asarray(cfg["accel_limits"])
    velocity_limits = jnp.asarray(cfg["velocity_limits"])
    ang_velocity_limits = jnp.asarray(cfg["ang_velocity_limits"])
    nominal_velocity = jnp.array([cfg["v0"], cfg["w0"]], dtype=jnp.float32)
    rev_gain = jnp.asarray(cfg["velocity_reversion_gain"])

    keys, subs = _split_keys(keys, 2)
    k1, k2 = subs
    if cfg["jerk_distribution"] == "uniform":
        u = _rand_uniform_batched(k1, (2,), minval=-1.0, maxval=1.0)
        jerk = u * jerk_std[None, :]
    else:
        jerk = _rand_normal_batched(k2, (2,)) * jerk_std[None, :]

    v, w, a_v, a_w = [control_state[:, i] for i in range(4)]
    a_v = jnp.clip(a_v + jerk[:, 0] * dt, -accel_limits[0], accel_limits[0])
    a_w = jnp.clip(a_w + jerk[:, 1] * dt, -accel_limits[1], accel_limits[1])

    v = v + a_v * dt
    w = w + a_w * dt
    v = v + rev_gain[0] * (nominal_velocity[0] - v) * dt
    w = w + rev_gain[1] * (nominal_velocity[1] - w) * dt
    v = jnp.clip(v, velocity_limits[0], velocity_limits[1])
    w = jnp.clip(w, ang_velocity_limits[0], ang_velocity_limits[1])

    next_state = jnp.stack([v, w, a_v, a_w], axis=1)
    control = jnp.stack([v, w], axis=1)
    accel = jnp.stack([a_v, a_w], axis=1)
    return keys, next_state, control, accel, jerk


def _apply_room_motion_constraints_batch(states: jnp.ndarray, control: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    world = jnp.asarray(cfg["world"])
    safety_margin = jnp.asarray(cfg["safety_margin"])
    safe_world = _safe_world(world, safety_margin)
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]

    x = states[:, 0]
    y = states[:, 1]
    th = states[:, 2]
    u = control

    front_dist = _raycast_distances(states, jnp.array([0.0], dtype=jnp.float32), world, jnp.asarray(cfg["ray_max_range"]))[:, 0]
    wall_margin = jnp.asarray(cfg["wall_margin"])
    slow_zone = jnp.maximum(wall_margin, jnp.float32(0.25))

    v_limits = jnp.asarray(cfg["velocity_limits"])
    w_limits = jnp.asarray(cfg["ang_velocity_limits"])
    target_v = v_limits[0] + (v_limits[1] - v_limits[0]) * (front_dist / slow_zone)
    u_v = jnp.where(front_dist < slow_zone, jnp.minimum(u[:, 0], target_v), u[:, 0])

    near_wall = ((x - x_min) < wall_margin) | ((x_max - x) < wall_margin) | ((y - y_min) < wall_margin) | ((y_max - y) < wall_margin)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    desired_theta = jnp.arctan2(cy - y, cx - x)
    heading_err = _wrap_to_pi(desired_theta - th)
    u_w = u[:, 1] + jnp.asarray(cfg["wall_turn_gain"]) * heading_err
    u_v = jnp.where(near_wall, jnp.minimum(u_v, jnp.asarray(cfg["wall_speed_scale"]) * v_limits[1]), u_v)

    min_active_speed = jnp.minimum(v_limits[1], jnp.maximum(v_limits[0], jnp.float32(0.12)))
    u_v = jnp.clip(u_v, min_active_speed, v_limits[1])
    u_w = jnp.clip(jnp.where(near_wall, u_w, u[:, 1]), w_limits[0], w_limits[1])
    return jnp.stack([u_v, u_w], axis=1)


def _maybe_apply_collisions_batch(
    keys: jnp.ndarray,
    states_prev: jnp.ndarray,
    u_model: jnp.ndarray,
    accel: jnp.ndarray,
    jerk: jnp.ndarray,
    cfg: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Optional spontaneous collision disturbance (batched over trials)."""
    if (not cfg["spontaneous_collisions"]) or float(cfg["collision_probability"]) <= 0.0:
        B = u_model.shape[0]
        zeros_imp = jnp.zeros((B, 2), dtype=jnp.float32)
        collided = jnp.zeros((B,), dtype=jnp.int32)
        return keys, u_model, accel, jerk, zeros_imp, collided

    dt = jnp.asarray(cfg["dt"])
    vel_lim = jnp.asarray(cfg["velocity_limits"])
    ang_lim = jnp.asarray(cfg["ang_velocity_limits"])
    loss_lo = jnp.asarray(cfg["collision_speed_loss_range"])[0]
    loss_hi = jnp.asarray(cfg["collision_speed_loss_range"])[1]
    ang_kick_max = jnp.asarray(cfg["collision_ang_vel_kick_max"])
    backstep_max = jnp.asarray(cfg["collision_backstep_max"])
    p = jnp.asarray(cfg["collision_probability"])

    keys, subs = _split_keys(keys, 4)
    k_event, k_loss, k_dw, k_bs = subs
    event_u = _rand_uniform_batched(k_event, ())
    collided = (event_u < p).astype(jnp.int32)
    collided_f = collided.astype(jnp.float32)

    loss_frac = _rand_uniform_batched(k_loss, (), minval=loss_lo, maxval=loss_hi)
    dv = -u_model[:, 0] * loss_frac * collided_f
    dw = _rand_uniform_batched(k_dw, (), minval=-ang_kick_max, maxval=ang_kick_max) * collided_f

    u_truth_v = jnp.clip(u_model[:, 0] + dv, 0.0, vel_lim[1])
    u_truth_w = jnp.clip(u_model[:, 1] + dw, ang_lim[0], ang_lim[1])
    u_truth = jnp.stack([u_truth_v, u_truth_w], axis=1)

    dt_safe = jnp.maximum(dt, jnp.float32(1e-6))
    accel_eff = accel + jnp.stack([dv / dt_safe, dw / dt_safe], axis=1)
    jerk_eff = jerk + jnp.stack([dv / (dt_safe * dt_safe), dw / (dt_safe * dt_safe)], axis=1)

    backstep = _rand_uniform_batched(k_bs, (), minval=0.0, maxval=jnp.maximum(0.0, backstep_max)) * collided_f
    theta_prev = states_prev[:, 2]
    truth_impulse_xy = -backstep[:, None] * jnp.stack([jnp.cos(theta_prev), jnp.sin(theta_prev)], axis=1)

    return keys, u_truth, accel_eff, jerk_eff, truth_impulse_xy, collided


def _compute_adaptive_motion_noise_batch(accel: jnp.ndarray, jerk: jnp.ndarray, cfg: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    sigma_0 = jnp.asarray(cfg["motion_noise_std"])
    accel_mag = jnp.linalg.norm(accel, axis=1)
    jerk_mag = jnp.linalg.norm(jerk, axis=1)
    if not cfg["adaptive_noise"]:
        sigma_t = jnp.tile(sigma_0[None, :], (accel.shape[0], 1))
        return sigma_t, accel_mag, jerk_mag
    scale = 1.0 + jnp.asarray(cfg["adaptive_noise_alpha"]) * jnp.abs(accel_mag) + jnp.asarray(cfg["adaptive_noise_beta"]) * jnp.abs(jerk_mag)
    scale = jnp.maximum(scale, jnp.float32(1e-6))
    sigma_t = sigma_0[None, :] * scale[:, None]
    return sigma_t, accel_mag, jerk_mag


def _smooth_sigma_batch(sigma_prev: jnp.ndarray, sigma_raw: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    if not cfg["adaptive_noise"] or not cfg["adaptive_noise_smoothing"]:
        return sigma_raw
    gamma = jnp.asarray(cfg["adaptive_noise_damping"])
    return gamma * sigma_prev + (1.0 - gamma) * sigma_raw


def _propagate_truth_in_box_batch(states: jnp.ndarray, control: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    dt = jnp.asarray(cfg["dt"])
    x = states[:, 0] + control[:, 0] * dt * jnp.cos(states[:, 2])
    y = states[:, 1] + control[:, 0] * dt * jnp.sin(states[:, 2])
    th = _wrap_to_pi(states[:, 2] + control[:, 1] * dt)

    safe_world = _safe_world(jnp.asarray(cfg["world"]), jnp.asarray(cfg["safety_margin"]))
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]
    hit_x = (x < x_min) | (x > x_max)
    hit_y = (y < y_min) | (y > y_max)
    x = jnp.clip(x, x_min, x_max)
    y = jnp.clip(y, y_min, y_max)
    th = jnp.where(hit_x, jnp.pi - th, th)
    th = jnp.where(hit_y, -th, th)
    return jnp.stack([x, y, _wrap_to_pi(th)], axis=1)


def _apply_truth_impulse_in_box_batch(states: jnp.ndarray, impulse_xy: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    s = states.at[:, 0:2].add(impulse_xy)
    safe_world = _safe_world(jnp.asarray(cfg["world"]), jnp.asarray(cfg["safety_margin"]))
    s = s.at[:, 0].set(jnp.clip(s[:, 0], safe_world[0, 0], safe_world[0, 1]))
    s = s.at[:, 1].set(jnp.clip(s[:, 1], safe_world[1, 0], safe_world[1, 1]))
    return s


def _simulate_measurement_batch(keys: jnp.ndarray, true_states: jnp.ndarray, cfg: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
    beam_angles = jnp.asarray(cfg["beam_angles"])
    world = jnp.asarray(cfg["world"])
    max_range = jnp.asarray(cfg["ray_max_range"])
    keys, subs = _split_keys(keys, 1)
    kn = subs[0]
    true_ranges = _raycast_distances(true_states, beam_angles, world, max_range)
    noisy = true_ranges + _rand_normal_batched(kn, (true_ranges.shape[1],)) * jnp.asarray(cfg["sensor_noise_std"])
    return keys, jnp.clip(noisy, 0.0, max_range)


def _ray_sensor_log_weights_batch(particles: jnp.ndarray, measured_ranges: jnp.ndarray, active_mask: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    """Batched particle log-likelihoods. particles: [B,P,3], measured_ranges: [B,M]."""
    beam_angles = jnp.asarray(cfg["weight_beam_angles"])
    predicted = _raycast_distances(particles, beam_angles, jnp.asarray(cfg["world"]), jnp.asarray(cfg["ray_max_range"]))
    residuals = measured_ranges[:, None, :] - predicted
    var = jnp.asarray(cfg["sensor_noise_std"]) ** 2
    log_w = -0.5 * jnp.sum((residuals * residuals) / var, axis=-1)
    # Exclude inactive particles from normalization.
    neg_inf = jnp.float32(-1e30)
    return jnp.where(active_mask, log_w, neg_inf)


def _normalize_weights_and_ess(log_w: jnp.ndarray, active_count: jnp.ndarray, cfg: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
    P = log_w.shape[1]
    idx = jnp.arange(P)[None, :]
    active_mask = idx < active_count[:, None]
    max_log = jnp.max(log_w, axis=1, keepdims=True)
    raw = jnp.exp(log_w - max_log) * active_mask.astype(jnp.float32)
    s = jnp.sum(raw, axis=1, keepdims=True)
    uniform = active_mask.astype(jnp.float32) / jnp.maximum(active_count[:, None].astype(jnp.float32), 1.0)
    weights = jnp.where(s > jnp.asarray(cfg["eps_weight"]), raw / jnp.maximum(s, 1e-30), uniform)
    ess = 1.0 / jnp.maximum(jnp.sum(weights * weights, axis=1), 1e-30)
    return weights, ess


def _resolve_ess_threshold_batch(active_count: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    th = jnp.asarray(cfg["ess_threshold"])
    ratio_case = jnp.maximum(1.0, th * active_count.astype(jnp.float32))
    return jnp.where(th <= 1.0, ratio_case, th)


def _choose_adaptive_particle_count_batch(active_count: jnp.ndarray, ess: jnp.ndarray, cfg: dict[str, Any]) -> jnp.ndarray:
    if not cfg["adaptive_particles"]:
        return active_count
    base_n = jnp.int32(cfg["base_particles"])
    max_n = jnp.int32(cfg["max_particles"])
    ess_low = _resolve_ess_threshold_batch(active_count, cfg)
    ess_high = jnp.maximum(ess_low, jnp.asarray(cfg["ess_high_ratio"]) * active_count.astype(jnp.float32))

    grown = jnp.ceil(active_count.astype(jnp.float32) * jnp.asarray(cfg["particle_growth_factor"])).astype(jnp.int32)
    grown = jnp.minimum(max_n, jnp.maximum(active_count + 1, grown))
    shrunk = jnp.maximum(base_n, active_count - jnp.int32(max(1, int(cfg["particle_shrink_step"]))))

    target = jnp.where(ess < ess_low, grown, active_count)
    target = jnp.where((ess > ess_high) & (active_count > base_n), shrunk, target)
    return target


def _systematic_resample_padded_batch(
    keys: jnp.ndarray,
    particles: jnp.ndarray,
    weights: jnp.ndarray,
    active_count: jnp.ndarray,
    target_count: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Systematic resampling with padded fixed shape and variable active counts."""
    B, P, _ = particles.shape
    keys, subs = _split_keys(keys, 1)
    ku = subs[0]
    target_count = jnp.clip(target_count, 1, P)
    cdf = jnp.cumsum(weights, axis=1)
    cdf = cdf.at[:, -1].set(1.0)

    idx = jnp.arange(P, dtype=jnp.float32)[None, :]
    u0 = _rand_uniform_batched(ku, ()) / target_count.astype(jnp.float32)
    positions = u0[:, None] + idx / target_count[:, None].astype(jnp.float32)
    valid_out = idx < target_count[:, None].astype(jnp.float32)
    positions_safe = jnp.where(valid_out, positions, 1.0)

    def _search(a, v):
        return jnp.searchsorted(a, v, side="left")

    indices = jax.vmap(_search, in_axes=(0, 0))(cdf, positions_safe)
    indices = jnp.clip(indices, 0, P - 1)

    gathered = jnp.take_along_axis(particles, indices[..., None], axis=1)
    gathered = jnp.where(valid_out[..., None], gathered, jnp.zeros_like(gathered))
    new_weights = jnp.where(valid_out, 1.0 / target_count[:, None].astype(jnp.float32), 0.0)
    new_active_count = target_count.astype(jnp.int32)
    return keys, gathered, new_weights, new_active_count


def _build_run_batch_fn(params: Params):
    """Create a jitted batched-trial runner specialized to a condition config."""
    cfg = _config_from_params(params)
    T = int(cfg["num_steps"])
    P = int(cfg["max_particles"])
    N0 = int(cfg["num_particles"])
    safe_world_np = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)  # placeholder for closure type stability
    _ = safe_world_np  # silence unused

    @jax.jit
    def _run_batch(seed_keys: jnp.ndarray) -> dict[str, jnp.ndarray]:
        B = seed_keys.shape[0]
        keys = seed_keys

        keys, particles, weights = _sample_particles_init(keys, cfg)
        active_count = jnp.full((B,), N0, dtype=jnp.int32)

        control_state = jnp.stack(
            [
                jnp.full((B,), cfg["v0"], dtype=jnp.float32),
                jnp.full((B,), cfg["w0"], dtype=jnp.float32),
                jnp.full((B,), cfg["a_v0"], dtype=jnp.float32),
                jnp.full((B,), cfg["a_w0"], dtype=jnp.float32),
            ],
            axis=1,
        )
        x_true0 = jnp.tile(jnp.asarray(cfg["x0_true"])[None, :], (B, 1))
        safe_world = _safe_world(jnp.asarray(cfg["world"]), jnp.asarray(cfg["safety_margin"]))
        x_true0 = x_true0.at[:, 0].set(jnp.clip(x_true0[:, 0], safe_world[0, 0], safe_world[0, 1]))
        x_true0 = x_true0.at[:, 1].set(jnp.clip(x_true0[:, 1], safe_world[1, 0], safe_world[1, 1]))

        x_est0 = _estimate_state(particles, weights)
        x_est0 = x_est0.at[:, 0].set(jnp.clip(x_est0[:, 0], safe_world[0, 0], safe_world[0, 1]))
        x_est0 = x_est0.at[:, 1].set(jnp.clip(x_est0[:, 1], safe_world[1, 0], safe_world[1, 1]))
        rmse0 = _compute_rmse(x_true0[:, :2], x_est0[:, :2])
        sigma_prev0 = jnp.tile(jnp.asarray(cfg["motion_noise_std"])[None, :], (B, 1))

        # Per-step histories (all required by existing research outputs)
        x_true_hist0 = x_true0[:, None, :]
        x_est_hist0 = x_est0[:, None, :]
        rmse_hist0 = rmse0[:, None]
        sigma_hist0 = sigma_prev0[:, None, :]
        ess_hist0 = active_count.astype(jnp.float32)[:, None]
        count_hist0 = active_count[:, None]
        accel_hist0 = control_state[:, 2:4][:, None, :]
        jerk_hist0 = jnp.zeros((B, 1, 2), dtype=jnp.float32)
        accel_mag_hist0 = jnp.zeros((B, 1), dtype=jnp.float32)
        jerk_mag_hist0 = jnp.zeros((B, 1), dtype=jnp.float32)
        collision_hist0 = jnp.zeros((B, 1), dtype=jnp.int32)

        def step_fn(carry, _step_idx):
            keys, particles, weights, active_count, control_state, x_true_prev, sigma_prev = carry

            idx = jnp.arange(P)[None, :]
            active_mask = idx < active_count[:, None]

            keys, (k_dyn,) = _split_keys(keys, 1)
            k_dyn, control_state_next, u_raw, accel, jerk = _step_control_dynamics_batch(k_dyn, control_state, cfg)
            u = _apply_room_motion_constraints_batch(x_true_prev, u_raw, cfg)

            keys, (k_col,) = _split_keys(keys, 1)
            k_col, u_truth, accel_eff, jerk_eff, truth_impulse_xy, collided = _maybe_apply_collisions_batch(
                k_col, x_true_prev, u, accel, jerk, cfg
            )

            sigma_raw, accel_mag, jerk_mag = _compute_adaptive_motion_noise_batch(accel_eff, jerk_eff, cfg)
            sigma_t = _smooth_sigma_batch(sigma_prev, sigma_raw, cfg)

            x_true = _propagate_truth_in_box_batch(x_true_prev, u_truth, cfg)
            x_true = _apply_truth_impulse_in_box_batch(x_true, truth_impulse_xy, cfg)

            keys, z = _simulate_measurement_batch(keys, x_true, cfg)
            z_weight = z[:, :: max(1, int(cfg["weight_beam_stride"]) if "weight_beam_stride" in cfg else 1)]

            keys, particles_pred = _motion_model(keys, particles, u, jnp.asarray(cfg["dt"]), sigma_t, active_mask)
            particles_pred = _clamp_particles_to_world(particles_pred, jnp.asarray(cfg["world"]), jnp.asarray(cfg["safety_margin"]), active_mask)

            log_w = _ray_sensor_log_weights_batch(particles_pred, z_weight, active_mask, cfg)
            weights_norm, ess = _normalize_weights_and_ess(log_w, active_count, cfg)

            x_est = _estimate_state(particles_pred, weights_norm)
            x_est = x_est.at[:, 0].set(jnp.clip(x_est[:, 0], safe_world[0, 0], safe_world[0, 1]))
            x_est = x_est.at[:, 1].set(jnp.clip(x_est[:, 1], safe_world[1, 0], safe_world[1, 1]))
            rmse = _compute_rmse(x_true[:, :2], x_est[:, :2])

            target_count = _choose_adaptive_particle_count_batch(active_count, ess, cfg)
            keys, particles_next, weights_next, active_count_next = _systematic_resample_padded_batch(
                keys, particles_pred, weights_norm, active_count, target_count
            )

            next_carry = (
                keys,
                particles_next,
                weights_next,
                active_count_next,
                control_state_next,
                x_true,
                sigma_t,
            )
            step_logs = {
                "x_true": x_true,
                "x_est": x_est,
                "rmse": rmse,
                "sigma_t": sigma_t,
                "ess": ess,
                "particle_count": active_count_next,
                "accel": accel_eff,
                "jerk": jerk_eff,
                "accel_mag": accel_mag,
                "jerk_mag": jerk_mag,
                "collision": collided,
            }
            return next_carry, step_logs

        init_carry = (keys, particles, weights, active_count, control_state, x_true0, sigma_prev0)
        _, logs = lax.scan(step_fn, init_carry, xs=jnp.arange(1, T, dtype=jnp.int32))

        # logs shapes are [T-1, B, ...]; transpose to [B, T-1, ...] then prepend step0
        x_true_rest = jnp.swapaxes(logs["x_true"], 0, 1)
        x_est_rest = jnp.swapaxes(logs["x_est"], 0, 1)
        rmse_rest = jnp.swapaxes(logs["rmse"], 0, 1)
        sigma_rest = jnp.swapaxes(logs["sigma_t"], 0, 1)
        ess_rest = jnp.swapaxes(logs["ess"], 0, 1)
        count_rest = jnp.swapaxes(logs["particle_count"], 0, 1)
        accel_rest = jnp.swapaxes(logs["accel"], 0, 1)
        jerk_rest = jnp.swapaxes(logs["jerk"], 0, 1)
        accel_mag_rest = jnp.swapaxes(logs["accel_mag"], 0, 1)
        jerk_mag_rest = jnp.swapaxes(logs["jerk_mag"], 0, 1)
        collision_rest = jnp.swapaxes(logs["collision"], 0, 1)

        x_true_hist = jnp.concatenate([x_true_hist0, x_true_rest], axis=1)
        x_est_hist = jnp.concatenate([x_est_hist0, x_est_rest], axis=1)
        rmse_hist = jnp.concatenate([rmse_hist0, rmse_rest], axis=1)
        sigma_hist = jnp.concatenate([sigma_hist0, sigma_rest], axis=1)
        ess_hist = jnp.concatenate([ess_hist0, ess_rest], axis=1)
        count_hist = jnp.concatenate([count_hist0, count_rest], axis=1)
        accel_hist = jnp.concatenate([accel_hist0, accel_rest], axis=1)
        jerk_hist = jnp.concatenate([jerk_hist0, jerk_rest], axis=1)
        accel_mag_hist = jnp.concatenate([accel_mag_hist0, accel_mag_rest], axis=1)
        jerk_mag_hist = jnp.concatenate([jerk_mag_hist0, jerk_mag_rest], axis=1)
        collision_hist = jnp.concatenate([collision_hist0, collision_rest], axis=1)

        return {
            "x_true": x_true_hist,
            "x_est": x_est_hist,
            "rmse_history": rmse_hist,
            "sigma_t_history": sigma_hist,
            "ess_history": ess_hist,
            "particle_count_history": count_hist,
            "accel_history": accel_hist,
            "jerk_history": jerk_hist,
            "accel_mag_history": accel_mag_hist,
            "jerk_mag_history": jerk_mag_hist,
            "collision_history": collision_hist,
        }

    return _run_batch


def _trial_payloads_from_batch_output(
    trial_specs: list[tuple[int, int]],
    batch_results: dict[str, np.ndarray],
    batch_runtime_s: float,
) -> list[dict[str, Any]]:
    B = len(trial_specs)
    per_trial_runtime = batch_runtime_s / max(1, B)
    payloads: list[dict[str, Any]] = []
    rmse_hist = np.asarray(batch_results["rmse_history"], dtype=float)
    for b, (trial_idx, seed) in enumerate(trial_specs):
        payloads.append(
            {
                "trial": int(trial_idx),
                "seed": int(seed),
                "runtime_s": float(per_trial_runtime),
                "results": {
                    "final_step_rmse": float(rmse_hist[b, -1]),
                    "mean_rmse": float(np.mean(rmse_hist[b])),
                    "min_rmse": float(np.min(rmse_hist[b])),
                    "max_rmse": float(np.max(rmse_hist[b])),
                    "rmse_history": np.asarray(batch_results["rmse_history"][b]),
                    "sigma_t_history": np.asarray(batch_results["sigma_t_history"][b]),
                    "ess_history": np.asarray(batch_results["ess_history"][b]),
                    "particle_count_history": np.asarray(batch_results["particle_count_history"][b]),
                    "accel_history": np.asarray(batch_results["accel_history"][b]),
                    "jerk_history": np.asarray(batch_results["jerk_history"][b]),
                    "accel_mag_history": np.asarray(batch_results["accel_mag_history"][b]),
                    "jerk_mag_history": np.asarray(batch_results["jerk_mag_history"][b]),
                    "collision_history": np.asarray(batch_results["collision_history"][b]),
                },
            }
        )
    return payloads


def compute_collision_response_metrics(
    rmse_history: np.ndarray,
    collision_history: np.ndarray,
    *,
    peak_window: int,
    recovery_window: int,
    recovery_pre_window: int,
    recovery_hold: int,
    recovery_threshold_abs: float,
    recovery_threshold_factor: float,
) -> dict[str, float]:
    """Compute event-based post-collision RMSE metrics for one trial.

    Metrics are computed over collision event steps (where collision_history[k] == 1).
    Recovery is the first time the RMSE stays below:
      max(recovery_threshold_abs, recovery_threshold_factor * pre_collision_baseline)
    for `recovery_hold` consecutive steps.
    """
    rmse = np.asarray(rmse_history, dtype=float).reshape(-1)
    coll = np.asarray(collision_history, dtype=int).reshape(-1)
    T = rmse.shape[0]
    if T == 0:
        return {
            "collision_count": 0.0,
            "collision_event_count_analyzed": 0.0,
            "collision_peak_rmse_mean": np.nan,
            "collision_peak_rmse_median": np.nan,
            "collision_peak_delta_mean": np.nan,
            "collision_recovery_time_mean_steps": np.nan,
            "collision_recovery_time_median_steps": np.nan,
            "collision_recovery_success_rate": np.nan,
            "collision_recovery_threshold_mean": np.nan,
        }

    # Rising edges are the event starts; this avoids double-counting if consecutive
    # collisions happen on adjacent steps.
    coll_bool = coll.astype(bool)
    prev = np.concatenate(([False], coll_bool[:-1]))
    event_steps = np.flatnonzero(coll_bool & ~prev)
    if event_steps.size == 0:
        return {
            "collision_count": float(np.sum(coll_bool)),
            "collision_event_count_analyzed": 0.0,
            "collision_peak_rmse_mean": np.nan,
            "collision_peak_rmse_median": np.nan,
            "collision_peak_delta_mean": np.nan,
            "collision_recovery_time_mean_steps": np.nan,
            "collision_recovery_time_median_steps": np.nan,
            "collision_recovery_success_rate": np.nan,
            "collision_recovery_threshold_mean": np.nan,
        }

    peaks: list[float] = []
    peak_deltas: list[float] = []
    recovery_times: list[float] = []
    recovery_thresholds: list[float] = []
    recovered_count = 0

    for k in event_steps:
        pre_lo = max(0, int(k) - int(recovery_pre_window))
        pre_hi = int(k)
        if pre_hi > pre_lo:
            pre_baseline = float(np.mean(rmse[pre_lo:pre_hi]))
        else:
            pre_baseline = float(rmse[int(k)])

        threshold = max(float(recovery_threshold_abs), float(recovery_threshold_factor) * pre_baseline)
        recovery_thresholds.append(threshold)

        peak_hi = min(T, int(k) + int(peak_window) + 1)
        peak_rmse = float(np.max(rmse[int(k):peak_hi]))
        peaks.append(peak_rmse)
        peak_deltas.append(peak_rmse - pre_baseline)

        rec_hi = min(T, int(k) + int(recovery_window) + 1)
        rec_time = np.nan
        for t in range(int(k), rec_hi):
            hold_end = t + int(recovery_hold)
            if hold_end > T:
                break
            if np.all(rmse[t:hold_end] <= threshold):
                rec_time = float(t - int(k))
                recovered_count += 1
                break
        recovery_times.append(rec_time)

    rt = np.asarray(recovery_times, dtype=float)
    rt_valid = rt[np.isfinite(rt)]
    return {
        "collision_count": float(np.sum(coll_bool)),
        "collision_event_count_analyzed": float(event_steps.size),
        "collision_peak_rmse_mean": float(np.mean(peaks)) if peaks else np.nan,
        "collision_peak_rmse_median": float(np.median(peaks)) if peaks else np.nan,
        "collision_peak_delta_mean": float(np.mean(peak_deltas)) if peak_deltas else np.nan,
        "collision_recovery_time_mean_steps": float(np.mean(rt_valid)) if rt_valid.size else np.nan,
        "collision_recovery_time_median_steps": float(np.median(rt_valid)) if rt_valid.size else np.nan,
        "collision_recovery_success_rate": float(recovered_count / event_steps.size) if event_steps.size else np.nan,
        "collision_recovery_threshold_mean": float(np.mean(recovery_thresholds)) if recovery_thresholds else np.nan,
    }


def _nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid))


def run_condition(
    name: str,
    params: Params,
    trials: int,
    base_seed: int,
    out_root: Path,
    console: Console,
    use_rich: bool,
    trial_batch: int,
) -> dict[str, float]:
    cond_dir = out_root / name
    rows: list[dict[str, float | int]] = []
    histories = {
        "rmse_history": [],
        "sigma_t_history": [],
        "ess_history": [],
        "particle_count_history": [],
        "accel_history": [],
        "jerk_history": [],
        "accel_mag_history": [],
        "jerk_mag_history": [],
        "collision_history": [],
    }
    mean_rmse_values: list[float] = []
    final_rmse_values: list[float] = []
    runtimes: list[float] = []
    collision_metric_rows: list[dict[str, float]] = []

    trial_specs = [(trial_idx, base_seed + trial_idx) for trial_idx in range(trials)]
    run_batch_jitted = _build_run_batch_fn(params)

    def process_trial_result(trial_idx: int, trial_seed: int, results: dict[str, np.ndarray | float], runtime_s: float) -> None:
        collision_metrics = compute_collision_response_metrics(
            np.asarray(results["rmse_history"], dtype=float),
            np.asarray(results["collision_history"], dtype=int),
            peak_window=int(getattr(params, "collision_peak_window", 50)),
            recovery_window=int(getattr(params, "collision_recovery_window", 200)),
            recovery_pre_window=int(getattr(params, "collision_recovery_pre_window", 20)),
            recovery_hold=int(getattr(params, "collision_recovery_hold", 5)),
            recovery_threshold_abs=float(getattr(params, "collision_recovery_threshold_abs", 0.03)),
            recovery_threshold_factor=float(getattr(params, "collision_recovery_threshold_factor", 2.0)),
        )
        row = {
            "trial": trial_idx,
            "seed": trial_seed,
            "steps": params.num_steps,
            "particles_start": params.num_particles,
            "particles_end": int(np.asarray(results["particle_count_history"], dtype=int)[-1]),
            "adaptive_noise": int(params.adaptive_noise),
            "adaptive_particles": int(params.adaptive_particles),
            "spontaneous_collisions": int(params.spontaneous_collisions),
            "final_step_rmse": float(results["final_step_rmse"]),
            "mean_rmse": float(results["mean_rmse"]),
            "min_rmse": float(results["min_rmse"]),
            "max_rmse": float(results["max_rmse"]),
            "runtime_s": runtime_s,
        }
        row.update(collision_metrics)
        rows.append(row)
        collision_metric_rows.append(collision_metrics)
        mean_rmse_values.append(row["mean_rmse"])
        final_rmse_values.append(row["final_step_rmse"])
        runtimes.append(runtime_s)
        for key in histories:
            dtype = int if ("count" in key or key == "collision_history") else float
            histories[key].append(np.asarray(results[key], dtype=dtype))

    batch_n = max(1, int(trial_batch))
    for i in range(0, len(trial_specs), batch_n):
        specs = trial_specs[i : i + batch_n]
        keys_np = np.stack([np.array(jax.random.PRNGKey(seed), dtype=np.uint32) for _, seed in specs], axis=0)
        keys = jnp.asarray(keys_np)
        t0 = time.perf_counter()
        batch_results = run_batch_jitted(keys)
        batch_results_np = jax.device_get(batch_results)
        runtime_s = time.perf_counter() - t0
        payloads = _trial_payloads_from_batch_output(specs, batch_results_np, runtime_s)
        for item in payloads:
            process_trial_result(int(item["trial"]), int(item["seed"]), item["results"], float(item["runtime_s"]))

    rows.sort(key=lambda r: int(r["trial"]))
    order = np.argsort(np.array([int(r["trial"]) for r in rows], dtype=int))
    if len(order) > 0:
        for key in histories:
            histories[key] = [histories[key][i] for i in order]

    write_condition_outputs(cond_dir, params, rows, histories)

    summary = {
        "condition": name,
        "trials": float(trials),
        "mean_of_mean_rmse": float(np.mean(mean_rmse_values)),
        "std_of_mean_rmse": float(np.std(mean_rmse_values)),
        "mean_final_rmse": float(np.mean(final_rmse_values)),
        "avg_runtime_s": float(np.mean(runtimes)),
        "total_runtime_s": float(np.sum(runtimes)),
    }
    if collision_metric_rows:
        cm = collision_metric_rows
        summary.update(
            {
                "mean_collision_count": _nanmean_or_nan([m["collision_count"] for m in cm]),
                "mean_collision_events_analyzed": _nanmean_or_nan([m["collision_event_count_analyzed"] for m in cm]),
                "mean_collision_peak_rmse": _nanmean_or_nan([m["collision_peak_rmse_mean"] for m in cm]),
                "mean_collision_peak_delta_rmse": _nanmean_or_nan([m["collision_peak_delta_mean"] for m in cm]),
                "mean_collision_recovery_time_steps": _nanmean_or_nan([m["collision_recovery_time_mean_steps"] for m in cm]),
                "mean_collision_recovery_success_rate": _nanmean_or_nan([m["collision_recovery_success_rate"] for m in cm]),
            }
        )

    if use_rich:
        table = Table(title=f"Condition: {name} (JAX)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Trials", str(trials))
        table.add_row("Trial Batch", str(batch_n))
        table.add_row("Adaptive Noise", str(params.adaptive_noise))
        table.add_row("Adaptive Particles", str(params.adaptive_particles))
        table.add_row("Collisions", str(params.spontaneous_collisions))
        table.add_row("Mean of Mean RMSE", f"{summary['mean_of_mean_rmse']:.4f}")
        table.add_row("Std of Mean RMSE", f"{summary['std_of_mean_rmse']:.4f}")
        table.add_row("Mean Final RMSE", f"{summary['mean_final_rmse']:.4f}")
        table.add_row("Avg Runtime / Trial (s)", f"{summary['avg_runtime_s']:.4f}")
        table.add_row("Total Runtime (s)", f"{summary['total_runtime_s']:.4f}")
        if params.spontaneous_collisions:
            table.add_row("Mean Collision Count", f"{summary.get('mean_collision_count', float('nan')):.2f}")
            table.add_row("Peak RMSE After Collision", f"{summary.get('mean_collision_peak_rmse', float('nan')):.4f}")
            table.add_row("Peak ΔRMSE After Collision", f"{summary.get('mean_collision_peak_delta_rmse', float('nan')):.4f}")
            table.add_row("Recovery Time (steps)", f"{summary.get('mean_collision_recovery_time_steps', float('nan')):.2f}")
            table.add_row("Recovery Success Rate", f"{summary.get('mean_collision_recovery_success_rate', float('nan')):.3f}")
        console.print(table)
    return summary


def main() -> None:
    args = parse_args()
    use_rich = not args.no_rich
    console = Console()

    base = make_base_params(args)
    conditions = ["baseline", "adaptive_noise", "adaptive_particles", "adaptive_both"]
    scenarios = [("standard", False)] if args.study_mode == "standard" else [("no_collision", False), ("with_collision", True)]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"aggressive_suite_{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    if use_rich:
        console.print(
            Panel(
                f"Running JAX MCL research suite\n"
                f"study_mode={args.study_mode}\n"
                f"scenarios={len(scenarios)} conditions={len(conditions)} trials/condition={args.trials_per_condition}\n"
                f"trial_batch={max(1, int(args.trial_batch))}\n"
                f"jax_devices={[str(d) for d in jax.devices()]}\n"
                f"room={args.room_size}m x {args.room_size}m, particles={args.particles}, steps={args.steps}",
                title="Research Benchmark (JAX)",
                border_style="blue",
            )
        )

    summaries: list[dict[str, float]] = []
    for scenario_idx, (scenario_name, collisions_enabled) in enumerate(scenarios):
        scenario_root = out_root if scenario_name == "standard" else (out_root / scenario_name)
        scenario_root.mkdir(parents=True, exist_ok=True)
        for cond_idx, cond_name in enumerate(conditions):
            params = condition_params(base, cond_name)
            params.spontaneous_collisions = bool(collisions_enabled)
            cond_seed = args.seed + scenario_idx * 1_000_000 + cond_idx * 100_000
            summary = run_condition(
                cond_name,
                params,
                trials=args.trials_per_condition,
                base_seed=cond_seed,
                out_root=scenario_root,
                console=console,
                use_rich=use_rich,
                trial_batch=args.trial_batch,
            )
            summary["scenario"] = scenario_name
            summaries.append(summary)

    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    if use_rich:
        final_table = Table(title="Research Suite Summary (JAX)")
        final_table.add_column("Scenario", style="cyan")
        final_table.add_column("Condition", style="cyan")
        final_table.add_column("Mean(Mean RMSE)", style="white")
        final_table.add_column("Mean Final RMSE", style="white")
        final_table.add_column("Total Runtime (s)", style="white")
        for s in summaries:
            final_table.add_row(
                str(s["scenario"]),
                str(s["condition"]),
                f"{s['mean_of_mean_rmse']:.4f}",
                f"{s['mean_final_rmse']:.4f}",
                f"{s['total_runtime_s']:.2f}",
            )
        console.print(final_table)
        console.print(f"Results saved to: {out_root}")
    else:
        print(f"Results saved to: {out_root}")


if __name__ == "__main__":
    main()
