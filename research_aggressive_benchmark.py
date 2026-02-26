"""Research benchmark runner for aggressive MCL trials.

Runs multiple high-aggression trials for four configurations:
1) baseline
2) adaptive_noise
3) adaptive_particles
4) adaptive_both

Default experiment settings (configurable via CLI):
- 20x20 m room
- 2,000 particles (base)
- 10,000 steps
- aggression = high

Outputs are written to a timestamped directory with one subdirectory per
configuration containing CSV/NPZ metrics and per-step logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from compute_rmse import compute_rmse
from dynamics import step_control_dynamics
from main import (
    Params,
    apply_aggression_preset,
    apply_room_motion_constraints,
    apply_truth_impulse_in_box,
    clamp_particles_to_world,
    compute_adaptive_motion_noise,
    get_beam_angles,
    get_safe_world,
    initialize_control_dynamics,
    initialize_particles,
    maybe_apply_spontaneous_collision,
    propagate_truth_in_box,
    resolve_base_particles,
    resolve_max_particles,
    run_mcl,
    simulate_measurement,
    smooth_sigma_t,
)
from motion_model import motion_model
from raycast_sensor import CupyRaycastWeighter, cupy_is_available
from resample_particles import resample_particles
from state_utils import estimate_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggressive MCL research benchmarks across adaptive modes")
    parser.add_argument(
        "--study-mode",
        choices=["standard", "collision_ablation"],
        default="standard",
        help="Standard suite or collision ablation (with/without spontaneous collisions).",
    )
    parser.add_argument("--trials-per-condition", type=int, default=20, help="Trials for each condition")
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
    parser.add_argument(
        "--weight-backend",
        choices=["numpy", "cupy"],
        default="numpy",
        help="Backend for particle likelihood raycasting/weights.",
    )
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA device index for CuPy backend")
    parser.add_argument("--safety-margin", type=float, default=0.5, help="Inset safety margin from walls")
    parser.add_argument("--init-radius", type=float, default=2.0, help="Particle spawn radius around start pose (m)")
    parser.add_argument("--output-dir", type=str, default="research_runs", help="Parent directory for results")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel worker processes for trials (1 = serial, 0 = auto).",
    )
    parser.add_argument(
        "--gpu-trial-batch",
        type=int,
        default=8,
        help="Batch size for lockstep CuPy trial weighting (fixed-particle conditions only, 1 disables).",
    )
    parser.add_argument("--collision-probability", type=float, default=0.01, help="Per-step spontaneous collision probability (collision ablation)")
    parser.add_argument("--collision-speed-loss-min", type=float, default=0.3)
    parser.add_argument("--collision-speed-loss-max", type=float, default=0.8)
    parser.add_argument("--collision-ang-kick-max", type=float, default=0.8)
    parser.add_argument("--collision-backstep-max", type=float, default=0.20)

    # Adaptive noise defaults for the adaptive_noise / adaptive_both conditions
    parser.add_argument("--adaptive-noise-alpha", type=float, default=0.5)
    parser.add_argument("--adaptive-noise-beta", type=float, default=0.2)
    parser.add_argument("--adaptive-noise-smoothing", action="store_true", help="Enable sigma_t EMA smoothing in adaptive-noise conditions")
    parser.add_argument("--adaptive-noise-damping", type=float, default=0.0, help="EMA damping gamma in [0,1] for sigma_t")

    # Adaptive particle defaults for the adaptive_particles / adaptive_both conditions
    parser.add_argument("--ess-threshold", type=float, default=0.4, help="ESS threshold (ratio if <=1)")
    parser.add_argument("--ess-high-ratio", type=float, default=0.85)
    parser.add_argument("--particle-growth-factor", type=float, default=1.5)
    parser.add_argument("--particle-shrink-step", type=int, default=50)
    parser.add_argument("--max-particles", type=int, default=2_000)

    # Runtime tweaks
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich terminal output")
    return parser.parse_args()


def resolve_workers(workers_arg: int, trials: int) -> int:
    """Resolve worker count from CLI (0 => auto)."""
    if workers_arg == 0:
        cpu_n = os.cpu_count() or 1
        return max(1, min(cpu_n, trials))
    return max(1, min(int(workers_arg), trials))


def _run_trial_worker(params: Params, trial_idx: int, trial_seed: int) -> dict:
    """Top-level worker function for process-based parallel trial execution."""
    t0 = time.perf_counter()
    results = run_mcl(params, seed=trial_seed, step_callback=None)
    runtime_s = time.perf_counter() - t0

    # Return only the arrays/metrics needed by the research benchmark outputs.
    payload = {
        "trial": int(trial_idx),
        "seed": int(trial_seed),
        "runtime_s": float(runtime_s),
        "results": {
            "final_step_rmse": float(results["final_step_rmse"]),
            "mean_rmse": float(results["mean_rmse"]),
            "min_rmse": float(results["min_rmse"]),
            "max_rmse": float(results["max_rmse"]),
            "sigma_t_history": np.asarray(results["sigma_t_history"]),
            "ess_history": np.asarray(results["ess_history"]),
            "particle_count_history": np.asarray(results["particle_count_history"]),
            "accel_history": np.asarray(results["accel_history"]),
            "jerk_history": np.asarray(results["jerk_history"]),
            "accel_mag_history": np.asarray(results["accel_mag_history"]),
            "jerk_mag_history": np.asarray(results["jerk_mag_history"]),
            "collision_history": np.asarray(results["collision_history"]),
        },
    }
    return payload


def _run_trial_batch_gpu_weighted_fixed_particles(
    params: Params,
    trial_specs: list[tuple[int, int]],
    gpu_weighter: CupyRaycastWeighter | None = None,
) -> list[dict]:
    """Run a batch of trials in lockstep, batching the CuPy weight update.

    This preserves the existing CPU-side MCL logic (motion, resampling, adaptive noise)
    but batches the particle raycast likelihood call across multiple trials to improve
    GPU occupancy. It assumes a fixed particle count (no adaptive particle resizing).
    """
    if params.weight_backend != "cupy":
        raise ValueError("GPU batched trial runner requires params.weight_backend='cupy'")
    if params.adaptive_particles:
        raise ValueError("GPU batched trial runner currently supports fixed particle count only")

    beam_angles = get_beam_angles(params)
    weight_beam_stride = max(1, params.weight_beam_stride)
    weight_beam_angles = beam_angles[::weight_beam_stride]
    local_weighter = gpu_weighter
    if local_weighter is None:
        local_weighter = CupyRaycastWeighter(
            weight_beam_angles,
            params.world,
            params.sensor_noise_std,
            params.ray_max_range,
            device_id=params.gpu_device_id,
            use_fp32=True,
        )
        local_weighter.warmup()
    t_batch0 = time.perf_counter()

    safe_world = get_safe_world(params.world, params.safety_margin)
    T = params.num_steps
    n_trials = len(trial_specs)

    trials_state: list[dict] = []
    for trial_idx, trial_seed in trial_specs:
        rng = np.random.default_rng(trial_seed)
        x_true = np.zeros((T, 3), dtype=float)
        x_est = np.zeros((T, 3), dtype=float)
        rmse_history = np.zeros(T, dtype=float)
        accel_history = np.zeros((T, 2), dtype=float)
        jerk_history = np.zeros((T, 2), dtype=float)
        sigma_t_history = np.zeros((T, 2), dtype=float)
        ess_history = np.zeros(T, dtype=float)
        particle_count_history = np.zeros(T, dtype=int)
        accel_mag_history = np.zeros(T, dtype=float)
        jerk_mag_history = np.zeros(T, dtype=float)
        collision_history = np.zeros(T, dtype=int)

        x_true[0] = params.x0_true.copy()
        x_true[0, 0] = np.clip(x_true[0, 0], safe_world[0, 0], safe_world[0, 1])
        x_true[0, 1] = np.clip(x_true[0, 1], safe_world[1, 0], safe_world[1, 1])
        particles, weights = initialize_particles(params, rng)
        control_state = initialize_control_dynamics(params)

        accel_history[0] = [control_state.a_v, control_state.a_w]
        x_est[0] = estimate_state(particles, weights)
        x_est[0, 0] = np.clip(x_est[0, 0], safe_world[0, 0], safe_world[0, 1])
        x_est[0, 1] = np.clip(x_est[0, 1], safe_world[1, 0], safe_world[1, 1])
        rmse_history[0] = compute_rmse(x_true[0, :2], x_est[0, :2])
        sigma_t_history[0] = np.asarray(params.motion_noise_std, dtype=float)
        ess_history[0] = float(particles.shape[0])
        particle_count_history[0] = int(particles.shape[0])

        trials_state.append(
            {
                "trial_idx": int(trial_idx),
                "trial_seed": int(trial_seed),
                "rng": rng,
                "x_true": x_true,
                "x_est": x_est,
                "rmse_history": rmse_history,
                "accel_history": accel_history,
                "jerk_history": jerk_history,
                "sigma_t_history": sigma_t_history,
                "ess_history": ess_history,
                "particle_count_history": particle_count_history,
                "accel_mag_history": accel_mag_history,
                "jerk_mag_history": jerk_mag_history,
                "collision_history": collision_history,
                "particles": particles,
                "weights": weights,
                "control_state": control_state,
                "sigma_t_prev": np.asarray(params.motion_noise_std, dtype=float).copy(),
                "z_weight": None,
            }
        )

    for k in range(1, T):
        particles_stack = []
        z_stack = []

        for t in trials_state:
            rng = t["rng"]
            control_state, u, accel, jerk = step_control_dynamics(
                t["control_state"],
                params.dt,
                rng,
                jerk_std=params.jerk_std,
                accel_limits=params.accel_limits,
                velocity_limits=params.velocity_limits,
                ang_velocity_limits=params.ang_velocity_limits,
                jerk_distribution=params.jerk_distribution,
                nominal_velocity=np.array([params.v0, params.w0], dtype=float),
                velocity_reversion_gain=params.velocity_reversion_gain,
            )
            t["control_state"] = control_state
            u = apply_room_motion_constraints(params, t["x_true"][k - 1], u)
            u_truth, accel_eff, jerk_eff, truth_impulse_xy, collided = maybe_apply_spontaneous_collision(
                params,
                t["x_true"][k - 1],
                u,
                accel,
                jerk,
                rng,
            )
            t["collision_history"][k] = int(collided)

            sigma_t_raw, accel_mag, jerk_mag = compute_adaptive_motion_noise(params, accel_eff, jerk_eff)
            sigma_t = smooth_sigma_t(params, t["sigma_t_prev"], sigma_t_raw)
            t["sigma_t_prev"] = sigma_t.copy()
            t["accel_history"][k] = accel_eff
            t["jerk_history"][k] = jerk_eff
            t["sigma_t_history"][k] = sigma_t
            t["accel_mag_history"][k] = accel_mag
            t["jerk_mag_history"][k] = jerk_mag

            t["x_true"][k] = propagate_truth_in_box(params, t["x_true"][k - 1], u_truth)
            t["x_true"][k] = apply_truth_impulse_in_box(params, t["x_true"][k], truth_impulse_xy)
            z = simulate_measurement(params, t["x_true"][k], rng)

            t["particles"] = motion_model(t["particles"], u, params.dt, sigma_t, rng)
            t["particles"] = clamp_particles_to_world(t["particles"], params.world, params.safety_margin)
            t["z_weight"] = z[::weight_beam_stride]
            particles_stack.append(t["particles"])
            z_stack.append(t["z_weight"])

        log_weights_batch = local_weighter.log_likelihood_batch(
            np.stack(particles_stack, axis=0),
            np.stack(z_stack, axis=0),
        )

        for b, t in enumerate(trials_state):
            log_weights = np.asarray(log_weights_batch[b], dtype=float)
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weight_sum = np.sum(weights)
            if weight_sum <= params.eps_weight:
                weights.fill(1.0 / t["particles"].shape[0])
            else:
                weights /= weight_sum
            t["weights"] = weights

            ess = float(1.0 / np.sum(weights**2))
            t["ess_history"][k] = ess

            t["x_est"][k] = estimate_state(t["particles"], weights)
            t["x_est"][k, 0] = np.clip(t["x_est"][k, 0], safe_world[0, 0], safe_world[0, 1])
            t["x_est"][k, 1] = np.clip(t["x_est"][k, 1], safe_world[1, 0], safe_world[1, 1])
            t["rmse_history"][k] = compute_rmse(t["x_true"][k, :2], t["x_est"][k, :2])

            t["particles"], t["weights"] = resample_particles(t["particles"], t["weights"], t["rng"])
            t["particle_count_history"][k] = int(t["particles"].shape[0])

    batch_runtime_s = time.perf_counter() - t_batch0
    runtime_per_trial = batch_runtime_s / max(1, n_trials)

    payloads = []
    for t in trials_state:
        payloads.append(
            {
                "trial": int(t["trial_idx"]),
                "seed": int(t["trial_seed"]),
                "runtime_s": float(runtime_per_trial),
                "results": {
                    "final_step_rmse": float(t["rmse_history"][-1]),
                    "mean_rmse": float(np.mean(t["rmse_history"])),
                    "min_rmse": float(np.min(t["rmse_history"])),
                    "max_rmse": float(np.max(t["rmse_history"])),
                    "sigma_t_history": np.asarray(t["sigma_t_history"]),
                    "ess_history": np.asarray(t["ess_history"]),
                    "particle_count_history": np.asarray(t["particle_count_history"]),
                    "accel_history": np.asarray(t["accel_history"]),
                    "jerk_history": np.asarray(t["jerk_history"]),
                    "accel_mag_history": np.asarray(t["accel_mag_history"]),
                    "jerk_mag_history": np.asarray(t["jerk_mag_history"]),
                    "collision_history": np.asarray(t["collision_history"]),
                },
            }
        )
    return payloads


def make_base_params(args: argparse.Namespace) -> Params:
    params = Params()
    params.mode = "benchmark"
    params.seed = args.seed
    params.num_steps = args.steps
    params.num_particles = args.particles
    params.base_particles = args.particles
    params.max_particles = args.max_particles
    params.dt = args.dt

    # Requested benchmark environment: 20x20 m square (default) and large runs.
    room = float(args.room_size)
    params.world = np.array([[0.0, room], [0.0, room]], dtype=float)

    # Avoid exact center in a square wall-only map; it is a symmetry trap for
    # raycast-only localization and can inflate RMSE over long horizons.
    start_x = 0.35 * room if args.start_x is None else float(args.start_x)
    start_y = 0.40 * room if args.start_y is None else float(args.start_y)
    start_theta = np.pi / 6.0 if args.start_theta is None else float(args.start_theta)
    params.x0_true = np.array([start_x, start_y, start_theta], dtype=float)

    params.ray_num_beams = args.ray_beams
    params.weight_beam_stride = max(1, args.weight_beam_stride)
    params.weight_backend = str(args.weight_backend)
    params.gpu_device_id = max(0, int(args.gpu_device))
    params.safety_margin = max(0.0, args.safety_margin)

    params.particle_init_mode = "local_radius"
    params.init_radius_m = max(0.05, args.init_radius)

    # Collision parameters (disabled by default; enabled in ablation scenarios)
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

    # Force high aggression as requested.
    apply_aggression_preset(params, "high")

    # Disable optional adaptivity here; conditions will toggle them.
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
    # Re-wrap ndarray fields that `asdict` converts to plain lists
    p.world = np.array(base.world, dtype=float)
    p.x0_true = np.array(base.x0_true, dtype=float)
    p.motion_noise_std = np.array(base.motion_noise_std, dtype=float)
    p.jerk_std = np.array(base.jerk_std, dtype=float)
    p.accel_limits = np.array(base.accel_limits, dtype=float)
    p.velocity_limits = np.array(base.velocity_limits, dtype=float)
    p.ang_velocity_limits = np.array(base.ang_velocity_limits, dtype=float)
    p.velocity_reversion_gain = np.array(base.velocity_reversion_gain, dtype=float)

    p.adaptive_noise = name in {"adaptive_noise", "adaptive_both"}
    p.adaptive_particles = name in {"adaptive_particles", "adaptive_both"}
    return p


def write_condition_outputs(
    out_dir: Path,
    params: Params,
    rows: list[dict[str, float | int]],
    histories: dict[str, list[np.ndarray]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-trial summary CSV
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Dense per-step logs for research analysis
    np.savez(
        out_dir / "metrics.npz",
        trial=np.array([r["trial"] for r in rows], dtype=int),
        seed=np.array([r["seed"] for r in rows], dtype=int),
        final_step_rmse=np.array([r["final_step_rmse"] for r in rows], dtype=float),
        mean_rmse=np.array([r["mean_rmse"] for r in rows], dtype=float),
        min_rmse=np.array([r["min_rmse"] for r in rows], dtype=float),
        max_rmse=np.array([r["max_rmse"] for r in rows], dtype=float),
        runtime_s=np.array([r["runtime_s"] for r in rows], dtype=float),
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
        f"weight_backend={params.weight_backend}",
        f"gpu_device_id={params.gpu_device_id}",
        f"safety_margin={params.safety_margin}",
        f"spontaneous_collisions={params.spontaneous_collisions}",
        f"collision_probability={params.collision_probability}",
        f"collision_speed_loss_range={params.collision_speed_loss_range.tolist()}",
        f"collision_ang_vel_kick_max={params.collision_ang_vel_kick_max}",
        f"collision_backstep_max={params.collision_backstep_max}",
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


def run_condition(
    name: str,
    params: Params,
    trials: int,
    base_seed: int,
    out_root: Path,
    console: Console,
    use_rich: bool,
    workers: int,
    gpu_trial_batch: int,
) -> dict[str, float]:
    cond_dir = out_root / name
    rows: list[dict[str, float | int]] = []
    histories = {
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

    # Multiple trial processes each creating a CUDA context on one GPU usually
    # hurts throughput and can OOM. Keep GPU runs single-process by default.
    worker_count = 1 if params.weight_backend == "cupy" else resolve_workers(workers, trials)

    trial_specs = [(trial_idx, base_seed + trial_idx) for trial_idx in range(trials)]

    def process_trial_result(trial_idx: int, trial_seed: int, results: dict[str, np.ndarray | float], runtime_s: float) -> None:
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
        rows.append(row)
        mean_rmse_values.append(row["mean_rmse"])
        final_rmse_values.append(row["final_step_rmse"])
        runtimes.append(runtime_s)

        for key in histories:
            dtype = int if ("count" in key or key == "collision_history") else float
            histories[key].append(np.asarray(results[key], dtype=dtype))

    use_gpu_trial_batch = (
        params.weight_backend == "cupy"
        and worker_count == 1
        and max(1, int(gpu_trial_batch)) > 1
        and not params.adaptive_particles
    )

    if use_gpu_trial_batch:
        batch_n = max(1, int(gpu_trial_batch))
        shared_gpu_weighter = CupyRaycastWeighter(
            get_beam_angles(params)[:: max(1, params.weight_beam_stride)],
            params.world,
            params.sensor_noise_std,
            params.ray_max_range,
            device_id=params.gpu_device_id,
            use_fp32=True,
        )
        shared_gpu_weighter.warmup()
        for i in range(0, len(trial_specs), batch_n):
            payloads = _run_trial_batch_gpu_weighted_fixed_particles(
                params,
                trial_specs[i : i + batch_n],
                gpu_weighter=shared_gpu_weighter,
            )
            payloads.sort(key=lambda item: item["trial"])
            for item in payloads:
                process_trial_result(
                    int(item["trial"]),
                    int(item["seed"]),
                    item["results"],
                    float(item["runtime_s"]),
                )
    elif worker_count == 1:
        for trial_idx, trial_seed in trial_specs:
            t0 = time.perf_counter()
            results = run_mcl(params, seed=trial_seed, step_callback=None)
            runtime_s = time.perf_counter() - t0
            process_trial_result(trial_idx, trial_seed, results, runtime_s)
    else:
        futures = []
        with cf.ProcessPoolExecutor(max_workers=worker_count) as ex:
            for trial_idx, trial_seed in trial_specs:
                futures.append(ex.submit(_run_trial_worker, params, trial_idx, trial_seed))

            # Gather and sort to keep output ordering deterministic.
            completed = [f.result() for f in futures]
            completed.sort(key=lambda item: item["trial"])
            for item in completed:
                process_trial_result(
                    int(item["trial"]),
                    int(item["seed"]),
                    item["results"],
                    float(item["runtime_s"]),
                )

    # Ensure deterministic ordering in saved CSV/NPZ even if parallel execution completed out of order.
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

    if use_rich:
        table = Table(title=f"Condition: {name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Trials", str(trials))
        table.add_row("Mean of Mean RMSE", f"{summary['mean_of_mean_rmse']:.4f}")
        table.add_row("Std of Mean RMSE", f"{summary['std_of_mean_rmse']:.4f}")
        table.add_row("Mean Final RMSE", f"{summary['mean_final_rmse']:.4f}")
        table.add_row("Avg Runtime / Trial (s)", f"{summary['avg_runtime_s']:.4f}")
        table.add_row("Total Runtime (s)", f"{summary['total_runtime_s']:.4f}")
        table.add_row("Workers", str(worker_count))
        table.add_row("Weight Backend", params.weight_backend)
        if params.weight_backend == "cupy":
            table.add_row("GPU Trial Batch", str(max(1, int(gpu_trial_batch))))
            if params.adaptive_particles:
                table.add_row("Batched GPU Weights", "off (adaptive particles fallback)")
            else:
                table.add_row("Batched GPU Weights", str(use_gpu_trial_batch))
        console.print(table)

    return summary


def main() -> None:
    args = parse_args()
    use_rich = not args.no_rich
    console = Console()

    base = make_base_params(args)
    if base.weight_backend == "cupy" and not cupy_is_available():
        raise RuntimeError(
            "CuPy backend requested but CuPy is not installed. "
            "Install a CUDA-matched package (e.g. `uv add cupy-cuda12x`) or use --weight-backend numpy."
        )
    conditions = ["baseline", "adaptive_noise", "adaptive_particles", "adaptive_both"]
    if args.study_mode == "collision_ablation":
        scenarios = [("no_collision", False), ("with_collision", True)]
    else:
        scenarios = [("standard", False)]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"aggressive_suite_{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    panel_workers = 1 if base.weight_backend == "cupy" else resolve_workers(args.workers, args.trials_per_condition)

    if use_rich:
        console.print(
            Panel(
                f"Running aggressive MCL research suite\n"
                f"study_mode={args.study_mode}\n"
                f"scenarios={len(scenarios)} conditions={len(conditions)} trials/condition={args.trials_per_condition}\n"
                f"workers={panel_workers}\n"
                f"weight_backend={base.weight_backend}"
                + (f" gpu_device={base.gpu_device_id}" if base.weight_backend == "cupy" else "")
                + "\n"
                f"room={args.room_size}m x {args.room_size}m, particles={args.particles}, steps={args.steps}",
                title="Research Benchmark",
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
                workers=args.workers,
                gpu_trial_batch=args.gpu_trial_batch,
            )
            summary["scenario"] = scenario_name
            summaries.append(summary)

    # Write top-level summary CSV
    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    if use_rich:
        final_table = Table(title="Research Suite Summary")
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
