"""2D Monte Carlo Localization (MCL) in NumPy with benchmark/visualize modes."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from compute_rmse import compute_rmse
from dynamics import ControlDynamicsState, step_control_dynamics
from motion_model import motion_model, wrap_to_pi
from raycast_sensor import CupyRaycastWeighter, cupy_is_available, ray_sensor_model, raycast_distances
from resample_particles import resample_particles
from state_utils import estimate_state, propagate_state


StepCallback = Callable[[int, np.ndarray, np.ndarray, np.ndarray, float], None]


@dataclass
class Params:
    # Mode selection
    mode: str = "visualize"  # {"visualize", "benchmark"}

    # Reproducibility
    seed: int = 42

    # Simulation settings
    num_steps: int = 200
    dt: float = 0.1
    num_particles: int = 1000

    # Benchmark settings
    num_trials: int = 20
    benchmark_output_dir: str = "benchmark_runs"

    # Visualization settings
    viz_stride: int = 5
    viz_pause_s: float = 0.0
    viz_follow_sim_timing: bool = True
    viz_time_scale: float = 1.5  # >1 is slower than real-time
    viz_show_window: bool = True
    viz_save_snapshot: bool = True
    viz_output_dir: str = "plots_py"
    viz_particle_size: float = 12.0
    viz_particle_alpha: float = 0.70
    viz_draw_rays: bool = True

    # World bounds: [[x_min, x_max], [y_min, y_max]]
    world: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 10.0], [0.0, 10.0]], dtype=float)
    )

    # MCL noise parameters
    motion_noise_std: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.03], dtype=float))
    sensor_noise_std: float = 0.15
    eps_weight: float = 1e-12
    ray_num_beams: int = 9
    ray_fov_deg: float = 180.0
    ray_max_range: float = 15.0
    weight_beam_stride: int = 1  # >1 speeds up particle likelihood updates
    weight_backend: str = "numpy"  # {"numpy", "cupy"} for particle likelihoods
    gpu_device_id: int = 0
    adaptive_noise: bool = False
    adaptive_noise_alpha: float = 0.0
    adaptive_noise_beta: float = 0.0
    adaptive_noise_smoothing: bool = False
    adaptive_noise_damping: float = 0.0  # EMA damping in [0,1], higher = smoother/slower
    adaptive_particles: bool = False
    base_particles: int | None = None
    ess_threshold: float = 0.5   # ratio if <=1, absolute if >1
    ess_high_ratio: float = 0.85
    particle_growth_factor: float = 1.5
    particle_shrink_step: int = 50
    max_particles: int | None = None
    spontaneous_collisions: bool = False
    collision_probability: float = 0.0        # per-step Bernoulli event probability
    collision_speed_loss_range: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.8], dtype=float))
    collision_ang_vel_kick_max: float = 0.8   # rad/s additive one-step truth disturbance
    collision_backstep_max: float = 0.20      # m instantaneous backward displacement on collision

    # Initial state / local particle initialization (range-only baseline)
    x0_true: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, np.pi / 6.0], dtype=float)
    )
    particle_init_mode: str = "local_radius"  # {"local_radius", "global"}
    init_radius_m: float = 2.0
    init_pos_std: float = 0.35
    init_theta_std: float = 0.20

    # Stochastic control dynamics (velocity + acceleration + jerk)
    v0: float = 0.45
    w0: float = 0.08
    a_v0: float = 0.0
    a_w0: float = 0.0
    jerk_std: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.30], dtype=float))
    accel_limits: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.35], dtype=float))
    velocity_limits: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.85], dtype=float))
    ang_velocity_limits: np.ndarray = field(default_factory=lambda: np.array([-0.35, 0.35], dtype=float))
    jerk_distribution: str = "uniform"  # {"uniform", "gaussian"}
    velocity_reversion_gain: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.8], dtype=float))
    wall_margin: float = 0.75
    wall_turn_gain: float = 1.6
    wall_speed_scale: float = 0.6
    safety_margin: float = 0.35
    aggression: str = "medium"  # {"low", "medium", "high"}

    # Terminal UI
    use_rich: bool = True


def initialize_particles(params: Params, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Initialize particles (local radius by default, global optional)."""
    n = params.num_particles
    particles = np.zeros((n, 3), dtype=float)

    safe_world = get_safe_world(params.world, params.safety_margin)
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]

    if params.particle_init_mode == "local":
        particles[:, 0] = params.x0_true[0] + rng.normal(0.0, params.init_pos_std, size=n)
        particles[:, 1] = params.x0_true[1] + rng.normal(0.0, params.init_pos_std, size=n)
        particles[:, 2] = params.x0_true[2] + rng.normal(0.0, params.init_theta_std, size=n)
        particles[:, 0] = np.clip(particles[:, 0], x_min, x_max)
        particles[:, 1] = np.clip(particles[:, 1], y_min, y_max)
        particles[:, 2] = wrap_to_pi(particles[:, 2])
    elif params.particle_init_mode == "local_radius":
        # Uniform sampling in a disk around the starting pose (radius in meters).
        r = params.init_radius_m * np.sqrt(rng.random(n))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        particles[:, 0] = params.x0_true[0] + r * np.cos(phi)
        particles[:, 1] = params.x0_true[1] + r * np.sin(phi)
        particles[:, 2] = wrap_to_pi(params.x0_true[2] + rng.normal(0.0, params.init_theta_std, size=n))
        particles[:, 0] = np.clip(particles[:, 0], x_min, x_max)
        particles[:, 1] = np.clip(particles[:, 1], y_min, y_max)
    else:
        particles[:, 0] = rng.uniform(x_min, x_max, size=n)
        particles[:, 1] = rng.uniform(y_min, y_max, size=n)
        particles[:, 2] = rng.uniform(-np.pi, np.pi, size=n)

    weights = np.full(n, 1.0 / n, dtype=float)
    return particles, weights


def initialize_control_dynamics(params: Params) -> ControlDynamicsState:
    """Initial control process state."""
    return ControlDynamicsState(v=params.v0, w=params.w0, a_v=params.a_v0, a_w=params.a_w0)


def get_beam_angles(params: Params) -> np.ndarray:
    """Beam angles (relative to robot heading) for the raycast sensor."""
    if params.ray_num_beams <= 1:
        return np.array([0.0], dtype=float)
    fov_rad = np.deg2rad(params.ray_fov_deg)
    return np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, params.ray_num_beams, dtype=float)


def resolve_base_particles(params: Params) -> int:
    """Base particle count used as the adaptive-particle target."""
    return int(params.num_particles if params.base_particles is None else params.base_particles)


def resolve_max_particles(params: Params) -> int:
    """Maximum allowed particle count for adaptive resizing."""
    base_n = resolve_base_particles(params)
    return int(base_n * 4 if params.max_particles is None else max(1, params.max_particles))


def resolve_ess_threshold(current_particle_count: int, ess_threshold_cfg: float) -> float:
    """Resolve ESS threshold from config (ratio if <=1, absolute if >1)."""
    if ess_threshold_cfg <= 1.0:
        return max(1.0, ess_threshold_cfg * current_particle_count)
    return float(ess_threshold_cfg)


def compute_adaptive_motion_noise(
    params: Params,
    accel: np.ndarray,
    jerk: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return motion noise sigma_t and the accel/jerk magnitudes used.

    If adaptive noise is disabled, this returns the baseline motion sigma.
    """
    sigma_0 = np.asarray(params.motion_noise_std, dtype=float)
    accel_mag = float(np.linalg.norm(accel))
    jerk_mag = float(np.linalg.norm(jerk))

    if not params.adaptive_noise:
        return sigma_0.copy(), accel_mag, jerk_mag

    scale = 1.0 + params.adaptive_noise_alpha * abs(accel_mag) + params.adaptive_noise_beta * abs(jerk_mag)
    scale = max(scale, 1e-6)
    return sigma_0 * scale, accel_mag, jerk_mag


def smooth_sigma_t(
    params: Params,
    sigma_prev: np.ndarray,
    sigma_raw: np.ndarray,
) -> np.ndarray:
    """Optional EMA smoothing/damping for adaptive motion noise.

    If disabled, returns ``sigma_raw`` unchanged (baseline-compatible behavior).
    """
    if not params.adaptive_noise or not params.adaptive_noise_smoothing:
        return np.asarray(sigma_raw, dtype=float)

    gamma = float(np.clip(params.adaptive_noise_damping, 0.0, 1.0))
    sigma_prev = np.asarray(sigma_prev, dtype=float)
    sigma_raw = np.asarray(sigma_raw, dtype=float)
    return gamma * sigma_prev + (1.0 - gamma) * sigma_raw


def maybe_apply_spontaneous_collision(
    params: Params,
    state_prev: np.ndarray,
    u_model: np.ndarray,
    accel: np.ndarray,
    jerk: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Optional unmodeled collision disturbance applied to truth dynamics.

    Returns
    -------
    u_truth, accel_eff, jerk_eff, truth_impulse_xy, collided
      - `u_truth` is used for ground-truth propagation (particles still use `u_model`)
      - `accel_eff` / `jerk_eff` can include collision impulse terms so adaptive noise
        has a chance to respond in collision-heavy regimes
      - `truth_impulse_xy` is an instantaneous position offset (backstep) applied to truth
    """
    u_truth = np.asarray(u_model, dtype=float).copy()
    accel_eff = np.asarray(accel, dtype=float).copy()
    jerk_eff = np.asarray(jerk, dtype=float).copy()
    truth_impulse_xy = np.zeros(2, dtype=float)

    if (not params.spontaneous_collisions) or (params.collision_probability <= 0.0):
        return u_truth, accel_eff, jerk_eff, truth_impulse_xy, False

    if rng.random() >= params.collision_probability:
        return u_truth, accel_eff, jerk_eff, truth_impulse_xy, False

    # Randomly drop forward speed and inject a one-step angular kick.
    loss_lo, loss_hi = np.asarray(params.collision_speed_loss_range, dtype=float)
    loss_lo = float(np.clip(loss_lo, 0.0, 1.0))
    loss_hi = float(np.clip(loss_hi, loss_lo, 1.0))
    loss_frac = rng.uniform(loss_lo, loss_hi)

    dv = -u_truth[0] * loss_frac
    dw = rng.uniform(-params.collision_ang_vel_kick_max, params.collision_ang_vel_kick_max)

    u_truth[0] = np.clip(u_truth[0] + dv, 0.0, params.velocity_limits[1])
    u_truth[1] = np.clip(u_truth[1] + dw, params.ang_velocity_limits[0], params.ang_velocity_limits[1])

    # Approximate the collision as an impulse in accel/jerk logs to support adaptive noise.
    dt = max(params.dt, 1e-6)
    accel_eff = accel_eff + np.array([dv / dt, dw / dt], dtype=float)
    jerk_eff = jerk_eff + np.array([dv / (dt * dt), dw / (dt * dt)], dtype=float)

    # Small instantaneous backstep along previous heading to emulate contact rebound.
    backstep = rng.uniform(0.0, max(0.0, params.collision_backstep_max))
    theta_prev = float(state_prev[2])
    truth_impulse_xy = -backstep * np.array([np.cos(theta_prev), np.sin(theta_prev)], dtype=float)

    return u_truth, accel_eff, jerk_eff, truth_impulse_xy, True


def choose_adaptive_particle_count(
    params: Params,
    current_particle_count: int,
    ess: float,
) -> int:
    """Choose next particle count from ESS, preserving stability when disabled."""
    if not params.adaptive_particles:
        return int(current_particle_count)

    base_n = max(1, resolve_base_particles(params))
    max_n = max(base_n, resolve_max_particles(params))
    ess_low = resolve_ess_threshold(current_particle_count, params.ess_threshold)
    ess_high = max(ess_low, params.ess_high_ratio * current_particle_count)

    if ess < ess_low:
        grown = int(np.ceil(current_particle_count * params.particle_growth_factor))
        return min(max_n, max(current_particle_count + 1, grown))

    if ess > ess_high and current_particle_count > base_n:
        shrunk = current_particle_count - max(1, params.particle_shrink_step)
        return max(base_n, shrunk)

    return int(current_particle_count)


def apply_aggression_preset(params: Params, preset: str) -> Params:
    """Set motion-dynamics aggressiveness presets (explicit CLI flags can override after)."""
    preset = preset.lower()
    params.aggression = preset

    if preset == "low":
        params.v0 = 0.35
        params.w0 = 0.05
        params.jerk_distribution = "uniform"
        params.jerk_std[:] = [0.18, 0.15]
        params.accel_limits[:] = [0.18, 0.20]
        params.velocity_limits[:] = [0.18, 0.65]
        params.ang_velocity_limits[:] = [-0.22, 0.22]
        params.velocity_reversion_gain[:] = [1.4, 2.1]
    elif preset == "high":
        params.v0 = 0.60
        params.w0 = 0.12
        params.jerk_distribution = "gaussian"
        params.jerk_std[:] = [0.80, 0.90]
        params.accel_limits[:] = [0.60, 0.80]
        params.velocity_limits[:] = [0.20, 1.15]
        params.ang_velocity_limits[:] = [-0.60, 0.60]
        params.velocity_reversion_gain[:] = [0.9, 1.3]
    else:
        # Medium (current defaults)
        params.v0 = 0.45
        params.w0 = 0.08
        params.jerk_distribution = "uniform"
        params.jerk_std[:] = [0.35, 0.30]
        params.accel_limits[:] = [0.30, 0.35]
        params.velocity_limits[:] = [0.20, 0.85]
        params.ang_velocity_limits[:] = [-0.35, 0.35]
        params.velocity_reversion_gain[:] = [1.2, 1.8]
    return params


def get_safe_world(world: np.ndarray, safety_margin: float) -> np.ndarray:
    """Return inset room bounds used as a safety box for robot/particles."""
    world = np.asarray(world, dtype=float)
    x_min, x_max = world[0]
    y_min, y_max = world[1]
    margin = max(0.0, float(safety_margin))

    # Prevent invalid bounds if the user sets a huge margin.
    margin_x = min(margin, 0.49 * (x_max - x_min))
    margin_y = min(margin, 0.49 * (y_max - y_min))
    return np.array(
        [[x_min + margin_x, x_max - margin_x], [y_min + margin_y, y_max - margin_y]],
        dtype=float,
    )


def clamp_particles_to_world(particles: np.ndarray, world: np.ndarray, safety_margin: float) -> np.ndarray:
    """Keep particles inside the safety box and reflect heading on wall hits."""
    safe_world = get_safe_world(world, safety_margin)
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]

    hit_x = (particles[:, 0] < x_min) | (particles[:, 0] > x_max)
    hit_y = (particles[:, 1] < y_min) | (particles[:, 1] > y_max)

    particles[:, 0] = np.clip(particles[:, 0], x_min, x_max)
    particles[:, 1] = np.clip(particles[:, 1], y_min, y_max)

    # Reflect heading for particles that hit a wall. This matches the truth-side
    # boundary handling better than pure clipping, reducing slow reconvergence.
    if np.any(hit_x):
        particles[hit_x, 2] = wrap_to_pi(np.pi - particles[hit_x, 2])
    if np.any(hit_y):
        particles[hit_y, 2] = wrap_to_pi(-particles[hit_y, 2])
    return particles


def apply_room_motion_constraints(params: Params, state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Bias control to stay inside the box and keep motion smooth/active.

    This shapes the stochastic control command before propagation:
    - slows down when facing a nearby wall
    - steers toward the room center when near a wall
    - keeps linear speed above a small floor so it doesn't stall
    """
    u = np.asarray(control, dtype=float).copy()
    x, y, theta = np.asarray(state, dtype=float)
    safe_world = get_safe_world(params.world, params.safety_margin)
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]

    # Forward clearance from current pose using a single beam straight ahead.
    front_dist = float(raycast_distances(np.array([x, y, theta]), np.array([0.0]), params.world, params.ray_max_range)[0])

    # Reduce speed near walls (soft constraint).
    slow_zone = max(params.wall_margin, 0.25)
    if front_dist < slow_zone:
        target_v = params.velocity_limits[0] + (params.velocity_limits[1] - params.velocity_limits[0]) * (front_dist / slow_zone)
        u[0] = min(u[0], target_v)

    # If close to any wall, steer back toward the room center.
    near_wall = (
        (x - x_min) < params.wall_margin
        or (x_max - x) < params.wall_margin
        or (y - y_min) < params.wall_margin
        or (y_max - y) < params.wall_margin
    )
    if near_wall:
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        desired_theta = np.arctan2(cy - y, cx - x)
        heading_err = float(wrap_to_pi(desired_theta - theta))
        u[1] = u[1] + params.wall_turn_gain * heading_err
        u[0] = min(u[0], params.wall_speed_scale * params.velocity_limits[1])

    # Keep movement "alive" (avoid full stop unless bounds force clipping).
    min_active_speed = min(params.velocity_limits[1], max(params.velocity_limits[0], 0.12))
    u[0] = np.clip(u[0], min_active_speed, params.velocity_limits[1])
    u[1] = np.clip(u[1], params.ang_velocity_limits[0], params.ang_velocity_limits[1])
    return u


def propagate_truth_in_box(params: Params, state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Propagate truth state and reflect heading if it would leave the box."""
    next_state = propagate_state(state, control, params.dt)

    safe_world = get_safe_world(params.world, params.safety_margin)
    x_min, x_max = safe_world[0]
    y_min, y_max = safe_world[1]
    x, y, theta = next_state
    hit_x = x < x_min or x > x_max
    hit_y = y < y_min or y > y_max

    x = float(np.clip(x, x_min, x_max))
    y = float(np.clip(y, y_min, y_max))

    # Reflect heading on collision as a safety net (usually avoided by control shaping).
    if hit_x:
        theta = np.pi - theta
    if hit_y:
        theta = -theta

    return np.array([x, y, wrap_to_pi(theta)], dtype=float)


def apply_truth_impulse_in_box(params: Params, state: np.ndarray, impulse_xy: np.ndarray) -> np.ndarray:
    """Apply an instantaneous XY impulse to truth and clamp back into the safety box."""
    if np.allclose(impulse_xy, 0.0):
        return state
    s = np.asarray(state, dtype=float).copy()
    s[0] += float(impulse_xy[0])
    s[1] += float(impulse_xy[1])
    safe_world = get_safe_world(params.world, params.safety_margin)
    s[0] = np.clip(s[0], safe_world[0, 0], safe_world[0, 1])
    s[1] = np.clip(s[1], safe_world[1, 0], safe_world[1, 1])
    return s


def get_visual_pause(params: Params) -> float:
    """Pause duration used by the live visualizer."""
    if params.viz_follow_sim_timing:
        return max(0.0, params.dt * params.viz_stride * params.viz_time_scale)
    return max(0.0, params.viz_pause_s)


def simulate_measurement(
    params: Params,
    true_state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate noisy ray distances to the square room walls."""
    beam_angles = get_beam_angles(params)
    true_ranges = raycast_distances(true_state, beam_angles, params.world, params.ray_max_range)
    noisy = true_ranges + rng.normal(0.0, params.sensor_noise_std, size=true_ranges.shape[0])
    return np.clip(noisy, 0.0, params.ray_max_range)


def run_mcl(
    params: Params,
    *,
    seed: int | None = None,
    step_callback: StepCallback | None = None,
) -> dict[str, np.ndarray | float]:
    """Run a single MCL trial."""
    rng = np.random.default_rng(params.seed if seed is None else seed)

    x_true = np.zeros((params.num_steps, 3), dtype=float)
    x_est = np.zeros((params.num_steps, 3), dtype=float)
    rmse_history = np.zeros(params.num_steps, dtype=float)
    control_history = np.zeros((params.num_steps, 2), dtype=float)  # [v, w]
    accel_history = np.zeros((params.num_steps, 2), dtype=float)    # [a_v, a_w]
    jerk_history = np.zeros((params.num_steps, 2), dtype=float)     # [j_v, j_w]
    sigma_t_history = np.zeros((params.num_steps, 2), dtype=float)  # dynamic motion sigma [v,w]
    ess_history = np.zeros(params.num_steps, dtype=float)
    particle_count_history = np.zeros(params.num_steps, dtype=int)
    accel_mag_history = np.zeros(params.num_steps, dtype=float)
    jerk_mag_history = np.zeros(params.num_steps, dtype=float)
    collision_history = np.zeros(params.num_steps, dtype=int)

    safe_world = get_safe_world(params.world, params.safety_margin)
    x_true[0] = params.x0_true.copy()
    x_true[0, 0] = np.clip(x_true[0, 0], safe_world[0, 0], safe_world[0, 1])
    x_true[0, 1] = np.clip(x_true[0, 1], safe_world[1, 0], safe_world[1, 1])
    particles, weights = initialize_particles(params, rng)
    control_state = initialize_control_dynamics(params)
    beam_angles = get_beam_angles(params)
    weight_beam_stride = max(1, params.weight_beam_stride)
    weight_beam_angles = beam_angles[::weight_beam_stride]
    gpu_weighter = None
    if params.weight_backend == "cupy":
        gpu_weighter = CupyRaycastWeighter(
            weight_beam_angles,
            params.world,
            params.sensor_noise_std,
            params.ray_max_range,
            device_id=params.gpu_device_id,
        )
        # Warm up CuPy once so benchmark timing isn't dominated by first-launch overhead.
        gpu_weighter.warmup()
    control_history[0] = [control_state.v, control_state.w]
    accel_history[0] = [control_state.a_v, control_state.a_w]

    x_est[0] = estimate_state(particles, weights)
    x_est[0, 0] = np.clip(x_est[0, 0], safe_world[0, 0], safe_world[0, 1])
    x_est[0, 1] = np.clip(x_est[0, 1], safe_world[1, 0], safe_world[1, 1])
    rmse_history[0] = compute_rmse(x_true[0, :2], x_est[0, :2])
    sigma_t_history[0] = np.asarray(params.motion_noise_std, dtype=float)
    ess_history[0] = float(particles.shape[0])  # uniform initial weights => ESS = N
    particle_count_history[0] = int(particles.shape[0])
    sigma_t_prev = np.asarray(params.motion_noise_std, dtype=float).copy()
    if step_callback is not None:
        step_callback(0, particles, x_true[0], x_est[0], float(rmse_history[0]))

    for k in range(1, params.num_steps):
        # Simulated control dynamics with bounded random acceleration and jerk
        control_state, u, accel, jerk = step_control_dynamics(
            control_state,
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
        # Shape stochastic control to keep motion realistic in a bounded room.
        u = apply_room_motion_constraints(params, x_true[k - 1], u)

        # Optional unmodeled collision disturbance: truth gets perturbed, particles still
        # propagate with the modeled control `u`. Adaptive noise can react via accel/jerk.
        u_truth, accel_eff, jerk_eff, truth_impulse_xy, collided = maybe_apply_spontaneous_collision(
            params,
            x_true[k - 1],
            u,
            accel,
            jerk,
            rng,
        )
        collision_history[k] = int(collided)

        control_history[k] = u
        accel_history[k] = accel_eff
        jerk_history[k] = jerk_eff
        sigma_t_raw, accel_mag, jerk_mag = compute_adaptive_motion_noise(params, accel_eff, jerk_eff)
        sigma_t = smooth_sigma_t(params, sigma_t_prev, sigma_t_raw)
        sigma_t_prev = sigma_t.copy()
        sigma_t_history[k] = sigma_t
        accel_mag_history[k] = accel_mag
        jerk_mag_history[k] = jerk_mag

        # Ground truth update
        x_true[k] = propagate_truth_in_box(params, x_true[k - 1], u_truth)
        x_true[k] = apply_truth_impulse_in_box(params, x_true[k], truth_impulse_xy)

        # Sensor measurement
        z = simulate_measurement(params, x_true[k], rng)

        # Particle prediction and correction
        particles = motion_model(particles, u, params.dt, sigma_t, rng)
        particles = clamp_particles_to_world(particles, params.world, params.safety_margin)
        z_weight = z[::weight_beam_stride]
        if gpu_weighter is None:
            log_weights = ray_sensor_model(
                particles,
                z_weight,
                weight_beam_angles,
                params.world,
                params.sensor_noise_std,
                params.ray_max_range,
            )
        else:
            log_weights = gpu_weighter.log_likelihood(particles, z_weight)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        weight_sum = np.sum(weights)
        if weight_sum <= params.eps_weight:
            weights.fill(1.0 / particles.shape[0])
        else:
            weights /= weight_sum

        ess = float(1.0 / np.sum(weights**2))
        ess_history[k] = ess

        x_est[k] = estimate_state(particles, weights)
        safe_world = get_safe_world(params.world, params.safety_margin)
        x_est[k, 0] = np.clip(x_est[k, 0], safe_world[0, 0], safe_world[0, 1])
        x_est[k, 1] = np.clip(x_est[k, 1], safe_world[1, 0], safe_world[1, 1])
        rmse_history[k] = compute_rmse(x_true[k, :2], x_est[k, :2])

        # Adaptive particle-count hook (before resampling, using current weights + ESS)
        target_particle_count = choose_adaptive_particle_count(params, particles.shape[0], ess)

        # Resample after computing estimate (optionally changing particle count)
        particles, weights = resample_particles(
            particles,
            weights,
            rng,
            num_output_particles=target_particle_count,
        )
        particle_count_history[k] = int(particles.shape[0])

        if step_callback is not None:
            step_callback(k, particles, x_true[k], x_est[k], float(rmse_history[k]))

    return {
        "x_true": x_true,
        "x_est": x_est,
        "rmse_history": rmse_history,
        "particles_final": particles,
        "control_history": control_history,
        "accel_history": accel_history,
        "jerk_history": jerk_history,
        "sigma_t_history": sigma_t_history,
        "ess_history": ess_history,
        "particle_count_history": particle_count_history,
        "accel_mag_history": accel_mag_history,
        "jerk_mag_history": jerk_mag_history,
        "collision_history": collision_history,
        "final_step_rmse": float(rmse_history[-1]),
        "mean_rmse": float(np.mean(rmse_history)),
        "min_rmse": float(np.min(rmse_history)),
        "max_rmse": float(np.max(rmse_history)),
    }


def configure_matplotlib() -> None:
    """Apply a clean plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "lines.linewidth": 2.0,
        }
    )


class LiveVisualizer:
    """Live matplotlib view for the visualize mode."""

    def __init__(self, params: Params) -> None:
        self.params = params
        self.frame_count = 0
        self.pause_s = get_visual_pause(params)
        self.true_quiver = None
        self.est_quiver = None
        self.true_beam_lines: list = []
        self.est_beam_lines: list = []
        self.beam_angles = get_beam_angles(params)

        configure_matplotlib()

        if params.viz_show_window:
            plt.ion()

        self.fig, (self.ax_map, self.ax_rmse) = plt.subplots(1, 2, figsize=(13, 6))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("MCL Visualize Mode")

        # Static map layers: square room boundary
        x_min, x_max = params.world[0]
        y_min, y_max = params.world[1]
        room_x = [x_min, x_max, x_max, x_min, x_min]
        room_y = [y_min, y_min, y_max, y_max, y_min]
        self.ax_map.plot(room_x, room_y, color="#2ca02c", lw=2.0, label="Room Walls", zorder=2)
        safe_world = get_safe_world(params.world, params.safety_margin)
        sx_min, sx_max = safe_world[0]
        sy_min, sy_max = safe_world[1]
        safe_x = [sx_min, sx_max, sx_max, sx_min, sx_min]
        safe_y = [sy_min, sy_min, sy_max, sy_max, sy_min]
        self.ax_map.plot(safe_x, safe_y, color="#2ca02c", lw=1.2, ls="--", alpha=0.7, label="Safety Margin", zorder=2)
        self.ax_map.set_title("Particle Filter State (Live)")
        self.ax_map.set_xlabel("x")
        self.ax_map.set_ylabel("y")
        self.ax_map.set_xlim(params.world[0, 0], params.world[0, 1])
        self.ax_map.set_ylim(params.world[1, 0], params.world[1, 1])
        self.ax_map.set_aspect("equal", adjustable="box")

        self.particles_scatter = self.ax_map.scatter(
            [], [], s=params.viz_particle_size, c="#444444", alpha=params.viz_particle_alpha, label="Particles"
        )
        (self.true_marker,) = self.ax_map.plot([], [], "o", color="#1f77b4", label="True Pose")
        (self.est_marker,) = self.ax_map.plot([], [], "x", color="#d62728", ms=8, mew=2, label="Est Pose")
        self.true_path_line, = self.ax_map.plot([], [], "-", color="#1f77b4", alpha=0.7, lw=1.5)
        self.est_path_line, = self.ax_map.plot([], [], "--", color="#d62728", alpha=0.7, lw=1.5)
        self.info_text = self.ax_map.text(
            0.02,
            0.98,
            "",
            transform=self.ax_map.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        self.ax_map.legend(loc="upper right")

        # RMSE subplot
        self.ax_rmse.set_title("RMSE (Live)")
        self.ax_rmse.set_xlabel("Step")
        self.ax_rmse.set_ylabel("Position RMSE")
        self.rmse_line, = self.ax_rmse.plot([], [], color="#9467bd")

        self.true_path_x: list[float] = []
        self.true_path_y: list[float] = []
        self.est_path_x: list[float] = []
        self.est_path_y: list[float] = []
        self.rmse_vals: list[float] = []
        self.rmse_steps: list[int] = []

        self.fig.tight_layout()

    def _update_heading_quiver(self, which: str, pose: np.ndarray, color: str) -> None:
        ax = self.ax_map
        x, y, theta = pose
        u = 0.45 * np.cos(theta)
        v = 0.45 * np.sin(theta)

        existing = self.true_quiver if which == "true" else self.est_quiver
        if existing is not None:
            existing.remove()

        q = ax.quiver(
            [x], [y], [u], [v],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.008,
            zorder=4,
        )

        if which == "true":
            self.true_quiver = q
        else:
            self.est_quiver = q

    def _clear_beam_lines(self, which: str) -> None:
        lines = self.true_beam_lines if which == "true" else self.est_beam_lines
        for line in lines:
            line.remove()
        lines.clear()

    def _update_beams(self, which: str, pose: np.ndarray, color: str, linestyle: str, alpha: float) -> None:
        self._clear_beam_lines(which)
        distances = raycast_distances(pose, self.beam_angles, self.params.world, self.params.ray_max_range)
        x0, y0, theta = pose
        beam_world_angles = theta + self.beam_angles
        x1 = x0 + distances * np.cos(beam_world_angles)
        y1 = y0 + distances * np.sin(beam_world_angles)

        lines = self.true_beam_lines if which == "true" else self.est_beam_lines
        for i in range(len(distances)):
            line, = self.ax_map.plot(
                [x0, x1[i]],
                [y0, y1[i]],
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                lw=1.0,
                zorder=1,
            )
            lines.append(line)

    def update(
        self,
        step_idx: int,
        particles: np.ndarray,
        true_pose: np.ndarray,
        est_pose: np.ndarray,
        rmse: float,
        *,
        force: bool = False,
    ) -> None:
        """Update the live view every `viz_stride` steps (or on force)."""
        if (not force) and (step_idx % self.params.viz_stride != 0):
            return

        self.true_path_x.append(float(true_pose[0]))
        self.true_path_y.append(float(true_pose[1]))
        self.est_path_x.append(float(est_pose[0]))
        self.est_path_y.append(float(est_pose[1]))
        self.rmse_vals.append(float(rmse))
        self.rmse_steps.append(int(step_idx))
        self.frame_count += 1

        self.particles_scatter.set_offsets(particles[:, :2])
        self.true_marker.set_data([true_pose[0]], [true_pose[1]])
        self.est_marker.set_data([est_pose[0]], [est_pose[1]])
        self.true_path_line.set_data(self.true_path_x, self.true_path_y)
        self.est_path_line.set_data(self.est_path_x, self.est_path_y)
        self._update_heading_quiver("true", true_pose, "#1f77b4")
        self._update_heading_quiver("est", est_pose, "#d62728")
        if self.params.viz_draw_rays:
            self._update_beams("true", true_pose, "#1f77b4", "-", 0.35)
            self._update_beams("est", est_pose, "#d62728", "--", 0.35)
        else:
            self._clear_beam_lines("true")
            self._clear_beam_lines("est")

        self.rmse_line.set_data(self.rmse_steps, self.rmse_vals)
        self.ax_rmse.relim()
        self.ax_rmse.autoscale_view()

        self.info_text.set_text(
            f"step={step_idx}\nrmse={rmse:.4f}\nN={self.params.num_particles}\n"
            f"stride={self.params.viz_stride}\nbeams={self.params.ray_num_beams}\n"
            f"pause={self.pause_s:.3f}s\nmargin={self.params.safety_margin:.2f}\n"
            f"rays={'on' if self.params.viz_draw_rays else 'off'}"
        )

        self.fig.canvas.draw_idle()
        if self.params.viz_show_window:
            plt.pause(self.pause_s)

    def finalize(self, out_dir: str | None = None) -> None:
        """Save a final snapshot and optionally hold the window open."""
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(out_path / "mcl_visualize_live_snapshot.png", dpi=150, bbox_inches="tight")

        if self.params.viz_show_window:
            plt.ioff()
            plt.show()
        else:
            plt.close(self.fig)


def save_static_plots(params: Params, results: dict[str, np.ndarray | float]) -> None:
    """Save trajectory and RMSE summary plots after a run."""
    out_dir = Path(params.viz_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    x_true = results["x_true"]
    x_est = results["x_est"]
    rmse_history = results["rmse_history"]
    particles_final = results["particles_final"]

    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5))
    ax1.plot(x_true[:, 0], x_true[:, 1], label="Ground Truth", color="#1f77b4")
    ax1.plot(x_est[:, 0], x_est[:, 1], "--", label="MCL Estimate", color="#d62728")
    x_min, x_max = params.world[0]
    y_min, y_max = params.world[1]
    room_x = [x_min, x_max, x_max, x_min, x_min]
    room_y = [y_min, y_min, y_max, y_max, y_min]
    ax1.plot(room_x, room_y, color="#2ca02c", lw=2.0, label="Room Walls")
    ax1.scatter(particles_final[:, 0], particles_final[:, 1], s=10, c="#555555", alpha=0.5, label="Particles")
    ax1.set_title("2D Monte Carlo Localization")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(params.world[0, 0], params.world[0, 1])
    ax1.set_ylim(params.world[1, 0], params.world[1, 1])
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(out_dir / "mcl_trajectory.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.0))
    ax2.plot(np.arange(params.num_steps), rmse_history, color="#9467bd")
    ax2.set_title("RMSE vs Ground Truth")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Position RMSE")
    fig2.tight_layout()
    fig2.savefig(out_dir / "mcl_rmse.png", dpi=150, bbox_inches="tight")

    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    control_history = results["control_history"]
    accel_history = results["accel_history"]
    jerk_history = results["jerk_history"]
    steps = np.arange(params.num_steps)
    ax3.plot(steps, control_history[:, 0], label="v")
    ax3.plot(steps, control_history[:, 1], label="w")
    ax3.set_ylabel("Velocity")
    ax3.set_title("Simulated Control Dynamics")
    ax3.legend(loc="best")
    ax4.plot(steps, accel_history[:, 0], label="a_v")
    ax4.plot(steps, accel_history[:, 1], label="a_w")
    ax4.plot(steps, jerk_history[:, 0], ":", label="j_v", alpha=0.7)
    ax4.plot(steps, jerk_history[:, 1], ":", label="j_w", alpha=0.7)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Accel / Jerk")
    ax4.legend(loc="best")
    fig3.tight_layout()
    fig3.savefig(out_dir / "mcl_control_dynamics.png", dpi=150, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def print_single_run_summary(params: Params, results: dict[str, np.ndarray | float], console: Console) -> None:
    x_true = results["x_true"]
    x_est = results["x_est"]
    final_true = x_true[-1, :2]
    final_est = x_est[-1, :2]
    ess_history = np.asarray(results["ess_history"], dtype=float)
    particle_count_history = np.asarray(results["particle_count_history"], dtype=int)
    sigma_t_history = np.asarray(results["sigma_t_history"], dtype=float)
    collision_history = np.asarray(results["collision_history"], dtype=int)

    if not params.use_rich:
        print(f"Mode: {params.mode}")
        print(f"Final GT: [{final_true[0]:.3f}, {final_true[1]:.3f}]")
        print(f"Final Est: [{final_est[0]:.3f}, {final_est[1]:.3f}]")
        print(f"Final RMSE: {float(results['final_step_rmse']):.4f}")
        print(f"Mean RMSE: {float(results['mean_rmse']):.4f}")
        return

    table = Table(title=f"MCL {params.mode.title()} Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Seed", str(params.seed))
    table.add_row("Steps", str(params.num_steps))
    table.add_row("Particles", str(params.num_particles))
    table.add_row("Init Mode", params.particle_init_mode)
    if params.particle_init_mode == "local_radius":
        table.add_row("Init Radius (m)", f"{params.init_radius_m:.2f}")
    table.add_row("Aggression", params.aggression)
    table.add_row("Sensor Model", "raycast (square room)")
    table.add_row("Weight Backend", params.weight_backend)
    if params.weight_backend == "cupy":
        table.add_row("GPU Device", str(params.gpu_device_id))
    table.add_row("Ray Beams / FOV", f"{params.ray_num_beams} / {params.ray_fov_deg:.0f} deg")
    table.add_row("Weight Beam Stride", str(params.weight_beam_stride))
    table.add_row("Ray Max Range", f"{params.ray_max_range:.2f}")
    table.add_row("Safety Margin", f"{params.safety_margin:.2f}")
    table.add_row("Adaptive Noise", str(params.adaptive_noise))
    if params.adaptive_noise:
        table.add_row("Adaptive Noise α/β", f"{params.adaptive_noise_alpha:.3f} / {params.adaptive_noise_beta:.3f}")
        table.add_row("Noise Smoothing", str(params.adaptive_noise_smoothing))
        if params.adaptive_noise_smoothing:
            table.add_row("Noise Damping γ", f"{params.adaptive_noise_damping:.3f}")
    table.add_row("Adaptive Particles", str(params.adaptive_particles))
    if params.adaptive_particles:
        table.add_row("ESS Threshold", f"{params.ess_threshold}")
        table.add_row("Base / Max Particles", f"{resolve_base_particles(params)} / {resolve_max_particles(params)}")
    table.add_row("Final ESS", f"{ess_history[-1]:.2f}")
    table.add_row("Final Particle Count", str(int(particle_count_history[-1])))
    table.add_row("Final sigma_t [v,w]", f"[{sigma_t_history[-1,0]:.4f}, {sigma_t_history[-1,1]:.4f}]")
    table.add_row("Collisions Enabled", str(params.spontaneous_collisions))
    if params.spontaneous_collisions:
        table.add_row("Collision Prob", f"{params.collision_probability:.4f}")
        table.add_row("Collision Count", str(int(np.sum(collision_history))))
    table.add_row("Final Ground Truth (x,y)", f"[{final_true[0]:.3f}, {final_true[1]:.3f}]")
    table.add_row("Final Estimate (x,y)", f"[{final_est[0]:.3f}, {final_est[1]:.3f}]")
    table.add_row("Final Step RMSE", f"{float(results['final_step_rmse']):.4f}")
    table.add_row("Mean RMSE", f"{float(results['mean_rmse']):.4f}")
    table.add_row("Min / Max RMSE", f"{float(results['min_rmse']):.4f} / {float(results['max_rmse']):.4f}")

    console.print(
        Panel(
            "Control inputs are generated by a stochastic dynamics model:\n"
            "bounded acceleration random walk driven by jerk noise (uniform by default).\n"
            "Raycast sensor mode uses lidar-like distance beams to room walls.",
            title="2D MCL (NumPy)",
            border_style="blue",
        )
    )
    console.print(table)


def run_visualize_mode(params: Params, console: Console) -> None:
    """Run a single trial with a live matplotlib visualization."""
    params.mode = "visualize"
    params.viz_show_window = True if params.viz_show_window else False

    visualizer = LiveVisualizer(params)

    def on_step(k: int, particles: np.ndarray, x_true_k: np.ndarray, x_est_k: np.ndarray, rmse_k: float) -> None:
        force = (k == params.num_steps - 1)
        visualizer.update(k, particles, x_true_k, x_est_k, rmse_k, force=force)

    results = run_mcl(params, step_callback=on_step)
    print_single_run_summary(params, results, console)

    # Save static post-run plots for comparison / reports.
    save_static_plots(params, results)
    if params.viz_save_snapshot:
        visualizer.finalize(params.viz_output_dir)
    else:
        visualizer.finalize(None)


def run_benchmark_mode(params: Params, console: Console) -> None:
    """Run multiple trials without plotting and log metrics to disk."""
    params.mode = "benchmark"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(params.benchmark_output_dir) / f"benchmark_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    mean_rmse_values = []
    final_rmse_values = []
    runtimes = []
    sigma_t_histories = []
    ess_histories = []
    particle_count_histories = []
    accel_histories = []
    jerk_histories = []
    collision_histories = []

    for trial_idx in range(params.num_trials):
        trial_seed = params.seed + trial_idx
        t0 = time.perf_counter()
        results = run_mcl(params, seed=trial_seed, step_callback=None)
        runtime_s = time.perf_counter() - t0

        row = {
            "trial": trial_idx,
            "seed": trial_seed,
            "steps": params.num_steps,
            "particles": params.num_particles,
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
        sigma_t_histories.append(np.asarray(results["sigma_t_history"], dtype=float))
        ess_histories.append(np.asarray(results["ess_history"], dtype=float))
        particle_count_histories.append(np.asarray(results["particle_count_history"], dtype=int))
        accel_histories.append(np.asarray(results["accel_history"], dtype=float))
        jerk_histories.append(np.asarray(results["jerk_history"], dtype=float))
        collision_histories.append(np.asarray(results["collision_history"], dtype=int))

    # Write CSV for easy inspection
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write NPZ for fast reload in Python
    npz_path = out_dir / "metrics.npz"
    np.savez(
        npz_path,
        trial=np.array([r["trial"] for r in rows], dtype=int),
        seed=np.array([r["seed"] for r in rows], dtype=int),
        final_step_rmse=np.array([r["final_step_rmse"] for r in rows], dtype=float),
        mean_rmse=np.array([r["mean_rmse"] for r in rows], dtype=float),
        min_rmse=np.array([r["min_rmse"] for r in rows], dtype=float),
        max_rmse=np.array([r["max_rmse"] for r in rows], dtype=float),
        runtime_s=np.array([r["runtime_s"] for r in rows], dtype=float),
        sigma_t_history=np.stack(sigma_t_histories, axis=0),
        ess_history=np.stack(ess_histories, axis=0),
        particle_count_history=np.stack(particle_count_histories, axis=0),
        accel_history=np.stack(accel_histories, axis=0),
        jerk_history=np.stack(jerk_histories, axis=0),
        collision_history=np.stack(collision_histories, axis=0),
    )

    # Save a minimal config snapshot
    config_path = out_dir / "config.txt"
    config_lines = [
        f"mode={params.mode}",
        f"seed={params.seed}",
        f"num_trials={params.num_trials}",
        f"num_steps={params.num_steps}",
        f"num_particles={params.num_particles}",
        f"particle_init_mode={params.particle_init_mode}",
        f"init_radius_m={params.init_radius_m}",
        f"aggression={params.aggression}",
        f"dt={params.dt}",
        f"motion_noise_std={params.motion_noise_std.tolist()}",
        f"sensor_noise_std={params.sensor_noise_std}",
        "sensor_model_type=raycast",
        f"weight_backend={params.weight_backend}",
        f"gpu_device_id={params.gpu_device_id}",
        f"ray_num_beams={params.ray_num_beams}",
        f"weight_beam_stride={params.weight_beam_stride}",
        f"ray_fov_deg={params.ray_fov_deg}",
        f"ray_max_range={params.ray_max_range}",
        f"safety_margin={params.safety_margin}",
        f"adaptive_noise={params.adaptive_noise}",
        f"adaptive_noise_alpha={params.adaptive_noise_alpha}",
        f"adaptive_noise_beta={params.adaptive_noise_beta}",
        f"adaptive_noise_smoothing={params.adaptive_noise_smoothing}",
        f"adaptive_noise_damping={params.adaptive_noise_damping}",
        f"adaptive_particles={params.adaptive_particles}",
        f"base_particles={resolve_base_particles(params)}",
        f"max_particles={resolve_max_particles(params)}",
        f"ess_threshold={params.ess_threshold}",
        f"ess_high_ratio={params.ess_high_ratio}",
        f"particle_growth_factor={params.particle_growth_factor}",
        f"particle_shrink_step={params.particle_shrink_step}",
        f"spontaneous_collisions={params.spontaneous_collisions}",
        f"collision_probability={params.collision_probability}",
        f"collision_speed_loss_range={params.collision_speed_loss_range.tolist()}",
        f"collision_ang_vel_kick_max={params.collision_ang_vel_kick_max}",
        f"collision_backstep_max={params.collision_backstep_max}",
        f"jerk_std={params.jerk_std.tolist()}",
        f"jerk_distribution={params.jerk_distribution}",
        f"accel_limits={params.accel_limits.tolist()}",
        f"velocity_limits={params.velocity_limits.tolist()}",
        f"ang_velocity_limits={params.ang_velocity_limits.tolist()}",
        f"velocity_reversion_gain={params.velocity_reversion_gain.tolist()}",
    ]
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    if params.use_rich:
        console.print(
            Panel(
                f"Benchmark completed: {params.num_trials} trials\n"
                f"CSV: {csv_path}\nNPZ: {npz_path}",
                title="Benchmark Mode",
                border_style="green",
            )
        )
        table = Table(title="Benchmark Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Trials", str(params.num_trials))
        table.add_row("Steps / Trial", str(params.num_steps))
        table.add_row("Particles", str(params.num_particles))
        table.add_row("Init Mode", params.particle_init_mode)
        if params.particle_init_mode == "local_radius":
            table.add_row("Init Radius (m)", f"{params.init_radius_m:.2f}")
        table.add_row("Aggression", params.aggression)
        table.add_row("Sensor Model", "raycast (square room)")
        table.add_row("Weight Backend", params.weight_backend)
        if params.weight_backend == "cupy":
            table.add_row("GPU Device", str(params.gpu_device_id))
        table.add_row("Weight Beam Stride", str(params.weight_beam_stride))
        table.add_row("Safety Margin", f"{params.safety_margin:.2f}")
        table.add_row("Adaptive Noise", str(params.adaptive_noise))
        if params.adaptive_noise:
            table.add_row("Noise Smoothing", str(params.adaptive_noise_smoothing))
        table.add_row("Adaptive Particles", str(params.adaptive_particles))
        table.add_row("Collisions", str(params.spontaneous_collisions))
        table.add_row("Mean of Mean RMSE", f"{np.mean(mean_rmse_values):.4f}")
        table.add_row("Std of Mean RMSE", f"{np.std(mean_rmse_values):.4f}")
        table.add_row("Mean Final RMSE", f"{np.mean(final_rmse_values):.4f}")
        table.add_row("Avg Runtime / Trial (s)", f"{np.mean(runtimes):.4f}")
        table.add_row("Total Runtime (s)", f"{np.sum(runtimes):.4f}")
        console.print(table)
    else:
        print(f"Benchmark complete. CSV: {csv_path}")
        print(f"Mean of mean RMSE: {np.mean(mean_rmse_values):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Monte Carlo Localization (NumPy)")

    # Mode
    parser.add_argument(
        "--mode",
        choices=["visualize", "benchmark"],
        default="visualize",
        help="Run a single live-visualized trial or multi-trial benchmark.",
    )

    # Core simulation
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps")
    parser.add_argument("--particles", type=int, default=None, help="Number of particles")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--dt", type=float, default=None, help="Time step")
    parser.add_argument("--motion-noise-v", type=float, default=None, help="Sigma of linear velocity noise")
    parser.add_argument("--motion-noise-w", type=float, default=None, help="Sigma of angular velocity noise")
    parser.add_argument("--sensor-noise", type=float, default=None, help="Range sensor sigma")
    parser.add_argument("--ray-beams", type=int, default=None, help="Number of lidar-like beams in raycast mode")
    parser.add_argument("--ray-fov-deg", type=float, default=None, help="Field of view for raycast sensor (deg)")
    parser.add_argument("--ray-max-range", type=float, default=None, help="Maximum range for raycast sensor")
    parser.add_argument("--weight-beam-stride", type=int, default=None, help="Use every Nth beam for particle likelihoods (speed/accuracy tradeoff)")
    parser.add_argument(
        "--weight-backend",
        choices=["numpy", "cupy"],
        default=None,
        help="Backend for particle likelihood raycasting/weights (default: numpy).",
    )
    parser.add_argument("--gpu-device", type=int, default=None, help="CUDA device index for CuPy weight backend")
    parser.add_argument("--adaptive-noise", action="store_true", help="Enable adaptive motion noise based on acceleration/jerk")
    parser.add_argument("--adaptive-noise-alpha", type=float, default=None, help="Alpha coefficient for adaptive noise scaling")
    parser.add_argument("--adaptive-noise-beta", type=float, default=None, help="Beta coefficient for adaptive noise scaling")
    parser.add_argument("--adaptive-noise-smoothing", action="store_true", help="Enable EMA smoothing/damping on sigma_t")
    parser.add_argument("--adaptive-noise-damping", type=float, default=None, help="EMA damping gamma in [0,1] for sigma_t (higher = smoother)")
    parser.add_argument("--adaptive-particles", action="store_true", help="Enable ESS-based adaptive particle count")
    parser.add_argument("--base-particles", type=int, default=None, help="Base particle count target when adaptive particle resizing is enabled")
    parser.add_argument("--max-particles", type=int, default=None, help="Maximum particle count when adaptive particle resizing is enabled")
    parser.add_argument("--ess-threshold", type=float, default=None, help="ESS threshold (ratio <=1 or absolute >1) for particle growth")
    parser.add_argument("--ess-high-ratio", type=float, default=None, help="High-ESS ratio for gradual particle reduction")
    parser.add_argument("--particle-growth-factor", type=float, default=None, help="Growth factor when ESS is low")
    parser.add_argument("--particle-shrink-step", type=int, default=None, help="Particles removed when ESS is high")
    parser.add_argument("--spontaneous-collisions", action="store_true", help="Enable random unmodeled collision disturbances")
    parser.add_argument("--collision-probability", type=float, default=None, help="Per-step collision event probability")
    parser.add_argument("--collision-speed-loss-min", type=float, default=None, help="Min fractional speed loss on collision")
    parser.add_argument("--collision-speed-loss-max", type=float, default=None, help="Max fractional speed loss on collision")
    parser.add_argument("--collision-ang-kick-max", type=float, default=None, help="Max one-step angular velocity kick on collision")
    parser.add_argument("--collision-backstep-max", type=float, default=None, help="Max instantaneous backstep distance on collision")
    parser.add_argument("--safety-margin", type=float, default=None, help="Inset margin from room walls for robot/particles")
    parser.add_argument(
        "--init-mode",
        choices=["local_radius", "global", "local"],
        default=None,
        help="Particle initialization mode (default: local_radius).",
    )
    parser.add_argument("--init-radius", type=float, default=None, help="Spawn radius (m) around start pose for local_radius init")

    # Dynamics parameters (random acceleration + jerk)
    parser.add_argument("--v0", type=float, default=None, help="Initial linear velocity")
    parser.add_argument("--w0", type=float, default=None, help="Initial angular velocity")
    parser.add_argument(
        "--aggression",
        choices=["low", "medium", "high"],
        default=None,
        help="Preset for stochastic acceleration/jerk aggressiveness.",
    )
    parser.add_argument("--jerk-std-v", type=float, default=None, help="Jerk std for linear acceleration")
    parser.add_argument("--jerk-std-w", type=float, default=None, help="Jerk std for angular acceleration")
    parser.add_argument(
        "--jerk-dist",
        choices=["uniform", "gaussian"],
        default=None,
        help="Distribution used for jerk samples (default: uniform).",
    )
    parser.add_argument("--accel-max-v", type=float, default=None, help="Max |a_v|")
    parser.add_argument("--accel-max-w", type=float, default=None, help="Max |a_w|")
    parser.add_argument("--v-min", type=float, default=None, help="Min linear velocity")
    parser.add_argument("--v-max", type=float, default=None, help="Max linear velocity")
    parser.add_argument("--w-min", type=float, default=None, help="Min angular velocity")
    parser.add_argument("--w-max", type=float, default=None, help="Max angular velocity")

    # Visualize mode
    parser.add_argument("--viz-stride", type=int, default=None, help="Update live plot every N steps")
    parser.add_argument("--viz-pause", type=float, default=None, help="Fixed pause per visual update (seconds)")
    parser.add_argument("--viz-time-scale", type=float, default=None, help="Scale factor for sim-timed playback (>1 slower)")
    parser.add_argument("--no-viz-follow-sim", action="store_true", help="Use fixed pause instead of dt-based playback")
    parser.add_argument("--no-window", action="store_true", help="Disable showing live matplotlib window")
    parser.add_argument("--viz-dir", type=str, default=None, help="Output directory for visualize-mode plots")
    parser.add_argument("--no-viz-snapshot", action="store_true", help="Do not save final live-view snapshot")
    parser.add_argument("--viz-particle-size", type=float, default=None, help="Live particle marker size")
    parser.add_argument("--viz-particle-alpha", type=float, default=None, help="Live particle marker alpha")
    parser.add_argument("--no-viz-rays", action="store_true", help="Disable live ray-beam drawing (faster)")

    # Benchmark mode
    parser.add_argument("--trials", type=int, default=None, help="Number of benchmark trials")
    parser.add_argument("--benchmark-dir", type=str, default=None, help="Benchmark output directory")

    # UI
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich terminal UI")
    return parser.parse_args()


def apply_cli_overrides(params: Params, args: argparse.Namespace) -> Params:
    params.mode = args.mode
    if args.aggression is not None:
        apply_aggression_preset(params, args.aggression)

    if args.steps is not None:
        params.num_steps = args.steps
    if args.particles is not None:
        params.num_particles = args.particles
    if args.seed is not None:
        params.seed = args.seed
    if args.dt is not None:
        params.dt = args.dt
    if args.motion_noise_v is not None:
        params.motion_noise_std[0] = args.motion_noise_v
    if args.motion_noise_w is not None:
        params.motion_noise_std[1] = args.motion_noise_w
    if args.sensor_noise is not None:
        params.sensor_noise_std = args.sensor_noise
    if args.ray_beams is not None:
        params.ray_num_beams = max(1, args.ray_beams)
    if args.ray_fov_deg is not None:
        params.ray_fov_deg = float(np.clip(args.ray_fov_deg, 1.0, 360.0))
    if args.ray_max_range is not None:
        params.ray_max_range = max(0.1, args.ray_max_range)
    if args.weight_beam_stride is not None:
        params.weight_beam_stride = max(1, args.weight_beam_stride)
    if args.weight_backend is not None:
        params.weight_backend = args.weight_backend
    if args.gpu_device is not None:
        params.gpu_device_id = max(0, int(args.gpu_device))
    if args.adaptive_noise:
        params.adaptive_noise = True
    if args.adaptive_noise_alpha is not None:
        params.adaptive_noise_alpha = float(args.adaptive_noise_alpha)
    if args.adaptive_noise_beta is not None:
        params.adaptive_noise_beta = float(args.adaptive_noise_beta)
    if args.adaptive_noise_smoothing:
        params.adaptive_noise_smoothing = True
    if args.adaptive_noise_damping is not None:
        params.adaptive_noise_damping = float(np.clip(args.adaptive_noise_damping, 0.0, 1.0))
    if args.adaptive_particles:
        params.adaptive_particles = True
    if args.base_particles is not None:
        params.base_particles = max(1, args.base_particles)
    if args.max_particles is not None:
        params.max_particles = max(1, args.max_particles)
    if args.ess_threshold is not None:
        params.ess_threshold = float(args.ess_threshold)
    if args.ess_high_ratio is not None:
        params.ess_high_ratio = float(np.clip(args.ess_high_ratio, 0.0, 1.0))
    if args.particle_growth_factor is not None:
        params.particle_growth_factor = max(1.01, float(args.particle_growth_factor))
    if args.particle_shrink_step is not None:
        params.particle_shrink_step = max(1, int(args.particle_shrink_step))
    if args.spontaneous_collisions:
        params.spontaneous_collisions = True
    if args.collision_probability is not None:
        params.collision_probability = float(np.clip(args.collision_probability, 0.0, 1.0))
    if args.collision_speed_loss_min is not None:
        params.collision_speed_loss_range[0] = float(np.clip(args.collision_speed_loss_min, 0.0, 1.0))
    if args.collision_speed_loss_max is not None:
        params.collision_speed_loss_range[1] = float(np.clip(args.collision_speed_loss_max, 0.0, 1.0))
    if params.collision_speed_loss_range[1] < params.collision_speed_loss_range[0]:
        params.collision_speed_loss_range[1] = params.collision_speed_loss_range[0]
    if args.collision_ang_kick_max is not None:
        params.collision_ang_vel_kick_max = max(0.0, float(args.collision_ang_kick_max))
    if args.collision_backstep_max is not None:
        params.collision_backstep_max = max(0.0, float(args.collision_backstep_max))
    if args.safety_margin is not None:
        params.safety_margin = max(0.0, args.safety_margin)
    if args.init_mode is not None:
        params.particle_init_mode = args.init_mode
    if args.init_radius is not None:
        params.init_radius_m = max(0.05, args.init_radius)

    if args.v0 is not None:
        params.v0 = args.v0
    if args.w0 is not None:
        params.w0 = args.w0
    if args.jerk_std_v is not None:
        params.jerk_std[0] = args.jerk_std_v
    if args.jerk_std_w is not None:
        params.jerk_std[1] = args.jerk_std_w
    if args.jerk_dist is not None:
        params.jerk_distribution = args.jerk_dist
    if args.accel_max_v is not None:
        params.accel_limits[0] = args.accel_max_v
    if args.accel_max_w is not None:
        params.accel_limits[1] = args.accel_max_w
    if args.v_min is not None:
        params.velocity_limits[0] = args.v_min
    if args.v_max is not None:
        params.velocity_limits[1] = args.v_max
    if args.w_min is not None:
        params.ang_velocity_limits[0] = args.w_min
    if args.w_max is not None:
        params.ang_velocity_limits[1] = args.w_max

    if args.viz_stride is not None:
        params.viz_stride = max(1, args.viz_stride)
    if args.viz_pause is not None:
        params.viz_pause_s = max(0.0, args.viz_pause)
    if args.viz_time_scale is not None:
        params.viz_time_scale = max(0.05, args.viz_time_scale)
    if args.no_viz_follow_sim:
        params.viz_follow_sim_timing = False
    if args.no_window:
        params.viz_show_window = False
    if args.viz_dir is not None:
        params.viz_output_dir = args.viz_dir
    if args.no_viz_snapshot:
        params.viz_save_snapshot = False
    if args.viz_particle_size is not None:
        params.viz_particle_size = max(1.0, args.viz_particle_size)
    if args.viz_particle_alpha is not None:
        params.viz_particle_alpha = float(np.clip(args.viz_particle_alpha, 0.05, 1.0))
    if args.no_viz_rays:
        params.viz_draw_rays = False

    if args.trials is not None:
        params.num_trials = max(1, args.trials)
    if args.benchmark_dir is not None:
        params.benchmark_output_dir = args.benchmark_dir

    if args.no_rich:
        params.use_rich = False
    if params.weight_backend == "cupy" and not cupy_is_available():
        raise RuntimeError(
            "CuPy backend requested but CuPy is not installed. "
            "Install a CUDA-matched package (e.g. `uv add cupy-cuda12x`) or use --weight-backend numpy."
        )
    return params


def main() -> None:
    args = parse_args()
    params = apply_cli_overrides(Params(), args)
    console = Console()

    if params.mode == "benchmark":
        run_benchmark_mode(params, console)
    else:
        run_visualize_mode(params, console)


if __name__ == "__main__":
    main()
