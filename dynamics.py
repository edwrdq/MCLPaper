"""Stochastic control dynamics with bounded acceleration and jerk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlDynamicsState:
    """Internal state for the simulated control process."""

    v: float
    w: float
    a_v: float
    a_w: float


def step_control_dynamics(
    state: ControlDynamicsState,
    dt: float,
    rng: np.random.Generator,
    *,
    jerk_std: np.ndarray,
    accel_limits: np.ndarray,
    velocity_limits: np.ndarray,
    ang_velocity_limits: np.ndarray,
    jerk_distribution: str = "gaussian",
    nominal_velocity: np.ndarray | None = None,
    velocity_reversion_gain: np.ndarray | None = None,
) -> tuple[ControlDynamicsState, np.ndarray, np.ndarray, np.ndarray]:
    """Advance the control process by one step.

    Model:
      - Jerk is sampled as Gaussian noise each step
      - Acceleration follows a bounded random walk via jerk integration
      - Velocity and angular velocity are integrated from acceleration
      - All values are clipped to configured bounds
    """
    jerk_std = np.asarray(jerk_std, dtype=float)
    accel_limits = np.asarray(accel_limits, dtype=float)
    velocity_limits = np.asarray(velocity_limits, dtype=float)
    ang_velocity_limits = np.asarray(ang_velocity_limits, dtype=float)

    if jerk_distribution == "uniform":
        # Uniform bounded jitter in [-jerk_std, +jerk_std]
        jerk = rng.uniform(low=-jerk_std, high=jerk_std, size=2)
    else:
        jerk = rng.normal(loc=0.0, scale=jerk_std, size=2)

    a_v = np.clip(state.a_v + jerk[0] * dt, -accel_limits[0], accel_limits[0])
    a_w = np.clip(state.a_w + jerk[1] * dt, -accel_limits[1], accel_limits[1])

    v = state.v + a_v * dt
    w = state.w + a_w * dt

    # Optional mean reversion keeps the motion "alive" and reduces
    # persistent spin / stall behavior from random-walk drift.
    if nominal_velocity is not None and velocity_reversion_gain is not None:
        nominal_velocity = np.asarray(nominal_velocity, dtype=float)
        velocity_reversion_gain = np.asarray(velocity_reversion_gain, dtype=float)
        v = v + velocity_reversion_gain[0] * (nominal_velocity[0] - v) * dt
        w = w + velocity_reversion_gain[1] * (nominal_velocity[1] - w) * dt

    v = np.clip(v, velocity_limits[0], velocity_limits[1])
    w = np.clip(w, ang_velocity_limits[0], ang_velocity_limits[1])

    next_state = ControlDynamicsState(v=float(v), w=float(w), a_v=float(a_v), a_w=float(a_w))
    control = np.array([next_state.v, next_state.w], dtype=float)
    accel = np.array([next_state.a_v, next_state.a_w], dtype=float)
    return next_state, control, accel, jerk.astype(float)
