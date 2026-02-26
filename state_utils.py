"""State propagation and estimation helpers."""

from __future__ import annotations

import numpy as np

from motion_model import wrap_to_pi


def propagate_state(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Deterministic unicycle propagation for ground-truth state."""
    state = np.asarray(state, dtype=float)
    v, w = np.asarray(control, dtype=float)
    x, y, theta = state

    return np.array(
        [
            x + v * dt * np.cos(theta),
            y + v * dt * np.sin(theta),
            wrap_to_pi(theta + w * dt),
        ],
        dtype=float,
    )


def estimate_state(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean estimate for [x, y, theta]."""
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    x_hat = np.zeros(3, dtype=float)
    x_hat[0] = np.sum(weights * particles[:, 0])
    x_hat[1] = np.sum(weights * particles[:, 1])

    # Circular mean for heading.
    c = np.sum(weights * np.cos(particles[:, 2]))
    s = np.sum(weights * np.sin(particles[:, 2]))
    x_hat[2] = np.arctan2(s, c)
    return x_hat

