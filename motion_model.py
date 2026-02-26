"""Motion model for 2D MCL."""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def motion_model(
    particles: np.ndarray,
    control: np.ndarray,
    dt: float,
    motion_noise_std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Propagate particles with a unicycle model and Gaussian control noise.

    Parameters
    ----------
    particles:
        Array of shape ``(N, 3)`` with columns ``[x, y, theta]``.
    control:
        Array-like ``[v, w]`` (linear and angular velocity).
    dt:
        Time step.
    motion_noise_std:
        Array-like ``[sigma_v, sigma_w]``.
    rng:
        NumPy random generator for reproducibility.
    """
    particles = np.asarray(particles, dtype=float)
    v, w = np.asarray(control, dtype=float)
    sigma_v, sigma_w = np.asarray(motion_noise_std, dtype=float)

    n = particles.shape[0]

    # Sample noisy controls independently for each particle.
    v_noisy = v + rng.normal(0.0, sigma_v, size=n)
    w_noisy = w + rng.normal(0.0, sigma_w, size=n)

    theta = particles[:, 2]

    updated = particles.copy()
    updated[:, 0] = particles[:, 0] + v_noisy * dt * np.cos(theta)
    updated[:, 1] = particles[:, 1] + v_noisy * dt * np.sin(theta)
    updated[:, 2] = wrap_to_pi(theta + w_noisy * dt)
    return updated

