"""Systematic resampling for particle filters."""

from __future__ import annotations

import numpy as np


def resample_particles(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    num_output_particles: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform systematic resampling and return uniform weights.

    Parameters
    ----------
    num_output_particles:
        Optional output particle count. If ``None``, preserves the input count.
    """
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    n_in = particles.shape[0]
    n_out = n_in if num_output_particles is None else int(num_output_particles)
    if n_out <= 0:
        raise ValueError("num_output_particles must be positive")

    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # Guard against numerical drift.

    u0 = rng.random() / n_out
    positions = u0 + np.arange(n_out, dtype=float) / n_out

    # Systematic resampling can be implemented via searchsorted on the CDF.
    indices = np.searchsorted(cdf, positions, side="left")
    resampled_particles = particles[indices]
    resampled_weights = np.full(n_out, 1.0 / n_out, dtype=float)
    return resampled_particles, resampled_weights
