"""RMSE utilities for Monte Carlo Localization benchmarking."""

from __future__ import annotations

import numpy as np


def compute_rmse(truth_xy: np.ndarray, estimate_xy: np.ndarray) -> float:
    """Compute position RMSE for one or more 2D samples.

    Parameters
    ----------
    truth_xy:
        Shape ``(2,)`` or ``(N, 2)``
    estimate_xy:
        Shape ``(2,)`` or ``(N, 2)``
    """
    diff = np.asarray(truth_xy, dtype=float) - np.asarray(estimate_xy, dtype=float)
    diff_2d = np.atleast_2d(diff)
    squared_error_per_row = np.sum(diff_2d**2, axis=1)
    return float(np.sqrt(np.mean(squared_error_per_row)))

