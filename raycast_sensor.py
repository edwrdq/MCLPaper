"""Raycasting-based range sensor model (distance-sensor / lidar-like)."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None


def _ensure_pose_array(poses: np.ndarray) -> tuple[np.ndarray, bool]:
    poses = np.asarray(poses, dtype=float)
    if poses.ndim == 1:
        return poses[None, :], True
    return poses, False


def _ensure_pose_array_xp(poses, xp, dtype=None):
    poses_arr = xp.asarray(poses, dtype=(float if dtype is None else dtype))
    squeeze = bool(getattr(poses_arr, "ndim", 0) == 1)
    if squeeze:
        poses_arr = poses_arr[None, :]
    return poses_arr, squeeze


def _raycast_distances_xp(
    poses,
    beam_angles,
    world,
    max_range: float,
    xp,
    dtype=None,
):
    poses_arr, squeeze = _ensure_pose_array_xp(poses, xp, dtype=dtype)
    arr_dtype = poses_arr.dtype
    beam_angles = xp.asarray(beam_angles, dtype=arr_dtype)
    beam_angles = beam_angles.reshape((1,) * (poses_arr.ndim - 1) + (-1,))
    world = xp.asarray(world, dtype=arr_dtype)

    x = poses_arr[..., 0:1]
    y = poses_arr[..., 1:2]
    theta = poses_arr[..., 2:3]

    angles = theta + beam_angles
    c = xp.cos(angles)
    s = xp.sin(angles)

    # Avoid division by zero for near-axis-aligned beams.
    eps = arr_dtype.type(1e-12)
    c_safe = xp.where(xp.abs(c) < eps, xp.sign(c) * eps + (c == 0) * eps, c)
    s_safe = xp.where(xp.abs(s) < eps, xp.sign(s) * eps + (s == 0) * eps, s)

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

    inf = xp.inf
    d_tx_min = xp.where(valid_tx_min, tx_min, inf)
    d_tx_max = xp.where(valid_tx_max, tx_max, inf)
    d_ty_min = xp.where(valid_ty_min, ty_min, inf)
    d_ty_max = xp.where(valid_ty_max, ty_max, inf)

    distances = xp.minimum(xp.minimum(d_tx_min, d_tx_max), xp.minimum(d_ty_min, d_ty_max))
    distances = xp.clip(distances, arr_dtype.type(0.0), arr_dtype.type(max_range))
    if squeeze:
        return distances[0]
    return distances


def raycast_distances(
    poses: np.ndarray,
    beam_angles: np.ndarray,
    world: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Cast rays to rectangular world boundaries and return distances.

    Parameters
    ----------
    poses:
        Shape ``(3,)`` or ``(N, 3)`` with columns ``[x, y, theta]``.
    beam_angles:
        Shape ``(B,)`` beam angles relative to robot heading (radians).
    world:
        Shape ``(2, 2)`` with ``[[x_min, x_max], [y_min, y_max]]``.
    max_range:
        Maximum sensor range.

    Returns
    -------
    np.ndarray
        Shape ``(B,)`` for a single pose input or ``(N, B)`` for batched poses.
    """
    return np.asarray(_raycast_distances_xp(poses, beam_angles, world, max_range, np), dtype=float)


def ray_sensor_model(
    particles: np.ndarray,
    measured_ranges: np.ndarray,
    beam_angles: np.ndarray,
    world: np.ndarray,
    sensor_noise_std: float,
    max_range: float,
) -> np.ndarray:
    """Compute log-likelihoods for raycast range measurements."""
    z = np.asarray(measured_ranges, dtype=float).reshape(1, -1)
    predicted = raycast_distances(particles, beam_angles, world, max_range)
    residuals = z - predicted
    var = float(sensor_noise_std) ** 2
    return -0.5 * np.sum((residuals**2) / var, axis=1)


def cupy_is_available() -> bool:
    """Return True if CuPy imported successfully."""
    return cp is not None


def _ray_sensor_model_xp(
    particles,
    measured_ranges,
    beam_angles,
    world,
    sensor_noise_std: float,
    max_range: float,
    xp,
    dtype=None,
):
    predicted = _raycast_distances_xp(particles, beam_angles, world, max_range, xp, dtype=dtype)
    z = xp.asarray(measured_ranges, dtype=predicted.dtype)
    if z.ndim == predicted.ndim - 1:
        z = xp.expand_dims(z, axis=-2)
    residuals = z - predicted
    var = predicted.dtype.type(float(sensor_noise_std) ** 2)
    return -predicted.dtype.type(0.5) * xp.sum((residuals**2) / var, axis=-1)


class CupyRaycastWeighter:
    """CuPy-backed particle raycast likelihood evaluator.

    This ports the raycast + log-likelihood hotspot to GPU while keeping the
    rest of the MCL pipeline in NumPy for minimal code churn.
    """

    def __init__(
        self,
        beam_angles: np.ndarray,
        world: np.ndarray,
        sensor_noise_std: float,
        max_range: float,
        device_id: int = 0,
        use_fp32: bool = True,
    ) -> None:
        if cp is None:
            raise RuntimeError(
                "CuPy backend requested but CuPy is not installed. "
                "Install a CUDA-matched package (e.g. `uv add cupy-cuda12x`)."
            )
        self._cp = cp
        self.device_id = int(device_id)
        self.device = cp.cuda.Device(self.device_id)
        self.dtype = cp.float32 if use_fp32 else cp.float64
        with self.device:
            self.beam_angles = cp.asarray(beam_angles, dtype=self.dtype)
            self.world = cp.asarray(world, dtype=self.dtype)
        self.sensor_noise_std = float(sensor_noise_std)
        self.max_range = float(max_range)
        self.use_fp32 = bool(use_fp32)

    def warmup(self) -> None:
        """Trigger a tiny kernel launch so the first benchmark step isn't skewed."""
        with self.device:
            dummy_particles = self._cp.asarray([[0.0, 0.0, 0.0]], dtype=float)
            dummy_z = self._cp.asarray([0.0], dtype=float)
            _ = _ray_sensor_model_xp(
                dummy_particles,
                dummy_z,
                self.beam_angles[:1],
                self.world,
                self.sensor_noise_std,
                self.max_range,
                self._cp,
                dtype=self.dtype,
            )
            self._cp.cuda.Stream.null.synchronize()

    def log_likelihood(self, particles: np.ndarray, measured_ranges: np.ndarray) -> np.ndarray:
        """Return per-particle log-likelihoods as a NumPy array."""
        with self.device:
            log_w = _ray_sensor_model_xp(
                self._cp.asarray(particles, dtype=self.dtype),
                measured_ranges,
                self.beam_angles,
                self.world,
                self.sensor_noise_std,
                self.max_range,
                self._cp,
                dtype=self.dtype,
            )
            return self._cp.asnumpy(log_w).astype(np.float64, copy=False)

    def log_likelihood_batch(self, particles_batch: np.ndarray, measured_ranges_batch: np.ndarray) -> np.ndarray:
        """Return batched per-particle log-likelihoods for shape (B, P, 3) / (B, M)."""
        with self.device:
            log_w = _ray_sensor_model_xp(
                self._cp.asarray(particles_batch, dtype=self.dtype),
                measured_ranges_batch,
                self.beam_angles,
                self.world,
                self.sensor_noise_std,
                self.max_range,
                self._cp,
                dtype=self.dtype,
            )
            return self._cp.asnumpy(log_w).astype(np.float64, copy=False)
