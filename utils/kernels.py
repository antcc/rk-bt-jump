"""Kernel functions."""

from __future__ import annotations

import numpy as np


def _as_1d(X: np.ndarray | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(X, dtype=float))


def squared_exponential_kernel(
    t: np.ndarray | float,
    s: np.ndarray | float,
    *,
    length_scale: float = 0.2,
    variance: float = 1.0,
) -> np.ndarray:
    """Squared-exponential kernel."""
    if length_scale <= 0:
        raise ValueError("`length_scale` must be > 0.")
    tt = _as_1d(t)
    ss = _as_1d(s)
    diff = (tt[:, None] - ss[None, :]) / length_scale
    return variance * np.exp(-0.5 * diff**2)


def brownian_kernel(
    t: np.ndarray | float,
    s: np.ndarray | float,
    *,
    variance: float = 1.0,
) -> np.ndarray:
    """Brownian-motion kernel."""
    tt = _as_1d(t)
    ss = _as_1d(s)
    return variance * np.minimum.outer(tt, ss)


def ornstein_uhlenbeck_kernel(
    t: np.ndarray | float,
    s: np.ndarray | float,
    *,
    length_scale: float = 0.2,
    variance: float = 1.0,
) -> np.ndarray:
    """Ornstein-Uhlenbeck kernel."""
    if length_scale <= 0:
        raise ValueError("`length_scale` must be > 0.")
    tt = _as_1d(t)
    ss = _as_1d(s)
    dist = np.abs(tt[:, None] - ss[None, :])
    return variance * np.exp(-dist / length_scale)
