"""Utility functions."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def make_K_grid(
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    grid: np.ndarray,
    *,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Evaluate K on a grid and (optionally) add diagonal jitter."""
    grid = np.asarray(grid, dtype=float)
    K = np.asarray(kernel_fn(grid, grid), dtype=float)
    if K.shape != (grid.size, grid.size):
        raise ValueError("`kernel_fn(grid, grid)` must return a square matrix.")
    K = 0.5 * (K + K.T)
    if jitter > 0:
        K += jitter * np.eye(K.shape[0], dtype=K.dtype)
    return K


def make_mu_grid(
    K: np.ndarray,
    atoms_idx: Sequence[int],
    coeffs: Sequence[float],
) -> np.ndarray:
    """Construct mu(s)=sum_j beta_j K(t_j,s) on the grid."""
    K = np.asarray(K, dtype=float)
    atoms_idx = np.asarray(atoms_idx, dtype=int)
    coeffs = np.asarray(coeffs, dtype=float)

    if atoms_idx.size != coeffs.size:
        raise ValueError("`atoms_idx` and `coeffs` must have the same length.")
    if np.any(atoms_idx < 0) or np.any(atoms_idx >= K.shape[0]):
        raise ValueError("`atoms_idx` contains out-of-range indices.")

    return coeffs @ K[atoms_idx]
