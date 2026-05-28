"""Parameter-space helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThetaSpace:
    """Parameter indexing and tau-to-grid mapping utilities."""

    grid: np.ndarray
    idx_tau: int = 0  # position of tau in the parameter vector

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("`grid` must be a one-dimensional array.")
        if grid.size < 2:
            raise ValueError("`grid` must contain at least two points.")
        if np.any(np.diff(grid) <= 0):
            raise ValueError("`grid` must be strictly increasing.")
        object.__setattr__(self, "grid", grid)

    @property
    def grid_min(self) -> float:
        return self.grid[0]

    @property
    def grid_max(self) -> float:
        return self.grid[-1]

    def tau_to_grid_index(self, tau: np.ndarray) -> np.ndarray:
        """Map tau values to nearest grid indices."""
        tau = np.asarray(tau, dtype=float)
        return np.abs(self.grid - tau[..., None]).argmin(axis=-1)

