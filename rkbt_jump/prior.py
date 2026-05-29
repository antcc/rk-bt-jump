"""Prior classes for the RKHS model."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.special import gammaln
from scipy.stats import uniform

from .parameters import ThetaSpace


def check_valid_tau_indices(
    tau_idx: np.ndarray,
    inds: np.ndarray,
    *,
    min_dist_tau: int = 0,
) -> np.ndarray:
    """Check minimum distance between active tau indices on each walker."""
    ntemps, nwalkers, _ = tau_idx.shape
    valid = np.ones((ntemps, nwalkers), dtype=bool)

    for t in range(ntemps):
        for w in range(nwalkers):
            active = tau_idx[t, w, inds[t, w]]
            if active.size <= 1:
                continue
            active_sorted = np.sort(active)
            if np.any(np.diff(active_sorted) <= min_dist_tau):
                valid[t, w] = False
    return valid


def generate_valid_tau(
    size: Iterable[int],
    grid: np.ndarray,
    *,
    min_dist_tau: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate tau values on grid from prior while enforcing minimum index separation."""
    if rng is None:
        rng = np.random.default_rng()

    ntemps, nwalkers, nleaves_max = size
    grid = np.asarray(grid, dtype=float)

    if nleaves_max > grid.size:
        raise ValueError("`nleaves_max` cannot exceed `grid.size`.")

    tau = np.empty((ntemps, nwalkers, nleaves_max), dtype=float)
    all_idx = np.arange(grid.size)

    for t in range(ntemps):
        for w in range(nwalkers):
            chosen = []
            available = all_idx.copy()
            for _ in range(nleaves_max):
                if available.size == 0:
                    # Fallback: if the min-distance condition is too strict, allow reuse.
                    idx = int(rng.integers(grid.size))
                else:
                    idx = int(rng.choice(available))
                chosen.append(idx)
                keep = np.abs(available - idx) > min_dist_tau
                available = available[keep]
            rng.shuffle(chosen)
            tau[t, w] = grid[np.asarray(chosen, dtype=int)]

    return tau


class TauRJPrior:
    """Joint prior object used by Eryn when
    `all_models_together` is enabled."""

    def __init__(
        self,
        theta_space: ThetaSpace,
        *,
        lambda_p: float | None = None,
        min_dist_tau: int = 0,
    ) -> None:
        self.ts = theta_space
        self.lambda_p = lambda_p
        self.min_dist_tau = int(min_dist_tau)
        self.prior_tau = uniform(
            loc=self.ts.grid_min, scale=self.ts.grid_max - self.ts.grid_min
        )

        # Bookkeeping for Eryn internals
        self.key_order = [self.ts.idx_tau]

        if self.lambda_p is not None and self.lambda_p <= 0:
            raise ValueError("`lambda_p` must be > 0 when provided.")

    def _log_prior_p(self, p: np.ndarray) -> np.ndarray:
        if self.lambda_p is None:
            return np.zeros_like(p, dtype=float)
        # Unnormalized Poisson prior on p (normalizing constant cancels in MCMC acceptance ratios).
        return p * np.log(self.lambda_p) - gammaln(p + 1.0)

    def logpdf_components(
        self,
        coords_components: np.ndarray,
        inds_components: np.ndarray | None = None,
    ) -> np.ndarray:
        tau = coords_components[..., self.ts.idx_tau]
        if inds_components is not None:
            tau = tau[inds_components]
        return self.prior_tau.logpdf(tau)

    def logpdf(self, coords, inds, supps=None, branch_supps=None):  # noqa: D401
        tau = np.asarray(coords["components"][..., self.ts.idx_tau], dtype=float)
        inds_comp = np.asarray(inds["components"], dtype=bool)

        lp_tau = self.prior_tau.logpdf(tau)
        lp_tau[~inds_comp] = 0.0
        lp = lp_tau.sum(axis=-1)

        p = inds_comp.sum(axis=-1).astype(float)
        lp = lp + self._log_prior_p(p)

        idx_tau = self.ts.tau_to_grid_index(tau)
        valid_tau = check_valid_tau_indices(
            idx_tau,
            inds_comp,
            min_dist_tau=self.min_dist_tau,
        )

        lp[~valid_tau] = -np.inf

        return lp

    def rvs(self, size, coords=None, inds=None):
        if isinstance(size, int):
            size_tuple = (size,)
        else:
            size_tuple = tuple(size)

        out = np.empty(size_tuple + (1,), dtype=float)
        out[..., self.ts.idx_tau] = self.prior_tau.rvs(size=size_tuple)
        return out
