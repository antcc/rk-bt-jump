"""Marginal likelihood for the model (integrated spike-and-slab)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import ThetaSpace

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _log_marginal_single_numba(
        idx: np.ndarray,
        x_sum: np.ndarray,
        K: np.ndarray,
        nsamples: int,
        eta_inv2: float,
        log_eta: float,
        log_pi0: float,
        log_1m_pi0: float,
    ) -> float:
        p = idx.shape[0]
        if p == 0:
            # m_slab = 1 and m = pi0 + (1-pi0)*1 = 1.
            return 0.0

        A = np.empty((p, p), dtype=np.float64)
        L = np.empty((p, p), dtype=np.float64)
        y = np.empty(p, dtype=np.float64)

        for i in range(p):
            ii = idx[i]
            y[i] = x_sum[ii]
            for j in range(p):
                jj = idx[j]
                val = nsamples * K[ii, jj]
                if i == j:
                    val += eta_inv2
                A[i, j] = val

        for i in range(p):
            for j in range(i + 1):
                s = A[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s <= 1e-10:
                        return -np.inf
                    L[i, j] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]

            for j in range(i + 1, p):
                L[i, j] = 0.0

        # y = L^{-1} S
        for i in range(p):
            s = y[i]
            for k in range(i):
                s -= L[i, k] * y[k]
            y[i] = s / L[i, i]

        quad = 0.0
        logdet = 0.0
        for i in range(p):
            quad += y[i] * y[i]
            logdet += 2.0 * np.log(L[i, i])

        log_m_slab = -p * log_eta - 0.5 * logdet + 0.5 * quad
        return np.logaddexp(log_pi0, log_1m_pi0 + log_m_slab)


@dataclass
class ConditionalSlabStats:
    """Conditional posterior quantities for fixed (p, tau)."""

    idx_tau: np.ndarray
    log_marginal: float
    log_m_slab: float

    # These are not needed for likelihood evaluation,
    # but are useful to recover the b samples afterwards
    psi_spike: float
    mean_b_slab: np.ndarray
    cov_b_slab: np.ndarray


class RKBTMarginalLikelihood:
    """Evaluate log m(X_{1:n} | p, tau) for Eryn."""

    def __init__(
        self,
        X: np.ndarray,
        K: np.ndarray,
        theta_space: ThetaSpace,
        *,
        pi0: float = 0.5,
        eta: float = 1.0,
        engine: str = "auto",
    ) -> None:
        self.X = np.asarray(X, dtype=float)
        self.X_sum = self.X.sum(axis=0)
        self.n = self.X.shape[0]
        self.K = np.asarray(K, dtype=float)
        self.ts = theta_space

        if self.X.ndim != 2:
            raise ValueError("`X` must have shape (nsamples, ngrid).")
        if self.X.shape[1] != self.ts.grid.size:
            raise ValueError("`X.shape[1]` must equal `len(grid)`.")
        if self.K.shape != (self.ts.grid.size, self.ts.grid.size):
            raise ValueError("`K` has incompatible shape.")
        if not (0.0 < pi0 < 1.0):
            raise ValueError("`pi0` must be in (0, 1).")
        if eta <= 0:
            raise ValueError("`eta` must be > 0.")

        self.pi0 = pi0
        self.eta = eta
        self.eta_inv2 = 1.0 / (self.eta**2)
        self.log_pi0 = np.log(self.pi0)
        self.log_1m_pi0 = np.log1p(-self.pi0)
        self.log_eta = np.log(self.eta)
        self.engine = self._resolve_engine(engine)
        self._stats_cache: dict[tuple[int, ...], ConditionalSlabStats] = {}
        self._log_marginal_cache: dict[tuple[int, ...], float] = {}

    @staticmethod
    def _resolve_engine(engine: str) -> str:
        mode = str(engine).strip().lower()
        if mode == "auto":
            return "numba" if NUMBA_AVAILABLE else "numpy"
        if mode == "numba" and not NUMBA_AVAILABLE:
            return "numpy"
            print("Module numba not found. Falling back to numpy engine.")
        if mode not in {"numpy", "numba"}:
            raise ValueError("`engine` must be one of {'numpy','numba','auto'}.")
        return mode

    def clear_cache(self) -> None:
        """Clear cached values to reuse Likelihood object (not recommended)."""
        self._stats_cache.clear()
        self._log_marginal_cache.clear()

    def _normalize_idx(self, idx_tau: np.ndarray) -> tuple[int, ...]:
        """Sort indices to ensure consistent keys in the cache."""
        idx = np.asarray(idx_tau, dtype=int).ravel()
        if idx.size == 0:
            return tuple()
        return tuple(np.sort(idx).tolist())

    def stats_from_indices(self, idx_tau: np.ndarray) -> ConditionalSlabStats:
        """Compute and cache conditional quantities for fixed active tau indices."""
        key = self._normalize_idx(idx_tau)
        cached = self._stats_cache.get(key)
        if cached is not None:
            return cached

        p = len(key)
        idx = np.asarray(key, dtype=int)

        if p == 0:
            stats = ConditionalSlabStats(
                idx_tau=idx,
                log_marginal=0.0,
                log_m_slab=0.0,
                psi_spike=self.pi0,
                mean_b_slab=np.empty(0, dtype=float),
                cov_b_slab=np.empty((0, 0), dtype=float),
            )
            self._stats_cache[key] = stats
            return stats

        S = self.X_sum[idx]  # S_n(tau)
        K_tau = self.K[np.ix_(idx, idx)]
        I_p = np.eye(p)
        Lambda = self.n * K_tau + self.eta_inv2 * I_p

        try:
            chol = np.linalg.cholesky(Lambda)
            tmp = np.linalg.solve(chol, S)
            quad = tmp @ tmp
            logdet = 2.0 * np.log(np.diag(chol)).sum()

            log_m_slab = -p * self.log_eta - 0.5 * logdet + 0.5 * quad
            log_marginal = np.logaddexp(self.log_pi0, self.log_1m_pi0 + log_m_slab)

            # Stats for recovering b samples later on
            mean_b = np.linalg.solve(chol.T, tmp)
            cov_b = np.linalg.solve(chol.T, np.linalg.solve(chol, I_p))
            psi = np.exp(self.log_pi0 - log_marginal)

        except np.linalg.LinAlgError:
            # Numerical failure should be rejected by MCMC via near-zero likelihood.
            log_m_slab = -np.inf
            log_marginal = -np.inf
            psi = 1.0
            mean_b = np.zeros(p, dtype=float)
            cov_b = np.eye(p, dtype=float)

        stats = ConditionalSlabStats(
            idx_tau=idx,
            log_marginal=log_marginal,
            log_m_slab=log_m_slab,
            psi_spike=psi,
            mean_b_slab=mean_b,
            cov_b_slab=cov_b,
        )
        self._stats_cache[key] = stats
        return stats

    def _log_marginal_from_indices_numpy(self, idx_tau: np.ndarray) -> float:
        return self.stats_from_indices(idx_tau).log_marginal

    def _log_marginal_from_indices_numba(self, idx_tau: np.ndarray) -> float:
        key = self._normalize_idx(idx_tau)
        cached = self._log_marginal_cache.get(key)
        if cached is not None:
            return cached

        idx = np.asarray(key, dtype=np.int64)
        val = _log_marginal_single_numba(
            idx,
            self.X_sum,
            self.K,
            self.n,
            self.eta_inv2,
            self.log_eta,
            self.log_pi0,
            self.log_1m_pi0,
        )
        self._log_marginal_cache[key] = val
        return val

    def _extract_theta_groups(self, theta, groups):
        # Eryn behavior:
        # - single branch: theta/groups are arrays
        # - multi-branch: theta/groups are lists/tuples
        if isinstance(theta, (tuple, list)):
            if len(theta) != 1:
                raise ValueError("This likelihood supports a single branch only.")
            theta_components = theta[0]
        else:
            theta_components = theta

        if isinstance(groups, (tuple, list)):
            if len(groups) != 1:
                raise ValueError("This likelihood supports a single branch only.")
            groups_components = groups[0]
        else:
            groups_components = groups

        return theta_components, groups_components

    def evaluate_vectorized(self, theta, groups) -> np.ndarray:
        """Vectorized log marginal likelihood expected by Eryn."""
        theta_components, groups_components = self._extract_theta_groups(theta, groups)
        tau_all = np.asarray(theta_components[:, self.ts.idx_tau], dtype=float)
        idx_all = self.ts.tau_to_grid_index(tau_all)

        unique_groups = np.unique(groups_components)
        out = np.empty(unique_groups.shape[0], dtype=float)

        for i, gid in enumerate(unique_groups):
            idx_tau = idx_all[groups_components == gid]
            if self.engine == "numba":
                out[i] = self._log_marginal_from_indices_numba(idx_tau)
            else:
                out[i] = self._log_marginal_from_indices_numpy(idx_tau)

        return out
