"""Covariance estimation for unknown K."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import OAS


class SampleCovEstimator:
    """Sample-covariance estimator for an unknown covariance kernel on a grid.

    When ``shrinkage=True`` (default), the Oracle Approximating Shrinkage
    (OAS) estimator from :func:`sklearn.covariance.OAS` is used::

        K_oas = (1 - rho) * K_sample + rho * (tr(K_sample) / G) * I_G
    """

    def __init__(
        self,
        *,
        shrinkage: bool = True,
        ddof: int = 1,
        eps: float = 1e-10,
        jitter: float = 1e-10,
    ) -> None:
        self.shrinkage = bool(shrinkage)
        self.ddof = int(ddof)
        self.eps = eps
        self.jitter = jitter

        if self.ddof < 0:
            raise ValueError("`ddof` must be >= 0.")
        if self.eps < 0:
            raise ValueError("`eps` must be >= 0.")
        if self.jitter < 0:
            raise ValueError("`jitter` must be >= 0.")

    def estimate(self, X: np.ndarray) -> np.ndarray:
        """Estimate the covariance matrix on a grid from functional data.

        Parameters
        ----------
        X : (nsamples, ngrid) array

        Returns
        -------
        cov : (ngrid, ngrid) array
            Estimated covariance matrix (symmetric, positive definite).
        """
        X = np.asarray(X, dtype=float)

        if self.shrinkage:
            cov = OAS().fit(X).covariance_
        else:
            cov = np.cov(X, rowvar=False, ddof=self.ddof)

            # Enforce pd
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, self.eps)
            cov = (evecs * evals) @ evecs.T

        if self.jitter > 0:
            cov += self.jitter * np.eye(cov.shape[0], dtype=cov.dtype)

        # Symmetry safeguard
        cov = 0.5 * (cov + cov.T)

        self.mean_ = X.mean(axis=0)
        self.cov_grid_ = cov

        return cov