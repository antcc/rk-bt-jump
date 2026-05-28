"""Bayesian one-sample functional test with RJMCMC (Eryn backend)."""

from .covariance import (
    SampleCovEstimator,
)
from .kernels import (
    brownian_kernel,
    ornstein_uhlenbeck_kernel,
    squared_exponential_kernel,
)
from .postprocess import (
    RKBTPosteriorSummary,
    compute_rope,
    posterior_draws_b_and_mu,
    simultaneous_credible_band,
    summarize_posterior,
)
from .sampler import (
    RKBTConfig,
    RKBTFitResult,
    RKBTModel,
    fit_rkbt,
)
from .utils import (
    make_K_grid,
    make_mu_grid,
)

__all__ = [
    "brownian_kernel",
    "compute_rope",
    "fit_rkbt",
    "make_K_grid",
    "make_mu_grid",
    "ornstein_uhlenbeck_kernel",
    "posterior_draws_b_and_mu",
    "RKBTConfig",
    "RKBTFitResult",
    "RKBTModel",
    "RKBTPosteriorSummary",
    "SampleCovEstimator",
    "simultaneous_credible_band",
    "squared_exponential_kernel",
    "summarize_posterior",
]
