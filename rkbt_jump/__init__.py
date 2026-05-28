"""Bayesian one-sample functional test with RJMCMC (Eryn backend)."""

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

__all__ = [
    "compute_rope",
    "fit_rkbt",
    "posterior_draws_b_and_mu",
    "RKBTConfig",
    "RKBTFitResult",
    "RKBTModel",
    "RKBTPosteriorSummary",
    "simultaneous_credible_band",
    "summarize_posterior",
]
