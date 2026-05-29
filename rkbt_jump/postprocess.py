"""Posterior summaries for the model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .likelihood import RKBTMarginalLikelihood


@dataclass
class RKBTPosteriorSummary:
    posterior_null_prob: float
    log_bf01: float
    log_bf10: float
    mcse_naive: float
    psi_samples: np.ndarray
    p_samples: np.ndarray
    posterior_mean_mu: np.ndarray


def unpack_cold_chain(sampler, idx_tau: int, *, discard: int = 0):
    """Extract cold-chain tau values and active indices."""
    chain = sampler.get_chain(discard=discard)
    inds = sampler.get_inds(discard=discard)

    chain_components = chain["components"]
    inds_components = inds["components"]

    if chain_components.ndim != 5:
        raise ValueError(
            "Unexpected chain dimensionality. Expected "
            "(nsteps, ntemps, nwalkers, nleaves_max, ndim)."
        )
    if inds_components.ndim != 4:
        raise ValueError(
            "Unexpected inds dimensionality. Expected "
            "(nsteps, ntemps, nwalkers, nleaves_max)."
        )

    tau_chain = chain_components[:, 0, ..., idx_tau]
    inds_chain = inds_components[:, 0, ...]
    return tau_chain, inds_chain


def summarize_posterior(
    likelihood: RKBTMarginalLikelihood,
    tau_chain: np.ndarray,
    inds_chain: np.ndarray,
    *,
    pi0: float = 0.5,
    eps_prob: float = 1e-10,
) -> RKBTPosteriorSummary:
    """Compute Rao-Blackwell posterior-null estimate and related summaries."""
    nsteps, nwalkers, nleaves_max = tau_chain.shape
    psi = np.empty((nsteps, nwalkers), dtype=float)
    p = inds_chain.sum(axis=-1).astype(int)
    mu_contrib = np.zeros((nsteps, nwalkers, likelihood.ts.grid.size), dtype=float)

    for s in range(nsteps):
        for w in range(nwalkers):
            active = inds_chain[s, w]
            idx = likelihood.ts.tau_to_grid_index(tau_chain[s, w, active])
            stats = likelihood.stats_from_indices(idx)

            psi_val = stats.psi_spike
            if np.isfinite(psi_val):
                psi[s, w] = psi_val
            else:
                warnings.warn(f"NaN psi value at step {s}, walker {w}.")
                psi[s, w] = np.nan
            if idx.size > 0:
                mu_contrib[s, w] = (1.0 - psi[s, w]) * (
                    stats.mean_b_slab @ likelihood.K[idx]
                )

    psi_flat = psi.reshape(-1)
    psi_valid = psi_flat[np.isfinite(psi_flat)]
    if psi_valid.size == 0:
        posterior_null_prob = np.nan
        mcse_naive = np.nan
    else:
        # Clip probabilities to [eps_prob, 1-eps_prob] to avoid numerical issues.
        posterior_null_prob = np.clip(np.mean(psi_valid), eps_prob, 1.0 - eps_prob)
        if psi_valid.size > 1:
            mcse_naive = psi_valid.std(ddof=1) / np.sqrt(psi_valid.size)
        else:
            mcse_naive = np.nan

    log_bf01 = (
        np.log((1 - pi0) / pi0)
        + np.log(posterior_null_prob)
        - np.log1p(-posterior_null_prob)
    )
    log_bf10 = -log_bf01

    posterior_mean_mu = mu_contrib.reshape(-1, mu_contrib.shape[-1]).mean(axis=0)

    return RKBTPosteriorSummary(
        posterior_null_prob=posterior_null_prob,
        log_bf01=log_bf01,
        log_bf10=log_bf10,
        mcse_naive=mcse_naive,
        psi_samples=psi,
        p_samples=p,
        posterior_mean_mu=posterior_mean_mu,
    )


def posterior_draws_b_and_mu(
    likelihood: RKBTMarginalLikelihood,
    tau_chain: np.ndarray,
    inds_chain: np.ndarray,
    *,
    ndraws: int | None = None,
    ndraws_extra: int = 0,
    rng: np.random.Generator | None = None,
):
    """Draw posterior samples for b and mu via the conditional mixture formula."""
    if rng is None:
        rng = np.random.default_rng()
    flat_tau = tau_chain.reshape(-1, tau_chain.shape[-1])
    flat_inds = inds_chain.reshape(-1, inds_chain.shape[-1])
    n_total = flat_tau.shape[0]

    if ndraws is None or ndraws >= n_total:
        idx_keep = np.arange(n_total)
    else:
        idx_keep = rng.choice(n_total, size=int(ndraws), replace=False)

    n_keep = idx_keep.size
    n_repeats = int(ndraws_extra) + 1
    ngrid = likelihood.ts.grid.size
    mu_draws = np.zeros((n_keep * n_repeats, ngrid), dtype=float)
    psi_draws = np.empty(n_keep, dtype=float)
    spike_draws = np.zeros(n_keep, dtype=bool)
    p_draws = np.empty(n_keep, dtype=int)
    tau_index_draws = []
    b_draws = []

    for i, idx_chain in enumerate(idx_keep):
        active = flat_inds[idx_chain]
        tau_active = flat_tau[idx_chain, active]
        idx_tau = likelihood.ts.tau_to_grid_index(tau_active)
        stats = likelihood.stats_from_indices(idx_tau)

        psi_draws[i] = stats.psi_spike
        p_draws[i] = idx_tau.size
        tau_index_draws.append(idx_tau.copy())

        is_spike = bool(rng.uniform() < stats.psi_spike)
        spike_draws[i] = is_spike

        for m in range(n_repeats):
            draw_idx = i * n_repeats + m
            if is_spike or idx_tau.size == 0:
                b_draws.append(np.zeros(idx_tau.size, dtype=float))
                mu_draws[draw_idx] = 0.0
                continue

            b = rng.multivariate_normal(stats.mean_b_slab, stats.cov_b_slab)
            b_draws.append(b)
            mu_draws[draw_idx] = b @ likelihood.K[idx_tau]

    return {
        "mu_draws": mu_draws,
        "b_draws": b_draws,
        "psi_draws": psi_draws,
        "spike_draws": spike_draws,
        "p_draws": p_draws,
        "tau_index_draws": tau_index_draws,
        "chain_indices": idx_keep,
    }


def simultaneous_credible_band(
    mu_draws: np.ndarray,
    posterior_mean_mu: np.ndarray,
    *,
    alpha: float = 0.05,
    eps: float = 1e-10,
):
    """Compute simultaneous credible bands from posterior draws
    based on quantiles of ``sup_t |mu(t)-mu_hat(t)|/sigma(t)``.

    Parameters
    ----------
    mu_draws
        Posterior draws of the latent mean function with shape
        ``(ndraws, ngrid)``.
    posterior_mean_mu
        Posterior mean function on the same grid.
    alpha
        Tail probability for the simultaneous band.
    eps
        Small tolerance for numerical stability.
    """
    mu_draws = np.asarray(mu_draws, dtype=float)
    posterior_mean_mu = np.asarray(posterior_mean_mu, dtype=float)

    if mu_draws.ndim != 2:
        raise ValueError("`mu_draws` must be a 2D array of shape (ndraws, ngrid).")
    if posterior_mean_mu.ndim != 1:
        raise ValueError("`posterior_mean_mu` must be a 1D array.")
    if mu_draws.shape[1] != posterior_mean_mu.size:
        raise ValueError(
            "Dimension mismatch between `mu_draws` and `posterior_mean_mu`."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be in (0, 1).")

    # Posterior standard deviation at each grid point
    sigma = np.std(mu_draws, axis=0, ddof=1)
    sigma = np.maximum(sigma, eps)  # avoid division by zero

    # Standardized absolute deviations
    abs_dev = np.abs(mu_draws - posterior_mean_mu[None, :])
    standardized_dev = abs_dev / sigma[None, :]

    # Supremum over the grid for each draw
    max_std_dev = np.max(standardized_dev, axis=1)

    # Quantile of the supremum
    q = np.quantile(max_std_dev, 1.0 - alpha)

    lower = posterior_mean_mu - q * sigma
    upper = posterior_mean_mu + q * sigma

    return lower, upper, q

def compute_rope(
    mu_draws: np.ndarray,
    epsilon: float = 0.05,
):
    """Compute the region of practical equivalence."""
    return np.mean(np.max(np.abs(mu_draws), axis=1) < epsilon)
