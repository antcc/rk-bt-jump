"""High-level model assembly and Eryn sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.state import State

from .likelihood import RKBTMarginalLikelihood
from .moves import TauGroupStretchMove, TauRJMove
from .parameters import ThetaSpace
from .postprocess import (
    RKBTPosteriorSummary,
    posterior_draws_b_and_mu,
    summarize_posterior,
    unpack_cold_chain,
)
from .prior import TauRJPrior, generate_valid_tau


@dataclass
class RKBTConfig:
    # Model hyperparameters
    pi0: float = 0.5
    eta: float | None = None
    eta_scaling_factor: float = 1.0
    lambda_p: float | None = None
    min_dist_tau: int = 0
    likelihood_engine: str = "auto"  # numpy | numba | auto

    # Sampler hyperparameters
    nwalkers: int = 32
    ntemps: int = 4
    nsteps: int = 1000
    nburn: int = 500
    thin_by: int = 1
    nleaves_min: int = 1
    nleaves_max: int = 5
    stretch_a: float = 2.0
    group_nfriends: int | None = None
    group_n_iter_update: int = 100
    seed: int = 42


@dataclass
class RKBTFitResult:
    config: RKBTConfig
    theta_space: ThetaSpace
    eta: float
    K: np.ndarray
    likelihood: RKBTMarginalLikelihood
    prior: TauRJPrior
    sampler: EnsembleSampler
    tau_chain: np.ndarray
    inds_chain: np.ndarray
    summary: RKBTPosteriorSummary

    def draw_posterior(
        self,
        *,
        ndraws: int | None = None,
        ndraws_extra: int = 0,
        rng: np.random.Generator | None = None,
    ):
        return posterior_draws_b_and_mu(
            self.likelihood,
            self.tau_chain,
            self.inds_chain,
            ndraws=ndraws,
            ndraws_extra=ndraws_extra,
            rng=rng,
        )


class RKBTModel:
    """Model wrapper for inference.

    Parameters
    ----------
    X : (nsamples, ngrid) array
        Observed functional data on the grid.
    K : (ngrid, ngrid) array
        Discretized covariance matrix on the grid.
    grid : (ngrid,) array
        Observation grid.
    config : RKBTConfig, optional
    """

    def __init__(
        self,
        X: np.ndarray,
        K: np.ndarray,
        grid: np.ndarray,
        *,
        config: RKBTConfig | None = None,
    ) -> None:
        self.config = RKBTConfig() if config is None else config

        X = np.asarray(X, dtype=float)
        K = np.asarray(K, dtype=float)
        grid = np.asarray(grid, dtype=float)

        if X.ndim != 2:
            raise ValueError("`X` must have shape (nsamples, ngrid).")
        if X.shape[1] != grid.size:
            raise ValueError("`X.shape[1]` must be equal to `len(grid)`.")
        if K.shape != (grid.size, grid.size):
            raise ValueError("`K` must have shape (ngrid, ngrid).")
        if self.config.nleaves_max < self.config.nleaves_min:
            raise ValueError("`nleaves_max` must be >= `nleaves_min`.")

        if self.config.eta is None:
            # pointwise_var = np.var(X, axis=0, ddof=1)
            # self.eta = self.config.eta_scaling_factor * np.sqrt(np.mean(pointwise_var))
            self.eta = self.config.eta_scaling_factor * np.sqrt(np.trace(K) / grid.size)
        else:
            self.eta = self.config.eta

        self.X = X
        self.K = K
        self.theta_space = ThetaSpace(grid=grid)
        self.prior = TauRJPrior(
            self.theta_space,
            lambda_p=self.config.lambda_p,
            min_dist_tau=self.config.min_dist_tau,
        )
        self.likelihood = RKBTMarginalLikelihood(
            self.X,
            self.K,
            self.theta_space,
            pi0=self.config.pi0,
            eta=self.eta,
            engine=self.config.likelihood_engine,
        )

        self.branch_names = ["components"]
        self.ndims = {
            "components": 1
        }  # since b is marginalized out, we only have one dimension (tau)
        self.nleaves_min = {"components": self.config.nleaves_min}
        self.nleaves_max = {"components": self.config.nleaves_max}

    def _setup_initial_state(self):
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        coords = {
            "components": np.zeros(
                (cfg.ntemps, cfg.nwalkers, cfg.nleaves_max, self.ndims["components"]),
                dtype=float,
            )
        }
        coords["components"][..., self.theta_space.idx_tau] = generate_valid_tau(
            (cfg.ntemps, cfg.nwalkers, cfg.nleaves_max),
            self.theta_space.grid,
            min_dist_tau=cfg.min_dist_tau,
            rng=rng,
        )

        inds = {
            "components": (
                rng.random(size=(cfg.ntemps, cfg.nwalkers, cfg.nleaves_max)) < 0.5
            )
        }

        # Ensure at least `nleaves_min` active leaves per walker.
        if cfg.nleaves_min > 0:
            active_counts = inds["components"].sum(axis=-1)
            too_few = np.where(active_counts < cfg.nleaves_min)
            inds["components"][*too_few, : cfg.nleaves_min] = True

        return coords, inds

    def _build_sampler(self) -> EnsembleSampler:
        cfg = self.config
        priors = {"all_models_together": self.prior}

        move_tau = TauGroupStretchMove(
            nfriends=cfg.nwalkers if cfg.group_nfriends is None else cfg.group_nfriends,
            n_iter_update=cfg.group_n_iter_update,
            gibbs_sampling_setup="components",
            a=cfg.stretch_a,
        )

        rj_move = TauRJMove(
            self.prior,
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            rj=True,
            gibbs_sampling_setup="components",
        )

        return EnsembleSampler(
            cfg.nwalkers,
            self.ndims,
            self.likelihood.evaluate_vectorized,
            priors,
            vectorize=True,
            provide_groups=True,
            tempering_kwargs=dict(ntemps=cfg.ntemps),
            nbranches=1,
            branch_names=self.branch_names,
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            moves=move_tau,
            rj_moves=rj_move,
        )

    def fit(self, *, progress: bool = True) -> RKBTFitResult:
        cfg = self.config
        np.random.seed(cfg.seed)

        sampler = self._build_sampler()
        coords, inds = self._setup_initial_state()
        state = State(coords, inds=inds)

        sampler.run_mcmc(
            state,
            cfg.nsteps,
            burn=cfg.nburn,
            thin_by=cfg.thin_by,
            progress=progress,
        )

        tau_chain, inds_chain = unpack_cold_chain(sampler, self.theta_space.idx_tau)
        summary = summarize_posterior(
            self.likelihood,
            tau_chain,
            inds_chain,
            pi0=cfg.pi0,
        )

        return RKBTFitResult(
            config=cfg,
            theta_space=self.theta_space,
            eta=self.eta,
            K=self.K,
            likelihood=self.likelihood,
            prior=self.prior,
            sampler=sampler,
            tau_chain=tau_chain,
            inds_chain=inds_chain,
            summary=summary,
        )


def fit_rkbt(
    X: np.ndarray,
    K: np.ndarray,
    grid: np.ndarray,
    *,
    config: RKBTConfig | None = None,
    progress: bool = True,
) -> RKBTFitResult:
    """Convenience wrapper: fit the RKBT model.

    Parameters
    ----------
    X : (nsamples, ngrid) array
    K : (ngrid, ngrid) array
    grid : (ngrid,) array
    config : RKBTConfig, optional
    progress : bool
    """
    model = RKBTModel(X, K, grid, config=config)
    return model.fit(progress=progress)
