"""Competitor one-sample functional tests (frequentist and Bayesian)."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.stats import chi2, ttest_1samp
from scipy.stats import f as f_dist
from utils import SampleCovEstimator


@dataclass
class TestResult:
    method: str
    statistic: float
    pvalue: float
    reject: bool
    runtime_sec: float
    score: float | None = None
    posterior_null_prob: float | None = None
    log_bf10: float | None = None
    fpca_components: int | None = None


def _pca(
    K: np.ndarray,
    *,
    var_explained: float = 0.9,
    max_components: int | None = None,
    min_components: int = 1,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    ngrid = K.shape[0]
    if max_components is None:
        max_components = ngrid
    elif max_components > ngrid or max_components < 1:
        raise ValueError("`max_components` must be in [1, ngrid].")

    eigvals, eigvecs = np.linalg.eigh(K)
    # eigh returns ascending order; reverse to get descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals_pos = eigvals[eigvals > eps]
    cum = np.cumsum(eigvals_pos) / np.sum(eigvals_pos)
    p_auto = int(np.searchsorted(cum, var_explained) + 1)
    p = int(
        np.clip(
            p_auto,
            min_components,
            min(max_components, eigvals_pos.size),
        )
    )

    # The eigenvectors are normalized in Euclidean norm, but
    # their sign can vary among calls to this function.

    return eigvals[:p], eigvecs[:, :p]


def point_bonferroni_t_test(X: np.ndarray, *, alpha: float = 0.05) -> TestResult:
    """[Zhang2013, 4.5.1]"""
    t0 = perf_counter()
    stats, pvals = ttest_1samp(X, popmean=0.0, axis=0)
    p_global = np.minimum(1.0, np.nanmin(pvals) * X.shape[1])
    stat_global = np.nanmax(np.abs(stats))
    runtime = perf_counter() - t0
    return TestResult(
        method="point_bonferroni_t",
        statistic=stat_global,
        pvalue=p_global,
        reject=p_global < alpha,
        runtime_sec=runtime,
    )


def global_l2_test(
    X: np.ndarray,
    K: np.ndarray,
    *,
    alpha: float = 0.05,
    nboot: int = 1000,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """[Zhang2013, 4.5.2]"""
    t0 = perf_counter()
    n, ngrid = X.shape

    # Calculate statistic T_n
    mean_x = X.mean(axis=0)
    stat = n * np.sum(mean_x**2)

    if n > 50:
        # Approximate null distribution of T_n
        trK = np.trace(K)
        trK_2 = np.sum(K**2)  # trace(K @ K)
        gamma = trK_2 / trK
        d = trK**2 / trK_2

        p_val = 1.0 - chi2.cdf(stat / gamma, df=d)
    else:  # nonparametric bootstrap
        if rng is None:
            rng = np.random.default_rng()
        X_boot = rng.choice(X, size=(nboot, n), replace=True)
        boot_means = X_boot.mean(axis=1) - mean_x  # (nboot, ngrid)
        stat_boot = n * np.sum(boot_means**2, axis=1)  # (nboot,)
        # Add 1 to include 'stat' itself and avoid pvalue of 0
        p_val = (1.0 + np.sum(stat_boot >= stat)) / (nboot + 1)

    runtime = perf_counter() - t0
    return TestResult(
        method="global_l2",
        statistic=stat,
        pvalue=p_val,
        reject=p_val < alpha,
        runtime_sec=runtime,
    )


def global_l2_fpca_test(
    X: np.ndarray,
    K: np.ndarray,
    *,
    alpha: float = 0.05,
    var_explained: float = 0.9,
    max_components: int | None = None,
    eps: float = 1e-10,
    nboot: int = 1000,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """[Zhang2013, 4.5.2] applied to FPCA scores."""
    t0 = perf_counter()
    n, _ = X.shape

    eigvals, eigvecs = _pca(
        K, var_explained=var_explained, max_components=max_components, eps=eps
    )
    if eigvals.size == 0:
        runtime = perf_counter() - t0
        return TestResult(
            method="global_l2_fpca",
            statistic=0.0,
            pvalue=1.0,
            reject=False,
            runtime_sec=runtime,
        )
    mean_x = X.mean(axis=0)
    mean_scores = mean_x @ eigvecs
    stat = n * np.sum((mean_scores**2) / eigvals)

    if n > 50:
        p_val = 1.0 - chi2.cdf(stat, df=eigvals.size)
    else:  # nonparametric bootstrap
        if rng is None:
            rng = np.random.default_rng()
        X_boot = rng.choice(X, size=(nboot, n), replace=True)

        # Full plug-in bootstrap: re-estimate covariance and FPCA
        # basis in each replicate (slower but allows the retained
        # dimension and eigenpairs to vary).
        stat_boot = np.empty(nboot)
        estimator = SampleCovEstimator()
        for b in range(nboot):
            cov_b = estimator.estimate(X_boot[b])
            eigvals_b, eigvecs_b = _pca(
                cov_b,
                var_explained=var_explained,
                max_components=max_components,
                eps=eps,
            )
            if eigvals_b.size == 0:
                stat_boot[b] = 0.0
                continue
            mean_scores_b = (X_boot[b].mean(axis=0) - mean_x) @ eigvecs_b
            stat_boot[b] = n * np.sum((mean_scores_b**2) / eigvals_b)

        # Add 1 to include 'stat' itself and avoid pvalue of 0
        p_val = (1.0 + np.sum(stat_boot >= stat)) / (nboot + 1)

    runtime = perf_counter() - t0
    return TestResult(
        method="global_l2_fpca",
        statistic=stat,
        pvalue=p_val,
        reject=p_val < alpha,
        runtime_sec=runtime,
        fpca_components=eigvals.size,
    )


def global_f_test(
    X: np.ndarray,
    K: np.ndarray,
    *,
    alpha: float = 0.05,
    nboot: int = 1000,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """F-type test for one-sample functional data [Zhang2013, 4.5.3].

    Statistic: F_n = n * ||X_bar||^2 / tr(K_hat).
    Null distribution approximated by F(f1, f2) with Satterthwaite d.f.:
        f1 = tr(K_hat)^2 / tr(K_hat^2),  f2 = (n-1) * f1.
    """
    t0 = perf_counter()
    n, ngrid = X.shape

    mean_x = X.mean(axis=0)
    trK = np.trace(K)

    if trK <= 0:
        runtime = perf_counter() - t0
        return TestResult(
            method="global_f",
            statistic=0.0,
            pvalue=1.0,
            reject=False,
            runtime_sec=runtime,
        )

    stat = n * np.sum(mean_x**2) / trK

    if n > 50:
        trK2 = np.sum(K**2)  # tr(K @ K)
        if trK2 <= 0:
            p_val = 1.0
        else:
            f1 = trK**2 / trK2
            f2 = (n - 1) * f1
            p_val = 1.0 - f_dist.cdf(stat, dfn=f1, dfd=f2)
    else:  # nonparametric bootstrap
        if rng is None:
            rng = np.random.default_rng()
        X_boot = rng.choice(X, size=(nboot, n), replace=True)
        boot_means = X_boot.mean(axis=1) - mean_x  # (nboot, ngrid)
        mean_sq = np.sum(boot_means**2, axis=1)  # (nboot,)
        # OAS shrinkage preserves tr(cov), so tr(cov_OAS) = tr(cov_emp).
        # Vectorize: tr(cov_emp) = ||X_c||_F^2 / (n-1)
        Xc_boot = X_boot - X_boot.mean(axis=1, keepdims=True)
        trK_boot = np.sum(Xc_boot**2, axis=(1, 2)) / (n - 1)
        stat_boot = np.where(trK_boot > 0, n * mean_sq / trK_boot, 0.0)
        p_val = (1.0 + np.sum(stat_boot >= stat)) / (nboot + 1)

    runtime = perf_counter() - t0
    return TestResult(
        method="global_f",
        statistic=stat,
        pvalue=p_val,
        reject=p_val < alpha,
        runtime_sec=runtime,
    )


def bayes_fpca_test(
    X: np.ndarray,
    K: np.ndarray,
    *,
    g: float | None = None,
    g_min: float = 1e-3,
    g_max: float = 1e3,
    bf10_threshold: float = 3.0,
    prior_odds_10: float = 1.0,
    var_explained: float = 0.9,
    max_components: int | None = None,
    eps: float = 1e-10,
) -> TestResult:
    """Bayesian one-sample test via Bayes factor on FPCA scores."""
    t0 = perf_counter()
    if g_min <= 0 or g_max <= 0 or g_min > g_max:
        raise ValueError("Require 0 < g_min <= g_max.")
    if (g is not None) and ((not np.isfinite(g)) or g <= 0):
        raise ValueError("`g` must be > 0 when provided.")
    if (not np.isfinite(bf10_threshold)) or bf10_threshold <= 0:
        raise ValueError("`bf10_threshold` must be > 0.")
    if (not np.isfinite(prior_odds_10)) or prior_odds_10 <= 0:
        raise ValueError("`prior_odds_10` must be > 0.")

    X = np.asarray(X, dtype=float)
    n, _ = X.shape

    eigvals, eigvecs = _pca(
        K,
        var_explained=var_explained,
        max_components=max_components,
        eps=eps,
    )
    p = eigvals.size
    if p == 0:
        runtime = perf_counter() - t0
        return TestResult(
            method="bayes_fpca",
            statistic=-np.inf,
            pvalue=np.nan,
            reject=False,
            runtime_sec=runtime,
            score=-np.inf,
            posterior_null_prob=1.0,
            log_bf10=-np.inf,
        )

    # Q = Mahalanobis form in score space
    mean_scores = X.mean(axis=0) @ eigvecs
    quad = np.sum((mean_scores**2) / eigvals)

    if g is None:
        hat_g = ((n * quad) / p - 1.0) / n
        g = np.clip(hat_g, g_min, g_max)

    log_bf10 = -0.5 * p * np.log1p(n * g) + 0.5 * ((n * n * g) / (1.0 + n * g)) * quad
    if np.isnan(log_bf10) or np.isposinf(log_bf10):
        log_bf10 = np.inf

    # Posterior P(H1|data) via posterior odds
    log_post_odds_10 = np.log(prior_odds_10) + log_bf10
    if log_post_odds_10 >= 0:
        post_h1 = 1.0 / (1.0 + np.exp(-log_post_odds_10))
    else:
        e = np.exp(log_post_odds_10)
        post_h1 = e / (1.0 + e)
    post_h1 = np.clip(post_h1, eps, 1.0 - eps)

    reject = bool(log_bf10 > np.log(bf10_threshold))

    runtime = perf_counter() - t0
    return TestResult(
        method="bayes_fpca",
        statistic=log_bf10,
        pvalue=np.nan,
        reject=reject,
        runtime_sec=runtime,
        score=log_bf10,
        posterior_null_prob=1.0 - post_h1,
        log_bf10=log_bf10,
        fpca_components=p,
    )
