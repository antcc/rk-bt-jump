# rk-bt

Bayesian one-sample functional test for the mean of Gaussian process data,
using a reproducing kernel Hilbert space (RKHS) representation and reversible
jump MCMC (Eryn backend).

The mathematical model is described in
`Papers/bayesian-test-rkhs/notes.tex`.

## Model overview

Given an i.i.d. sample $X_1,\dots,X_n \sim \mathrm{GP}(\mu, K)$ observed on a
grid $\mathcal S = \{s_1,\dots,s_G\} \subset [0,1]$, we test

$$H_0\colon \mu \equiv 0 \qquad\text{vs.}\qquad H_1\colon \mu \not\equiv 0,$$

under the finite-dimensional RKHS parametrization

$$\mu(\cdot)=\sum_{j=1}^{p}\beta_j K(t_j,\cdot),\qquad
\Pi(b\mid p)=\pi_0\,\delta_0 + (1-\pi_0)\,\mathcal N(0,\eta^2 I_p).$$

The coefficient vector $b = (\beta_1,\dots,\beta_p)'$ is analytically
marginalized out, yielding a closed-form marginal likelihood $m(X_{1:n}|p,\tau)$
(see §1 of `notes.tex`, eq. 8).  RJMCMC then targets the posterior of
$(p,\tau)$ only, and posterior samples for $b$ (and hence $\mu$) are recovered
afterwards via the conditional mixture formula.

An ordering constraint $t_1 < \cdots < t_p$ is imposed to avoid label
switching. In the code this is handled by treating active time-point indices as
a **set** (sorted before every cache look-up/computation) and rejecting
duplicate grid indices via the prior.

### Unknown covariance

When $K$ is unknown, a plug-in approach is used: estimate $\hat K$ from the
data (by default via the Oracle Approximating Shrinkage (OAS) estimator from
`scikit-learn`) and run the model as if $K = \hat K$.

## Project structure

```
rkbt/
  __init__.py        – public API re-exports
  parameters.py      – ThetaSpace: grid helpers and tau-to-grid indexing
  kernels.py         – kernel functions (squared-exponential, Brownian, OU)
  prior.py           – TauRJPrior: joint prior on (p, tau), min-distance check
  likelihood.py      – RKBTMarginalLikelihood: log m(X|p,tau) with NumPy/Numba engines
  moves.py           – TauGroupStretchMove + TauRJMove (birth/death RJ)
  sampler.py         – RKBTModel, RKBTConfig, fit_rkbt() high-level API
  postprocess.py     – posterior summaries, posterior draws for (b, mu),
                       simultaneous credible bands, ROPE
  covariance.py      – SampleCovEstimator (OAS / raw sample covariance)
  utils.py           – make_K_grid, make_mu_grid

reference_methods.py – frequentist and Bayesian competitor tests:
                       point Bonferroni t-test, global L² test,
                       L² FPCA test, F-type test, Bayesian FPCA g-prior BF
experiments.py       – reproducible experiments (synthetic + real datasets)
test_reference.py    – quick smoke-test for reference methods
demo_rkbt.ipynb      – interactive demo notebook
```

## Minimal usage

```python
import numpy as np
from rkbt import (
    RKBTConfig,
    SampleCovEstimator,
    fit_rkbt,
    make_K_grid,
    squared_exponential_kernel,
    simultaneous_credible_band,
    compute_rope,
)

# Data ------------------------------------------------------------------
grid = np.linspace(0.0, 1.0, 100)
X = ...  # shape (nsamples, len(grid))

# Known covariance -------------------------------------------------------
kernel_fn = lambda a, b: squared_exponential_kernel(
    a, b, length_scale=0.2, variance=1.0,
)
K = make_K_grid(kernel_fn, grid)

# Unknown covariance (plug-in) -------------------------------------------
# K = SampleCovEstimator(shrinkage=True).estimate(X)

# Configure and fit -------------------------------------------------------
cfg = RKBTConfig(
    pi0=0.5,
    eta=None,              # auto-scaled from data variance
    lambda_p=2.5,          # Poisson prior on p (None = flat)
    min_dist_tau=0,        # 0 = distinct grid indices
    likelihood_engine="auto",  # numpy | numba | auto
    nwalkers=32,
    ntemps=2,
    nsteps=1000,
    nburn=500,
    nleaves_min=1,
    nleaves_max=5,
    seed=42,
)

result = fit_rkbt(X, K, grid, config=cfg, progress=True)

# Summaries ---------------------------------------------------------------
print("P(H0|X) :", result.summary.posterior_null_prob)
print("log B10 :", result.summary.log_bf10)

draws = result.draw_posterior(ndraws=5000)
lower, upper, c = simultaneous_credible_band(
    draws["mu_draws"], result.summary.posterior_mean_mu, alpha=0.05,
)
rope = compute_rope(draws["mu_draws"], epsilon=0.01)
```

## Output

`fit_rkbt` returns an `RKBTFitResult` with:

| Attribute    | Description |
|---|---|
| `summary.posterior_null_prob` | Rao-Blackwell estimate of $\Pi_n(H_0 \mid X)$ |
| `summary.log_bf01` / `log_bf10` | log Bayes factors |
| `summary.mcse_naive` | naive MCSE of the posterior null probability |
| `summary.p_samples` | posterior draws of the dimension $p$ |
| `summary.posterior_mean_mu` | posterior mean function on the grid |
| `tau_chain`, `inds_chain` | raw MCMC chain for $(p,\tau)$ |
| `draw_posterior(...)` | draw posterior samples for $b$ and $\mu$ |

## Experiments

```bash
python experiments.py \
  --dataset synthetic_sqexp \
  --n-reps 20 \
  --n-samples 50 \
  --n-grid 100 \
  --nwalkers 32 \
  --ntemps 2 \
  --nsteps 1500 \
  --nburn 500 \
  --nleaves-min 1 \
  --nleaves-max 5 \
  --likelihood-engine auto \
  --n-boot-competitors 5000 \
  --save-results \
  --output-dir results/run1
```

Available datasets: `synthetic_sqexp`, `synthetic_brownian`, `synthetic_ou`,
`real_weather`.

Each experiment runs:

| Method | Type | Key output |
|---|---|---|
| **RKBT** (ours) | Bayesian RJMCMC | $\Pi_n(H_0\mid X)$, $\log B_{10}$ |
| Bonferroni $t$-test | Frequentist | $p$-value |
| Global $L^2$ test | Frequentist | $p$-value |
| $L^2$ FPCA test | Frequentist | $p$-value |
| $F$-type test | Frequentist | $p$-value |
| Bayesian FPCA $g$-prior | Bayesian | $\log B_{10}$ |

Results are saved to CSV files (details + summary with Type I error, power,
AUC, and runtimes).

## Likelihood engines

The marginal likelihood evaluation supports two back-ends, selectable via
`RKBTConfig.likelihood_engine`:

| Engine | Description |
|---|---|
| `"numpy"` | Pure NumPy with `np.linalg.cholesky`. Caches full conditional stats (needed for post-processing). |
| `"numba"` | JIT-compiled Cholesky + forward substitution. Caches only the scalar log-marginal. |
| `"auto"` | Use Numba if available, else NumPy. |

Both engines compute mathematically identical quantities; see the derivation of
$\log m(X_{1:n}\mid p,\tau)$ in `notes.tex`, eq. (8).