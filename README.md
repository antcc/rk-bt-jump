# rk-bt-jump

Bayesian one-sample functional test for the mean of a Gaussian process, using a reproducing kernel Hilbert space (RKHS) representation and reversible jump MCMC (RJMCMC) for posterior approximation.

## Model overview

Given an i.i.d. sample $X_1,\dots,X_n \sim \mathrm{GP}(\mu, K)$ with $\mu\in\mathcal H(K)$, where $\mathcal H(K)$ is the RKHS associated with the covariance function $K$, we test

$$H_0\colon \mu \equiv 0 \qquad\text{vs.}\qquad H_1\colon \mu \not\equiv 0,$$

under the finite-dimensional RKHS parametrization

$$\mu_\theta(\cdot)=\sum_{j=1}^{p}\beta_j K(t_j,\cdot), \quad \theta=(p,\tau, b).$$

We assume that $\pi_0\in(0,1)$ is the prior probability of $H_0$, and we put a spike-and-slab prior on the coefficient vector $b = (\beta_1,\dots,\beta_p)\in \mathbb{R}^p$, a continuous prior on the time instants $\tau=(t_1,\dots,t_p)\in [0,1]^p$, and a discrete prior on the number of components $p\in\mathbb{N}$. The parameter $b$ is analytically marginalized out, yielding a closed-form marginal likelihood $m(X_{1:n}|p,\tau)$. Samples from the corresponding posterior are obtained through RJMCMC via the [Eryn](https://github.com/mikekatz04/Eryn) library.

### Unknown covariance

When $K$ is unknown, a plug-in approach is used: estimate $\hat K$ from the data (by default via the Oracle Approximating Shrinkage (OAS) estimator from `scikit-learn`) and run the model as if $K = \hat K$.

## Project structure

*Python 3.14*

```
rkbt_jump/
  likelihood.py         – log m(X|p,tau) with NumPy/Numba engines
  moves.py              – birth/death RJMCMC moves
  parameters.py         – grid helpers and tau-to-grid indexing
  postprocess.py        – posterior summaries
  prior.py              – joint prior on (p, tau)
  sampler.py            – posterior sampling

utils/
  covariance.py         – covariance estimation
  grid.py               – discretization on the grid
  kernels.py            – kernel functions

reference_methods/
  reference_methods.py  – frequentist and Bayesian competitor tests

experiments.py          – reproducible experiments
demo_rkbt.ipynb         – interactive demo notebook with simulations
demo_real_weather.ipynb – interactive demo notebook with real data
```

## Minimal usage

```python
import numpy as np
from rkbt_jump import (
    RKBTConfig,
    fit_rkbt,
)
from utils import SampleCovEstimator

# Data
grid = np.linspace(0.0, 1.0, 100)
X = ...  # shape (nsamples, len(grid))

# Unknown covariance (plug-in)
K_hat = SampleCovEstimator(shrinkage=True).estimate(X)

# Configure and fit 
cfg = RKBTConfig(
    pi0=0.5,
    lambda_p=2.5,  # Poisson prior on p (None = uniform)
    likelihood_engine="auto",
    nwalkers=32,
    ntemps=2,
    nsteps=1000,
    nburn=500,
    nleaves_min=1,
    nleaves_max=5,
    seed=42,
)
result = fit_rkbt(X, K_hat, grid, config=cfg, progress=True)

# Summaries 
print("P(H0|X) :", result.summary.posterior_null_prob)
print("log BF_10 :", result.summary.log_bf10)
```

## Likelihood engine

The marginal likelihood evaluation supports two backends, selectable via `RKBTConfig.likelihood_engine`:

| Engine | Description |
|---|---|
| `"numpy"` | Pure NumPy with `np.linalg.cholesky`. Caches full conditional stats (needed for post-processing). |
| `"numba"` | JIT-compiled Cholesky + forward substitution. Caches only the scalar log-marginal. |
| `"auto"` | Use Numba if available, else NumPy. |

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
  --save-results \
  --output-dir results/
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
| Bayesian FPCA $g$-prior | Bayesian | $\Pi_n(H_0\mid X)$, $\log B_{10}$ |

Results are saved to CSV files with Type I error, power, AUC, and runtimes.

