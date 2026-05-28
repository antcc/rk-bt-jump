#!/usr/bin/env python3
"""Experiments for RKBT and competitor methods."""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from reference_methods import (
    bayes_fpca_test,
    global_f_test,
    global_l2_fpca_test,
    global_l2_test,
    point_bonferroni_t_test,
)
from rkbt import (
    RKBTConfig,
    SampleCovEstimator,
    brownian_kernel,
    fit_rkbt,
    make_K_grid,
    make_mu_grid,
    ornstein_uhlenbeck_kernel,
    squared_exponential_kernel,
)

try:
    from skfda.preprocessing.smoothing import BasisSmoother
    from skfda.representation.basis import FourierBasis
    from skfda.representation.grid import FDataGrid

    SKFDA_AVAILABLE = True
except Exception:
    SKFDA_AVAILABLE = False
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Scenario:
    name: str
    is_alternative: bool
    atoms_idx: tuple[int, ...] | None = None
    coeffs: tuple[float, ...] | None = None


def _nanmean_if_any(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    return np.nanmean(arr)


def make_synthetic_kernel_grid(
    dataset: str,
    grid: np.ndarray,
    *,
    length_scale: float,
    variance: float,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Build the true covariance matrix on the grid for synthetic experiments."""
    if dataset == "synthetic_sqexp":

        def kernel_fn(t, s):
            return squared_exponential_kernel(
                t,
                s,
                length_scale=length_scale,
                variance=variance,
            )
    elif dataset == "synthetic_brownian":

        def kernel_fn(t, s):
            return brownian_kernel(t, s, variance=variance)
    elif dataset == "synthetic_ou":

        def kernel_fn(t, s):
            return ornstein_uhlenbeck_kernel(
                t,
                s,
                length_scale=length_scale,
                variance=variance,
            )
    else:
        raise ValueError(f"Unsupported synthetic dataset {dataset!r}.")

    return make_K_grid(kernel_fn, grid, jitter=jitter)


def simulate_dataset(
    scenario: Scenario,
    K_true: np.ndarray,
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    nsamples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario.atoms_idx:
        mu = make_mu_grid(
            K_true,
            atoms_idx=scenario.atoms_idx,
            coeffs=scenario.coeffs,
        )
        # mu = 0.3 * np.sin(2 * np.pi * grid)
    else:
        mu = np.zeros(K_true.shape[0], dtype=float)

    X = rng.multivariate_normal(mu, K_true, size=nsamples)
    if noise_std > 0:
        X = X + rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    return X, mu


def _load_real_weather_period_data(
    period_defs,
    *,
    monthly: bool = True,
    n_limit: int = 200,
    smoothing: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:

    from datetime import datetime

    import meteostat as met

    print("Loading real weather data...")

    start_dt = datetime(period_defs[0][0], 1, 1)
    end_dt = datetime(period_defs[-1][-1], 12, 31)

    stations = met.stations.query(
        f"SELECT id FROM stations WHERE country='AU' LIMIT {int(n_limit)}"
    )
    stations_data = []
    for sid in stations["id"].astype(str).tolist():
        if monthly:
            raw_data = met.monthly(sid, start_dt, end_dt).fetch()
        else:
            raw_data = met.daily(sid, start_dt, end_dt).fetch()
        if raw_data is None or len(raw_data) == 0:
            continue

        stations_data.append(raw_data)

    curves_by_period = [[] for _ in range(len(period_defs))]
    ngrid = 12 if monthly else 365
    grid_points = np.linspace(0.0, 1.0, ngrid)
    smoother = None
    if smoothing and SKFDA_AVAILABLE:
        n_basis = ngrid // 2
        smoother = BasisSmoother(
            basis=FourierBasis(domain_range=(0.0, 1.0), n_basis=n_basis)
        )

    for station in stations_data:
        temp_col = "tavg" if "tavg" in station.columns else "temp"
        if temp_col not in station.columns:
            continue

        series = station[temp_col].dropna()
        if len(series) == 0:
            continue

        if not monthly:  # remove Feb 29th
            series = series[~((series.index.month == 2) & (series.index.day == 29))]

        period_curves: list[np.ndarray] = []
        valid_station = True

        for y0, y1 in period_defs:
            block = series[(series.index.year >= y0) & (series.index.year <= y1)]
            if len(block) == 0:
                valid_station = False
                break

            groupby = block.index.month if monthly else block.index.dayofyear

            curve = (
                block.groupby(groupby)
                .mean()
                .reindex(range(1, ngrid + 1), fill_value=np.nan)
                .to_numpy(dtype=float)
            )
            if np.isnan(curve).any():
                valid_station = False
                break

            if smoother is not None:
                fd = FDataGrid(
                    data_matrix=curve[np.newaxis, :], grid_points=grid_points
                )
                curve = np.asarray(smoother.fit_transform(fd).data_matrix).reshape(-1)
            period_curves.append(curve)

        if not valid_station:
            continue

        for idx_p in range(len(period_defs)):
            curves_by_period[idx_p].append(period_curves[idx_p])

    n_stations = len(curves_by_period[0])
    if n_stations == 0:
        raise RuntimeError(
            "No station had complete daily data across all three periods."
        )

    period_arrays = [np.vstack(curves) for curves in curves_by_period]

    period_labels = [f"real_weather_{i}" for i in range(1, len(period_arrays) + 1)]
    return grid_points, period_arrays, period_labels


def load_real_dataset(
    dataset: str,
    *,
    smoothing: bool = False,
    load_from_file: bool = True,
) -> tuple[np.ndarray, list[tuple[np.ndarray, Scenario]]]:
    if dataset != "real_weather":
        raise ValueError(f"Unsupported real dataset {dataset!r}.")

    if load_from_file:
        filename = "real_weather_smoothed.npz" if smoothing else "real_weather.npz"
        data = np.load("data/" + filename)
        grid = data["grid"]
        period_arrays = data["period_arrays"]
        period_labels = data["period_labels"]
    else:
        period_defs = [(1990, 1999), (2000, 2009), (2010, 2019)]
        grid, period_arrays, period_labels = _load_real_weather_period_data(
            period_defs,
            monthly=True,
            n_limit=200,
            smoothing=smoothing,
        )

        # Save arrays to file
        # np.savez_compressed(
        #     "data/real_weather_smoothed.npz" if smoothing else "data/real_weather.npz",
        #     grid=grid,
        #     period_arrays=period_arrays,
        #     period_labels=period_labels,
        # )

    n_periods = len(period_arrays)
    pairings = list(itertools.combinations(range(n_periods), 2))
    experiment_names = [f"real_weather_{i}" for i in range(1, n_periods + 1)]
    cases: list[tuple[np.ndarray, Scenario]] = []
    for exp_name, (i, j) in zip(experiment_names, pairings):
        diff_curves = period_arrays[j] - period_arrays[i]
        scenario = Scenario(
            name=exp_name,
            is_alternative=True,
            atoms_idx=(),
            coeffs=(),
        )
        cases.append((diff_curves, scenario))

    return grid, cases


def record_result(
    rows: list[dict],
    *,
    dataset: str,
    scenario: Scenario,
    replicate: int,
    method: str,
    reject: bool,
    score: float,
    runtime_sec: float,
    pvalue: float | None = None,
    posterior_null_prob: float | None = None,
    log_bf10: float | None = None,
    posterior_mean_p: float | None = None,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "scenario": scenario.name,
            "is_alternative": int(scenario.is_alternative),
            "replicate": replicate,
            "method": method,
            "reject": int(reject),
            "score": score,
            "runtime_sec": runtime_sec,
            "pvalue": np.nan if pvalue is None else pvalue,
            "posterior_null_prob": np.nan
            if posterior_null_prob is None
            else posterior_null_prob,
            "log_bf10": np.nan if log_bf10 is None else log_bf10,
            "posterior_mean_p": np.nan
            if posterior_mean_p is None
            else posterior_mean_p,
        }
    )


def summarize_rows(rows: list[dict]) -> list[dict]:
    datasets = sorted({r["dataset"] for r in rows})
    methods = sorted({r["method"] for r in rows})

    out = []
    for dataset in datasets:
        for method in methods:
            rr = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not rr:
                continue

            if dataset.startswith("real"):
                scenario_groups = sorted({str(r["scenario"]) for r in rr})
            else:
                scenario_groups = ["all"]

            for scenario_name in scenario_groups:
                rr_sc = (
                    [r for r in rr if str(r["scenario"]) == scenario_name]
                    if scenario_name != "all"
                    else rr
                )
                h0 = [r for r in rr_sc if r["is_alternative"] == 0]
                h1 = [r for r in rr_sc if r["is_alternative"] == 1]
                labels = np.asarray([r["is_alternative"] for r in rr_sc], dtype=int)
                scores = np.asarray([r["score"] for r in rr_sc], dtype=float)

                type1 = np.mean([r["reject"] for r in h0]) if h0 else np.nan
                power = np.mean([r["reject"] for r in h1]) if h1 else np.nan
                auc = roc_auc_score(labels, scores) if h0 and h1 else np.nan

                entry = {
                    "dataset": dataset,
                    "scenario": scenario_name,
                    "method": method,
                    "type1_error": type1,
                    "power": power,
                    "auc": auc,
                    "mean_runtime_sec": _nanmean_if_any(
                        [r["runtime_sec"] for r in rr_sc]
                    ),
                    "mean_score_h0": _nanmean_if_any([r["score"] for r in h0]),
                    "mean_score_h1": _nanmean_if_any([r["score"] for r in h1]),
                }

                if method.startswith("rkbt") or method.startswith("bayes_"):
                    entry["mean_posterior_null_h0"] = _nanmean_if_any(
                        [r["posterior_null_prob"] for r in h0]
                    )
                    entry["mean_posterior_null_h1"] = _nanmean_if_any(
                        [r["posterior_null_prob"] for r in h1]
                    )
                    entry["mean_log_bf10_h0"] = _nanmean_if_any(
                        [r["log_bf10"] for r in h0]
                    )
                    entry["mean_log_bf10_h1"] = _nanmean_if_any(
                        [r["log_bf10"] for r in h1]
                    )
                    if method.startswith("rkbt"):
                        entry["mean_p_h0"] = _nanmean_if_any(
                            [r["posterior_mean_p"] for r in h0]
                        )
                        entry["mean_p_h1"] = _nanmean_if_any(
                            [r["posterior_mean_p"] for r in h1]
                        )

                out.append(entry)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(summary_rows: list[dict]) -> None:
    print("\nSummary metrics (Type I / Power / AUC / Runtime):")
    print("-" * 90)
    print(
        f"{'dataset':<20} {'method':<28} {'type_I':>8} {'power':>8} {'auc':>8} {'runtime_s':>12}"
    )
    print("-" * 90)
    for r in summary_rows:
        print(
            f"{r['dataset']:<20} {r['method']:<28} "
            f"{r['type1_error']:>8.3f} {r['power']:>8.3f} {r['auc']:>8.3f} "
            f"{r['mean_runtime_sec']:>12.3f}"
        )
    print("-" * 90)


def print_bayesian_details_table(detail_rows: list[dict]) -> None:
    bayes_rows = [
        r
        for r in detail_rows
        if str(r.get("method", "")).startswith("rkbt")
        or str(r.get("method", "")).startswith("bayes_")
    ]
    if not bayes_rows:
        return

    bayes_rows = sorted(
        bayes_rows,
        key=lambda r: (
            str(r.get("dataset", "")),
            str(r.get("scenario", "")),
            str(r.get("method", "")),
            int(r.get("replicate", 0)),
        ),
    )

    print("\nBayesian details (posterior diagnostics by replicate):")
    print("-" * 90)
    print(
        f"{'dataset':<16} {'scenario':<16} {'rep':<7} {'method':<16} "
        f"{'post_H0':>10} {'log_bf10':>10}"
    )
    print("-" * 90)
    for r in bayes_rows:
        print(
            f"{str(r.get('dataset', '')):<16} {str(r.get('scenario', '')):<16} "
            f"{int(r.get('replicate', 0) + 1):<7d} "
            f"{str(r.get('method', '')):<16} "
            f"{r.get('posterior_null_prob', np.nan):>10.3f} "
            f"{r.get('log_bf10', np.nan):>10.3f} "
        )
    print("-" * 90)


def run_frequentist_competitors(
    X: np.ndarray,
    K: np.ndarray,
    *,
    alpha: float,
    nboot: int,
    seed: int,
) -> list[tuple[str, bool, float, float, float]]:
    rng = np.random.default_rng(seed)
    tests = [
        point_bonferroni_t_test(X, alpha=alpha),
        global_l2_test(X, K, alpha=alpha, nboot=nboot, rng=rng),
        global_l2_fpca_test(X, K, alpha=alpha, nboot=nboot, rng=rng),
        global_f_test(X, K, alpha=alpha, nboot=nboot, rng=rng),
    ]
    return [
        (
            t.method,
            bool(t.reject),
            (1.0 - t.pvalue) if t.score is None else t.score,
            t.runtime_sec,
            t.pvalue,
        )
        for t in tests
    ]


def run_bayesian_competitors(
    X: np.ndarray,
    K: np.ndarray,
    *,
    bf10_threshold: float,
    pi0: float,
    g_prior: float | None,
    g_min: float,
    g_max: float,
) -> list[tuple[str, bool, float, float, float, float | None, float | None]]:
    """Return rows for Bayesian competitor (FPCA-based g-prior BF)."""
    res = bayes_fpca_test(
        X,
        K,
        g=g_prior,
        g_min=g_min,
        g_max=g_max,
        bf10_threshold=bf10_threshold,
        prior_odds_10=(1.0 - pi0) / pi0,
    )
    return [
        (
            res.method,
            bool(res.reject),
            res.score,
            res.runtime_sec,
            res.pvalue,
            res.posterior_null_prob,
            res.log_bf10,
        )
    ]


def run_experiments(args) -> tuple[list[dict], list[dict]]:
    rng_master = np.random.default_rng(args.seed)
    is_synthetic = args.dataset.startswith("synthetic_")

    if is_synthetic:
        grid = np.linspace(0.0, 1.0, args.ngrid)
        K_true = make_synthetic_kernel_grid(
            args.dataset,
            grid,
            length_scale=args.true_length_scale,
            variance=args.true_variance,
            jitter=args.kernel_jitter,
        )

        idx_a = int(round(0.15 * (args.ngrid - 1)))
        idx_b = int(round(0.40 * (args.ngrid - 1)))
        idx_c = int(round(0.76 * (args.ngrid - 1)))
        h1_coeffs = (0.2, -0.3, 0.5)
        scenarios = [
            Scenario("H0_true", False, None, None),
            Scenario("H1_true", True, (idx_a, idx_b, idx_c), h1_coeffs),
        ]

    rkbt_cfg = RKBTConfig(
        pi0=args.pi0,
        eta=args.eta,
        eta_scaling_factor=args.eta_scaling_factor,
        lambda_p=args.lambda_p,
        min_dist_tau=args.min_dist_tau,
        likelihood_engine=args.likelihood_engine,
        nwalkers=args.nwalkers,
        ntemps=args.ntemps,
        nsteps=args.nsteps,
        nburn=args.nburn,
        thin_by=1,
        nleaves_min=args.nleaves_min,
        nleaves_max=args.nleaves_max,
        seed=args.seed,
    )

    cov_estimator = SampleCovEstimator(jitter=args.kernel_jitter)

    details: list[dict] = []
    nreps_eff = args.nreps if is_synthetic else 1
    if not is_synthetic and args.nreps != 1:
        print(
            f"Real dataset selected: forcing nreps=1 and ignoring --ngrid={args.ngrid} and --nsamples={args.nsamples}."
        )

    for rep in range(nreps_eff):
        rep_seed = int(rng_master.integers(0, 2**31 - 1))
        rng_rep = np.random.default_rng(rep_seed)

        if not is_synthetic:
            grid, real_cases = load_real_dataset(
                args.dataset,
                smoothing=args.smoothing_real,
            )
            scenarios = [scenario for _, scenario in real_cases]
            x_by_scenario = {scenario.name: X for X, scenario in real_cases}

            if rep == 0:
                print(
                    f"Using ngrid={len(grid)} and nsamples={real_cases[0][0].shape[0]} for real datasets."
                )

        if args.verbose > 0:
            print(f"\nReplicate {rep + 1}/{nreps_eff} (seed={rep_seed})")

        for scenario in scenarios:
            dataset_name = args.dataset if is_synthetic else scenario.name
            if is_synthetic:
                X, _ = simulate_dataset(
                    scenario,
                    K_true,
                    grid,
                    rng_rep,
                    nsamples=args.nsamples,
                    noise_std=args.synthetic_noise_std,
                )
            else:
                X = x_by_scenario[scenario.name]

            # Estimate covariance from data (shrinkage is enabled by default)
            K_hat = cov_estimator.estimate(X)

            if not args.no_competitors:
                # --- Frequentist competitors ---
                comp_frequentist = run_frequentist_competitors(
                    X,
                    K_hat,
                    alpha=args.alpha,
                    nboot=args.nboot_competitors,
                    seed=rep_seed + 101,
                )

                # --- Bayesian competitor (FPCA g-prior) ---
                comp_bayes = run_bayesian_competitors(
                    X,
                    K_hat,
                    bf10_threshold=args.bayes_bf10_threshold,
                    pi0=args.pi0,
                    g_prior=args.bayes_g,
                    g_min=args.bayes_g_min,
                    g_max=args.bayes_g_max,
                )

                for method, reject, score, runtime_sec, pvalue in comp_frequentist:
                    record_result(
                        details,
                        dataset=dataset_name,
                        scenario=scenario,
                        replicate=rep,
                        method=method,
                        reject=reject,
                        score=score,
                        runtime_sec=runtime_sec,
                        pvalue=pvalue,
                    )

                for (
                    method,
                    reject,
                    score,
                    runtime_sec,
                    pvalue,
                    posterior_null_prob,
                    log_bf10_comp,
                ) in comp_bayes:
                    record_result(
                        details,
                        dataset=dataset_name,
                        scenario=scenario,
                        replicate=rep,
                        method=method,
                        reject=reject,
                        score=score,
                        runtime_sec=runtime_sec,
                        pvalue=pvalue,
                        posterior_null_prob=posterior_null_prob,
                        log_bf10=log_bf10_comp,
                    )

            # --- RKBT (our method) ---
            t0 = perf_counter()
            fit_result = fit_rkbt(
                X,
                K_hat,
                grid,
                config=rkbt_cfg,
                progress=args.verbose > 1,
            )
            t_rkbt = perf_counter() - t0
            post_null = fit_result.summary.posterior_null_prob
            log_bf10 = fit_result.summary.log_bf10
            record_result(
                details,
                dataset=dataset_name,
                scenario=scenario,
                replicate=rep,
                method="rkbt",
                reject=log_bf10 > np.log(args.bayes_bf10_threshold),
                score=log_bf10,
                runtime_sec=t_rkbt,
                posterior_null_prob=post_null,
                log_bf10=log_bf10,
                posterior_mean_p=np.mean(fit_result.summary.p_samples),
            )

    summary = summarize_rows(details)
    return details, summary


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "RKBT experiments (estimated covariance + reference methods)"
    )
    p.add_argument("--nreps", type=int, default=1)
    p.add_argument("--nsamples", type=int, default=30)
    p.add_argument("--ngrid", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument(
        "--dataset",
        type=str,
        default="synthetic_sqexp",
        choices=[
            "synthetic_sqexp",
            "synthetic_brownian",
            "synthetic_ou",
            "real_weather",
        ],
    )

    # True data-generating kernel
    p.add_argument("--true-length-scale", type=float, default=0.2)
    p.add_argument("--true-variance", type=float, default=1.5)
    p.add_argument("--kernel-jitter", type=float, default=1e-10)
    p.add_argument("--synthetic-noise-std", type=float, default=0.0)

    # Real data parameters
    p.add_argument("--smoothing-real", action="store_true")

    # RKBT configuration
    p.add_argument("--pi0", type=float, default=0.5)
    p.add_argument("--eta", type=float)  # Default is None
    p.add_argument("--eta-scaling-factor", type=float, default=0.25)
    p.add_argument("--lambda-p", type=float)  # Default is None
    p.add_argument("--min-dist-tau", type=int, default=0)
    p.add_argument(
        "--likelihood-engine", choices=["numpy", "numba", "auto"], default="auto"
    )
    p.add_argument("--nwalkers", type=int, default=32)
    p.add_argument("--ntemps", type=int, default=2)
    p.add_argument("--nsteps", type=int, default=500)
    p.add_argument("--nburn", type=int, default=1000)
    p.add_argument("--nleaves-min", type=int, default=1)
    p.add_argument("--nleaves-max", type=int, default=5)
    p.add_argument("--bayes-bf10-threshold", type=float, default=3.0)

    # Competitor methods controls.
    p.add_argument("--no-competitors", action="store_true")
    p.add_argument("--bayes-g", type=float)  # default is None
    p.add_argument("--bayes-g-min", type=float, default=1e-3)
    p.add_argument("--bayes-g-max", type=float, default=1e3)
    p.add_argument("--nboot-competitors", type=int, default=5000)

    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")
    return p


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    print("Running RKBT experiments with configuration:")
    print(
        f"dataset={args.dataset}, noise={args.synthetic_noise_std}, nreps={args.nreps}, nsamples={args.nsamples}, "
        f"ngrid={args.ngrid}, nwalkers={args.nwalkers}, ntemps={args.ntemps}, "
        f"nburn={args.nburn}, nsteps={args.nsteps}, nleaves=[{args.nleaves_min},{args.nleaves_max}], seed={args.seed}"
    )

    t0 = perf_counter()
    details, summary = run_experiments(args)
    elapsed = perf_counter() - t0

    print_summary_table(summary)
    print_bayesian_details_table(details)

    if args.save_results:
        output_dir = Path(args.output_dir)
        dataset_name = "".join(
            ch if (ch.isalnum() or ch in "-_") else "_" for ch in args.dataset
        )
        details_path = output_dir / f"{dataset_name}_details.csv"
        summary_path = output_dir / f"{dataset_name}_summary.csv"
        write_csv(details_path, details)
        write_csv(summary_path, summary)

        print("\nSaved:")
        print(f"  {details_path.as_posix()}")
        print(f"  {summary_path.as_posix()}")

    print(f"Total elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
