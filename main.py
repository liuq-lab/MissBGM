import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from missbgm.datasets import prepare_real_benchmark_data, resolve_dataset_name
from missbgm.datasets import simulate_mnar_oracle_data
from missbgm.models import MissBGM
from missbgm.utils import benchmark_mnar_imputers, rmse_on_missing_entries


def run_synthetic_experiment(params: dict) -> dict:
    missing_rate = params["missing_rate"]
    data = simulate_mnar_oracle_data(
        n_samples=params["n_samples"],
        x_dim=params["x_dim"],
        n_anchor=5,
        missing_rate=params["missing_rate"],
        alpha=0.05,
        seed=123,
    )

    model = MissBGM(params, random_seed=42)
    model.fit(data=data["x_obs"], mask=data["mask"], x_true=data["x_full"], verbose=1)
    mcmc_imputed, intervals = model.predict(
        data=data["x_obs"],
        mask=data["mask"],
        x_true=data["x_full"],
        alpha=0.05,
        n_mcmc=1000,
        burn_in=1000,
        step_size=0.1,
        num_leapfrog_steps=5,
        seed=42,
        verbose=1,
    )

    if model.last_prediction_ is None:
        raise RuntimeError("MissBGM.predict() did not populate prediction diagnostics.")
    map_imputed = model.last_prediction_["map_imputed"]

    summary = {
        "missingness_rate": missing_rate,
        "map_rmse": rmse_on_missing_entries(data["x_full"], map_imputed, data["mask"]),
        "mcmc_rmse": rmse_on_missing_entries(data["x_full"], mcmc_imputed, data["mask"]),
    }
    print(summary)
    return summary


def run_real_experiment(params: dict, missing_rate: float = 0.2, seed: int = 123, force: bool = False) -> dict:
    dataset_name = resolve_dataset_name(params["dataset"])
    benchmark = prepare_real_benchmark_data(dataset_name, missing_rate=missing_rate, seed=seed, force=force)

    x_full_norm = benchmark["x_full_norm"]
    x_obs_norm = benchmark["x_obs_norm"]
    mask = benchmark["mask"]

    model = MissBGM(params, random_seed=42)
    model.fit(data=x_obs_norm, mask=mask, x_true=x_full_norm, verbose=1)
    mcmc_imputed, intervals = model.predict(
        data=x_obs_norm,
        mask=mask,
        x_true=x_full_norm,
        alpha=0.05,
        n_mcmc=1000, 
        burn_in=1000, 
        step_size=0.1, 
        num_leapfrog_steps=5, 
        seed=42,
        verbose=1
    )
    if model.last_prediction_ is None:
        raise RuntimeError("MissBGM.predict() did not populate prediction diagnostics.")
    map_imputed = model.last_prediction_["map_imputed"]

    summary = {
        "dataset": dataset_name,
        "missing_rate": missing_rate,
        "mcmc_rmse": rmse_on_missing_entries(x_full_norm, mcmc_imputed, mask),
        "map_rmse": rmse_on_missing_entries(x_full_norm, map_imputed, mask),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="the path to config file")
    parser.add_argument("--beta", type=float, default=0.01, help="the beta parameter for the MissBGM model")
    parser.add_argument("--n_samples", type=int, default=5000, help="number of samples for Synthetic_MNAR (overrides config)")
    parser.add_argument("--missing_rate", type=float, default=0.2, help="missing rate for Synthetic_MNAR (overrides config)")
    parser.add_argument("--x_dim", type=int, default=50, help="feature dimensionality for Synthetic_MNAR (overrides config)")
    parser.add_argument("--force", action="store_true", help="rerun even if cached outputs exist")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    if params["dataset"] == "Synthetic_MNAR":
        run_synthetic_experiment(params)
    else:
        run_real_experiment(params, force=args.force)
