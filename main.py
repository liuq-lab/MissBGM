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
    map_imputed = model.x_map_imputed_

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

    summary = {
        "missingness_rate": missing_rate,
        "map_rmse": rmse_on_missing_entries(data["x_full"], map_imputed, data["mask"]),
        "mcmc_rmse": rmse_on_missing_entries(data["x_full"], mcmc_imputed, data["mask"]),
    }
    if "rmse_missing_only" in model.training_history_[0]:
        best_entry = min(model.training_history_, key=lambda x: x["rmse_missing_only"])
        summary["best_epoch"] = best_entry["epoch"]
        summary["best_map_rmse"] = best_entry["rmse_missing_only"]
    print(summary)
    return summary


def run_real_experiment(params: dict, missing_rate: float = 0.2, seed: int = 123, force: bool = False) -> dict:
    dataset_name = resolve_dataset_name(params["dataset"])
    benchmark = prepare_real_benchmark_data(dataset_name, missing_rate=missing_rate, seed=seed, force=force)

    x_full_norm = benchmark["x_full_norm"]
    x_obs_norm = benchmark["x_obs_norm"]
    mask = benchmark["mask"]

    params["x_dim"] = x_obs_norm.shape[1]
    model = MissBGM(params, random_seed=42)
    model.fit(data=x_obs_norm, mask=mask, x_true=x_full_norm, verbose=1)
    map_imputed = model.x_map_imputed_

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

    summary = {
        "dataset": dataset_name,
        "missing_rate": missing_rate,
        "mcmc_rmse": rmse_on_missing_entries(x_full_norm, mcmc_imputed, mask),
        "map_rmse": rmse_on_missing_entries(x_full_norm, map_imputed, mask),
    }
    if "rmse_missing_only" in model.training_history_[0]:
        best_entry = min(model.training_history_, key=lambda x: x["rmse_missing_only"])
        summary["best_epoch"] = best_entry["epoch"]
        summary["best_map_rmse"] = best_entry["rmse_missing_only"]
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="the path to config file")
    parser.add_argument("--force", action="store_true", help="rerun even if cached outputs exist")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    if params["dataset"] == "Sim_MNAR_oracle":
        run_synthetic_experiment(params)
    else:
        run_real_experiment(params, force=args.force)
