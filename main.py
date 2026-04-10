import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from missbgm.datasets import prepare_real_benchmark_data, resolve_dataset_name
from missbgm.datasets import simulate_mnar_factor_data, simulate_mnar_oracle_data
from missbgm.models import MissBGM as BGM_MNAR
from missbgm.utils import benchmark_mnar_imputers, rmse_on_missing_entries


def configure_cpu_only():
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def run_synthetic_experiment(params: dict) -> dict:
    missing_rate = 0.2
    data = simulate_mnar_oracle_data(
        n_samples=params["n_samples"],
        x_dim=params["x_dim"],
        n_anchor=5,
        missing_rate=0.2,
        alpha=0.05,
        seed=123,
    )

    model = BGM_MNAR(params, random_seed=None)
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

    metrics = {
        "method_name": "missbgm",
        "missingness_rate": missing_rate,
        "map_rmse": rmse_on_missing_entries(data["x_full"], map_imputed, data["mask"]),
        "mcmc_rmse": rmse_on_missing_entries(data["x_full"], mcmc_imputed, data["mask"]),
        "save_dir": model.save_dir,
    }
    np.savez("intervals.npz", intervals=np.array(intervals, dtype=object))
    save_dir = Path(model.save_dir)

    baseline = benchmark_mnar_imputers(
        x_true=data["x_full"],
        x_obs=data["x_obs"],
        mask=data["mask"],
        params=params,
        missing_rate=missing_rate,
        seed=123,
        results_path=str(save_dir / "baseline_results.csv"),
    )[["method_name", "missingness_rate", "rmse"]].copy()
    bgm_rows = pd.DataFrame(
        [
            {
                "method_name": "missbgm_mcmc",
                "missingness_rate": missing_rate,
                "rmse": metrics["mcmc_rmse"],
            },
            {
                "method_name": "missbgm_map",
                "missingness_rate": missing_rate,
                "rmse": metrics["map_rmse"],
            },
        ]
    )
    comparison = pd.concat([bgm_rows, baseline], ignore_index=True)
    comparison = comparison.sort_values("rmse", kind="stable").reset_index(drop=True)
    comparison["delta_vs_missbgm_map"] = comparison["rmse"] - metrics["map_rmse"]
    comparison.to_csv(save_dir / f"comparison_missing_rate_{missing_rate:.1f}.csv", index=False)
    print(f"\nMissBGM comparison at missing rate {missing_rate:.1f}")
    print(comparison.to_string(index=False))
    print("\nMissBGM metrics")
    print(json.dumps(metrics, indent=2))
    return metrics


def run_real_experiment(params: dict, missing_rate: float = 0.2, seed: int = 123, force: bool = False) -> dict:
    dataset_name = resolve_dataset_name(params["dataset"])
    benchmark = prepare_real_benchmark_data(dataset_name, missing_rate=missing_rate, seed=seed, force=force)

    output_root = Path(params.get("output_dir", "."))
    result_dir = output_root / "results" / "Real_MNAR" / dataset_name / f"missing_nolinear_rate_{missing_rate:.1f}_seed_{seed}"
    summary_path = result_dir / "missbgm_summary.json"
    if summary_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps(summary, indent=2))
        #return summary

    x_full_norm = benchmark["x_full_norm"]
    x_obs_norm = benchmark["x_obs_norm"]
    mask = benchmark["mask"]

    model = BGM_MNAR(params, random_seed=42)
    model.fit(data=x_obs_norm, mask=mask, x_true=x_full_norm, verbose=1)

    training_history = pd.DataFrame(model.training_history_)
    result_dir.mkdir(parents=True, exist_ok=True)
    history_path = result_dir / "training_history.csv"
    training_history.to_csv(history_path, index=False)
    np.save(result_dir / "mask.npy", mask.astype(np.float32))
    np.save(result_dir / "x_full_norm.npy", x_full_norm.astype(np.float32))
    np.save(result_dir / "x_obs_norm.npy", x_obs_norm.astype(np.float32))

    metric_col = "rmse_missing_only"
    best_row = training_history.sort_values([metric_col, "epoch"], kind="stable").iloc[0]
    best_epoch = int(best_row["epoch"])

    summary = {
        "dataset": dataset_name,
        "missing_rate": missing_rate,
        "seed": seed,
        "num_rows": int(x_full_norm.shape[0]),
        "num_features": int(x_full_norm.shape[1]),
        "normalized_with_standard_scaler": True,
        "actual_missing_rate": float(benchmark["metadata"]["actual_missing_rate"]),
        "epochs": int(params["epochs"]),
        "best_epoch_metric": metric_col,
        "best_epoch": best_epoch,
        "lowest_rmse_normalized": float(best_row[metric_col]),
        "training_history_csv": str(history_path),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="the path to config file")
    parser.add_argument("--force", action="store_true", help="rerun even if cached outputs exist")
    args = parser.parse_args()

    configure_cpu_only()

    with open(args.config, "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    if params["dataset"] == "Synthetic_MNAR":
        run_synthetic_experiment(params)
    else:
        run_real_experiment(params, force=args.force)
