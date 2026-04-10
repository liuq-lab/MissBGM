"""Real-dataset staging and benchmark preparation utilities for MissBGM."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .simulators import _calibrate_logit_intercept, _sigmoid


DATASET_REGISTRY: Dict[str, dict] = {
    "Wine": {
        "uci_id": 109,
        "slug": "wine",
        "raw_dir_name": "Wine",
        "features": 13,
        "num_categorical_or_binary_features": 0,
        "notes": ["Uses wine.data. Drops the class label column and keeps the 13 numeric features."],
    },
    "Concrete": {
        "uci_id": 165,
        "slug": "concrete+compressive+strength",
        "raw_dir_name": "Concrete",
        "features": 8,
        "num_categorical_or_binary_features": 0,
        "notes": ["Uses Concrete_Data.xls. Drops the target strength column and keeps the 8 numeric predictors."],
    },
    "Libras": {
        "uci_id": 181,
        "slug": "libras+movement",
        "raw_dir_name": "Libras",
        "features": 90,
        "num_categorical_or_binary_features": 0,
        "notes": ["Uses movement_libras.data. Drops the final class label column and keeps 90 numeric features."],
    },
    "Breast": {
        "uci_id": 15,
        "slug": "breast+cancer+wisconsin+original",
        "raw_dir_name": "Breast",
        "features": 9,
        "num_categorical_or_binary_features": 9,
        "notes": [
            "Uses breast-cancer-wisconsin.data from Breast Cancer Wisconsin (Original).",
            "Drops the sample code number ID column.",
            "Rows with native '?' values are removed before simulating new MNAR missingness.",
        ],
    },
    "Superconductivity": {
        "uci_id": 464,
        "slug": "superconductivty+data",
        "raw_dir_name": "Superconductivity",
        "features": 81,
        "num_categorical_or_binary_features": 0,
        "notes": ["Uses train.csv. Drops the target critical temperature column and keeps 81 numeric features."],
    },
    "GasSensorDrift": {
        "uci_id": 224,
        "slug": "gas+sensor+array+drift+dataset",
        "raw_dir_name": "GasSensorDrift",
        "features": 128,
        "num_categorical_or_binary_features": 0,
        "notes": ["Parses the 10 libsvm-style batch*.dat files into a dense 128-feature matrix."],
    },
    "HAR_AAL": {
        "uci_id": 364,
        "slug": "smartphone+dataset+for+human+activity+recognition+har+in+ambient+assisted+living+aal",
        "raw_dir_name": "HAR_AAL",
        "features": 561,
        "num_categorical_or_binary_features": 0,
        "notes": ["Stacks dataset_uci/final_X_train.txt and final_X_test.txt into one 561-feature matrix."],
    },
}

DATASET_ALIASES = {
    "breastoriginal": "Breast",
    "breast": "Breast",
    "wine": "Wine",
    "concrete": "Concrete",
    "libras": "Libras",
    "superconductivity": "Superconductivity",
    "gassensordrift": "GasSensorDrift",
    "har_aal": "HAR_AAL",
    "har-aal": "HAR_AAL",
    "haraal": "HAR_AAL",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_name(name: str) -> str:
    if name in DATASET_REGISTRY:
        return name
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise KeyError(f"Unsupported real dataset: {name}")


def get_dataset_dir(dataset_name: str) -> Path:
    spec = DATASET_REGISTRY[resolve_dataset_name(dataset_name)]
    return repo_root() / "datasets" / spec["raw_dir_name"]


def get_benchmark_dir(dataset_name: str, missing_rate: float = 0.2, seed: int = 123) -> Path:
    canonical = resolve_dataset_name(dataset_name)
    return repo_root() / "datasets" / canonical / "benchmark" / f"missing_nolinear_rate_{missing_rate:.1f}_seed_{seed}"


def _download_zip(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        destination.write_bytes(response.read())


def _extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_wine(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    df = pd.read_csv(extracted_dir / "wine.data", header=None)
    x = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = df.iloc[:, 0].to_numpy(dtype=np.float32)
    feature_names = [
        "Alcohol",
        "Malicacid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315_of_diluted_wines",
        "Proline",
    ]
    return x, y, feature_names, {"dropped_rows": 0}


def _load_concrete(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    df = pd.read_excel(extracted_dir / "Concrete_Data.xls")
    x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)
    return x, y, df.columns[:-1].tolist(), {"dropped_rows": 0}


def _load_libras(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    df = pd.read_csv(extracted_dir / "movement_libras.data", header=None)
    x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)
    feature_names = [f"feature_{i + 1}" for i in range(x.shape[1])]
    return x, y, feature_names, {"dropped_rows": 0}


def _load_breast(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    df = pd.read_csv(extracted_dir / "breast-cancer-wisconsin.data", header=None, na_values=["?"])
    before = len(df)
    df = df.dropna(axis=0).reset_index(drop=True)
    dropped = before - len(df)
    x = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)
    feature_names = [
        "Clump_thickness",
        "Uniformity_of_cell_size",
        "Uniformity_of_cell_shape",
        "Marginal_adhesion",
        "Single_epithelial_cell_size",
        "Bare_nuclei",
        "Bland_chromatin",
        "Normal_nucleoli",
        "Mitoses",
    ]
    return x, y, feature_names, {"dropped_rows": int(dropped), "native_missing_rows_removed": int(dropped)}


def _load_superconductivity(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    df = pd.read_csv(extracted_dir / "train.csv")
    x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)
    return x, y, df.columns[:-1].tolist(), {"dropped_rows": 0}


def _parse_libsvm_dense(path: Path, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            labels.append(float(tokens[0]))
            row = np.zeros(n_features, dtype=np.float32)
            for token in tokens[1:]:
                index_text, value_text = token.split(":", 1)
                row[int(index_text) - 1] = np.float32(value_text)
            rows.append(row)
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.float32)


def _load_gas_sensor_drift(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    batch_dir = extracted_dir / "Dataset"
    batch_paths = sorted(batch_dir.glob("batch*.dat"), key=lambda path: int(re.findall(r"\d+", path.stem)[0]))
    x_parts = []
    y_parts = []
    for batch_path in batch_paths:
        x_batch, y_batch = _parse_libsvm_dense(batch_path, n_features=128)
        x_parts.append(x_batch)
        y_parts.append(y_batch)
    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    feature_names = [f"sensor_feature_{i + 1}" for i in range(x.shape[1])]
    return x, y, feature_names, {"dropped_rows": 0, "num_batches": len(batch_paths)}


def _load_har_aal(extracted_dir: Path) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    dataset_dir = extracted_dir / "dataset_uci"
    x_train = np.loadtxt(dataset_dir / "final_X_train.txt", dtype=np.float32, delimiter=",")
    x_test = np.loadtxt(dataset_dir / "final_X_test.txt", dtype=np.float32, delimiter=",")
    y_train = np.loadtxt(dataset_dir / "final_y_train.txt", dtype=np.float32)
    y_test = np.loadtxt(dataset_dir / "final_y_test.txt", dtype=np.float32)
    x = np.vstack([x_train, x_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]).astype(np.float32)
    features_path = dataset_dir / "features.txt"
    if features_path.exists():
        feature_names = pd.read_csv(features_path, header=None, sep=r"\s+", names=["index", "name"])["name"].tolist()
    else:
        feature_names = [f"feature_{i + 1}" for i in range(x.shape[1])]
    return x, y, feature_names, {"dropped_rows": 0}


LOADERS = {
    "Wine": _load_wine,
    "Concrete": _load_concrete,
    "Libras": _load_libras,
    "Breast": _load_breast,
    "Superconductivity": _load_superconductivity,
    "GasSensorDrift": _load_gas_sensor_drift,
    "HAR_AAL": _load_har_aal,
}


def stage_real_dataset(dataset_name: str, force: bool = False) -> dict:
    canonical = resolve_dataset_name(dataset_name)
    spec = DATASET_REGISTRY[canonical]
    dataset_dir = get_dataset_dir(canonical)
    raw_dir = dataset_dir / "raw"
    extracted_dir = raw_dir / "extracted"
    processed_dir = dataset_dir / "processed"
    metadata_path = processed_dir / "metadata.json"

    if metadata_path.exists() and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    zip_path = raw_dir / f"uci_{spec['uci_id']}.zip"
    url = f"https://archive.ics.uci.edu/static/public/{spec['uci_id']}/{spec['slug']}.zip"
    if force or not zip_path.exists():
        _download_zip(url, zip_path)
    if force or not extracted_dir.exists() or not any(extracted_dir.iterdir()):
        _extract_zip(zip_path, extracted_dir)

    x, y, feature_names, extra_metadata = LOADERS[canonical](extracted_dir)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "X_full_original.npy", x)
    np.save(processed_dir / "y.npy", y)
    _write_json(processed_dir / "feature_names.json", {"feature_names": list(feature_names)})

    metadata = {
        "dataset": canonical,
        "uci_id": spec["uci_id"],
        "zip_url": url,
        "num_rows": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "num_categorical_or_binary_features": int(spec.get("num_categorical_or_binary_features", 0)),
        "feature_min": float(np.min(x)),
        "feature_max": float(np.max(x)),
        "native_missing_values_removed": int(extra_metadata.get("native_missing_rows_removed", 0)),
        "notes": spec["notes"],
        **extra_metadata,
    }
    _write_json(metadata_path, metadata)
    return metadata


def simulate_real_mnar_mask(x_model: np.ndarray, missing_rate: float = 0.2, seed: int = 123) -> np.ndarray:
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must lie strictly between 0 and 1.")
    x_model = np.asarray(x_model, dtype=np.float32)
    rng = np.random.default_rng(seed)

    x_dim = x_model.shape[1]
    coupling = rng.normal(scale=0.3, size=(x_dim, x_dim)).astype(np.float32)
    coupling = coupling / np.sqrt(max(float(x_dim), 1.0))
    shared_signal = np.tanh(x_model @ coupling).astype(np.float32)
    missingness_scores = (0.6 * x_model + 0.4 * shared_signal).astype(np.float32)

    intercept = _calibrate_logit_intercept(missingness_scores, 1.0 - missing_rate)
    observation_prob = _sigmoid(missingness_scores + intercept).astype(np.float32)
    mask = rng.binomial(1, observation_prob).astype(np.float32)

    for row_idx in range(mask.shape[0]):
        if mask[row_idx].sum() == 0:
            mask[row_idx, int(rng.integers(0, mask.shape[1]))] = 1.0
    for col_idx in range(mask.shape[1]):
        if mask[:, col_idx].sum() == 0:
            mask[int(rng.integers(0, mask.shape[0])), col_idx] = 1.0
    return mask.astype(np.float32)


def prepare_real_benchmark_data(
    dataset_name: str,
    missing_rate: float = 0.2,
    seed: int = 123,
    force: bool = False,
) -> dict:
    canonical = resolve_dataset_name(dataset_name)
    stage_real_dataset(canonical, force=force)

    dataset_dir = get_dataset_dir(canonical)
    processed_dir = dataset_dir / "processed"
    benchmark_dir = get_benchmark_dir(canonical, missing_rate=missing_rate, seed=seed)
    metadata_path = benchmark_dir / "metadata.json"

    if metadata_path.exists() and not force and (benchmark_dir / "x_full_norm.npy").exists() and (benchmark_dir / "x_obs_norm.npy").exists():
        return load_real_benchmark_data(canonical, missing_rate=missing_rate, seed=seed)

    x_full_original = np.load(processed_dir / "X_full_original.npy").astype(np.float32)
    scaler = StandardScaler()
    x_full_norm = scaler.fit_transform(x_full_original).astype(np.float32)

    mask = simulate_real_mnar_mask(x_full_norm, missing_rate=missing_rate, seed=seed)
    x_obs_norm = x_full_norm.copy()
    x_obs_norm[mask == 0.0] = np.nan

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / "x_full_norm.npy", x_full_norm.astype(np.float32))
    np.save(benchmark_dir / "mask.npy", mask.astype(np.float32))
    np.save(benchmark_dir / "x_obs_norm.npy", x_obs_norm.astype(np.float32))

    for stale_name in ["x_full_model.npy", "x_obs_model.npy", "x_full_original.npy", "x_obs_original.npy"]:
        stale_path = benchmark_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    processed_metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "dataset": canonical,
        "missing_rate": float(missing_rate),
        "seed": int(seed),
        "missingness_mechanism": "nonlinear_factor_coupled",
        "num_rows": int(x_full_original.shape[0]),
        "num_features": int(x_full_original.shape[1]),
        "num_categorical_or_binary_features": int(processed_metadata.get("num_categorical_or_binary_features", 0)),
        "original_min": float(np.min(x_full_original)),
        "original_max": float(np.max(x_full_original)),
        "norm_min": float(np.nanmin(x_full_norm)),
        "norm_max": float(np.nanmax(x_full_norm)),
        "should_standardize": True,
        "actual_missing_rate": float(1.0 - mask.mean()),
        "normalization": {
            "method": "StandardScaler",
            "mean": scaler.mean_.astype(np.float32).tolist(),
            "scale": scaler.scale_.astype(np.float32).tolist(),
        },
    }
    _write_json(metadata_path, metadata)
    return load_real_benchmark_data(canonical, missing_rate=missing_rate, seed=seed)


def load_real_benchmark_data(dataset_name: str, missing_rate: float = 0.2, seed: int = 123) -> dict:
    canonical = resolve_dataset_name(dataset_name)
    dataset_dir = get_dataset_dir(canonical)
    processed_dir = dataset_dir / "processed"
    benchmark_dir = get_benchmark_dir(canonical, missing_rate=missing_rate, seed=seed)
    metadata = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
    x_full_original = np.load(processed_dir / "X_full_original.npy").astype(np.float32)
    mask = np.load(benchmark_dir / "mask.npy").astype(np.float32)
    x_obs_original = x_full_original.copy()
    x_obs_original[mask == 0.0] = np.nan
    x_full_norm = np.load(benchmark_dir / "x_full_norm.npy").astype(np.float32)
    x_obs_norm = np.load(benchmark_dir / "x_obs_norm.npy").astype(np.float32)

    return {
        "dataset": canonical,
        "benchmark_dir": benchmark_dir,
        "x_full_original": x_full_original,
        "x_obs_original": x_obs_original.astype(np.float32),
        "x_full_norm": x_full_norm,
        "x_obs_norm": x_obs_norm,
        "x_full_model": x_full_norm,
        "x_obs_model": x_obs_norm,
        "mask": mask,
        "metadata": metadata,
        "standardizer": None,
    }
