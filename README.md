# MissBGM

Missingness-aware data imputation with Bayesian generative modeling and uncertainty quantification.

This repository contains:
- the core **MissBGM** model implementation
- synthetic MNAR experiments
- real-data benchmark preparation utilities
- non-diffusion baseline runners for comparison
- stress-test and summary scripts used for the NeurIPS draft

## Repository layout

```text
MissBGM/
├── configs/                       # YAML configs for synthetic + real-data runs
├── datasets/                      # staged raw data, processed arrays, benchmark caches
├── missbgm/
│   ├── datasets/                  # synthetic simulators + real-data staging/prep
│   ├── models/                    # MissBGM / BGM model code
│   └── utils/                     # RMSE + MNAR benchmark helpers
├── main.py                        # main entry point for MissBGM runs
├── run_real_baselines.py          # real-data baselines on normalized benchmark arrays
├── run_oracle_stress_missbgm.py   # synthetic oracle stress sweep for MissBGM
├── run_oracle_stress_baselines.py # synthetic oracle stress sweep for baselines
├── summarize_real_benchmarks.py   # aggregate cached real-data benchmark outputs
├── summarize_oracle_stress.py     # aggregate oracle-stress outputs
└── export_paper_tables.py         # helper for paper-ready table rows
```

## Main ideas

### Synthetic MNAR experiments
`main.py` can run the synthetic MNAR setting using `configs/BGM_MNAR.yaml`.

### Real-data benchmarks
The real-data pipeline:
- downloads and stages supported UCI datasets on demand
- standardizes each dataset with `StandardScaler`
- simulates a **nonlinear factor-coupled MNAR** mask
- saves benchmark arrays under:
  - `datasets/<DATASET>/benchmark/missing_nolinear_rate_0.2_seed_123/`

The canonical benchmark arrays are:
- `x_full_norm.npy`
- `x_obs_norm.npy`
- `mask.npy`

### Baselines
The real-data benchmark compares MissBGM against these non-diffusion baselines:
- `mean`
- `KNN`
- `miceforest`
- `ice`
- `softimpute`
- `OT`
- `GAIN`

## Supported real datasets

Current real-data registry:
- Wine
- Concrete
- Libras
- Breast Cancer Wisconsin (Original)
- HAR_AAL
- GasSensorDrift
- Superconductivity

Notes:
- Breast uses **UCI Breast Cancer Wisconsin (Original)**.
- Rows with native `?` values are removed before new missingness is simulated.

## Setup

There is no packaged install script yet, so setup is currently environment-based.

### Core dependencies
Typical dependencies used by this repo include:
- Python 3.9 or 3.10
- TensorFlow
- NumPy
- pandas
- PyYAML
- scikit-learn
- miceforest
- lightgbm
- torch (for OT baseline)
- openpyxl / xlrd as needed for UCI spreadsheets

### Baseline dependency caveat
`run_real_baselines.py` currently imports several utilities from a local ForestDiffusion baseline checkout through an absolute path:
- `gain`
- `imputers.OTimputer`
- `softimpute`
- related helper utilities

Before running baselines on another machine, update `FORESTDIFF_ROOT` in:
- `run_real_baselines.py`
- `run_oracle_stress_baselines.py`

## Quick start

### 1) Synthetic MissBGM run
```bash
python main.py -c configs/BGM_MNAR.yaml
```

### 2) Real-data MissBGM run
Example:
```bash
python main.py -c configs/BGM_REAL_Wine.yaml --force
```

This will:
- stage the dataset if needed
- build the nonlinear MNAR benchmark cache
- train MissBGM on normalized arrays only
- save results under `results/Real_MNAR/<DATASET>/missing_nolinear_rate_0.2_seed_123/`

### 3) Real-data baselines
Example:
```bash
python run_real_baselines.py --datasets Wine --methods mean KNN miceforest ice softimpute OT GAIN --force
```

Outputs are saved under:
```text
results/Real_MNAR/<DATASET>/missing_nolinear_rate_0.2_seed_123/baselines/
```

## Real-data output format

### MissBGM outputs
Each real-data MissBGM run writes:
- `missbgm_summary.json`
- `training_history.csv`
- `x_full_norm.npy`
- `x_obs_norm.npy`
- `mask.npy`

Important fields in `missbgm_summary.json`:
- `best_epoch`
- `lowest_rmse_normalized`
- `final_rmse_normalized`
- `actual_missing_rate`

### Baseline outputs
Each baseline method writes:
- `summary.json`
- `imputations.npy`
- `point_estimate_norm.npy`

The dataset-level baseline folder also writes:
- `summary.csv`

## Experiment conventions in the current real-data benchmark

For the current real-data protocol:
- all training/evaluation is done in **normalized space only**
- benchmark folder tag is exactly:
  - `missing_nolinear_rate_0.2_seed_123`
- MissBGM real-data runs use a single `model.fit(...)`
- real-data MissBGM summaries report the full training RMSE trace and best epoch

## Oracle stress scripts

Synthetic robustness sweeps are available through:
- `run_oracle_stress_missbgm.py`
- `run_oracle_stress_baselines.py`
- `run_oracle_stress_suite.sh`
- `summarize_oracle_stress.py`

These scripts evaluate sweeps over:
- missingness rate
- sample size
- feature dimension

Outputs are written under:
```text
results/Oracle_Stress/
```

## Data caching and staging

Real datasets are cached in stages:
- `datasets/<DATASET>/raw/`
- `datasets/<DATASET>/processed/`
- `datasets/<DATASET>/benchmark/...`

The staging code downloads UCI zip files directly from `archive.ics.uci.edu` when needed.

## Known limitations

- No `requirements.txt` or environment file is included yet.
- Baseline scripts depend on a separate local ForestDiffusion checkout.
- Some UCI downloads may fail transiently with HTTP errors and need reruns.
- Some helper scripts still reflect earlier paper workflows and may need cleanup before open-sourcing.

## Suggested next cleanup steps

If this repo is being prepared for broader reuse, the highest-value improvements would be:
1. add a reproducible environment file
2. remove machine-specific absolute paths
3. add a small smoke-test script
4. unify result-summary scripts around the current nonlinear real-data protocol
5. document expected runtime and hardware for each benchmark stage

## License

See `LICENSE`.
