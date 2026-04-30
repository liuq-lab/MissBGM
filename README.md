# MissBGM

Missingness-aware data imputation with Bayesian generative modeling and uncertainty quantification.

MissBGM is a missingness-aware Bayesian generative model for imputing data with potential **non-ignorable missingness** (Missing Not At Random, MNAR). It jointly models the **data-generating process** and the **missingness mechanism**, enabling both accurate imputations and principled uncertainty quantification via posterior sampling.

## Highlights

- Joint modeling data-generating process and missingness mechanisms
- MAP imputation for fast point estimates
- Posterior sampling (MCMC) for uncertainty quantification and prediction intervals
- Bridging Bayesian deep learning and missing data imputation

## Model overview

<p align="center">
  <img src="https://raw.githubusercontent.com/liuq-lab/MissBGM/main/assets/model.png" width="800" alt="MissBGM model overview" />
</p>
<p align="center">
  <em>Figure. MissBGM model overview.</em>
</p>

## Installation

### Create a conda environment

```bash
conda create -n missbgm python=3.12 -y
conda activate missbgm
```

### Install from source

```bash
git clone https://github.com/liuq-lab/MissBGM.git && cd MissBGM
pip install -e .
```

### Install from PyPI

```bash
pip install missbgm
```

### Dependencies

This project is tested in a `python 3.12` Linux environment with:

- `tensorflow==2.18.0`
- `tf-keras==2.18.0`
- `tensorflow-probability==0.25.0`
- `numpy`, `pandas`, `pyyaml`, `scikit-learn`

## Quickstart (Python API)

Below is a minimal end-to-end example that mirrors the logic in `main.py`: simulate MNAR data, train MissBGM, read MAP imputations, and then draw posterior samples to produce uncertainty intervals.

```python
import yaml

from missbgm.models import MissBGM
from missbgm.datasets import simulate_mnar_oracle_data

# Load experiment configuration
params = yaml.safe_load(open("configs/Sim_MNAR_oracle.yaml", "r"))

# Simulate MNAR-oracle data (synthetic benchmark)
data = simulate_mnar_oracle_data(
    n_samples=500,
    x_dim=50,
    n_anchor=5,
    missing_rate=0.5,
    alpha=0.05,
    seed=123,
)

# Instantiate the MissBGM model with the configuration
model = MissBGM(params, random_seed=42)

# Train the MissBGM model
model.fit(data=data["x_obs"], mask=data["mask"], x_true=data["x_full"], verbose=1)

# Get the MAP imputation
map_imputed = model.x_map_imputed_

# Make posterior predictions with uncertainty quantification
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
```

## Reproducing experiments with `main.py`

`main.py` is the primary experiment entrypoint. It reads a YAML config (`-c`) and runs:

- **Synthetic simulation** when `dataset: Sim_MNAR_oracle`
- **Real-data benchmark** otherwise (the code normalizes data, trains, and evaluates RMSE on missing entries)

### 1) Synthetic MNAR simulation

Use `configs/Sim_MNAR_oracle.yaml`:

```bash
python main.py -c configs/Sim_MNAR_oracle.yaml
```

### 2) Real-data benchmarks (4 datasets)

The following configs are provided:

- `configs/Real_Wine.yaml` (UCI Wine)
- `configs/Real_Concrete.yaml` (UCI Concrete)
- `configs/Real_Breast.yaml` (UCI Breast Cancer Wisconsin Original)
- `configs/Real_Gisette.yaml` (UCI Gisette (high-dimensional))

Run each dataset:

```bash
python main.py -c configs/Real_Wine.yaml
python main.py -c configs/Real_Concrete.yaml
python main.py -c configs/Real_Breast.yaml
python main.py -c configs/Real_Gisette.yaml
```

If you want to **re-run** even when cached staged files / outputs exist, add `--force`:

```bash
python main.py -c configs/Real_Wine.yaml --force
```

### What `main.py` does (pipeline summary)

For each run:

- **Loads config** from YAML (e.g., `configs/Real_Wine.yaml`)
- **Prepares data**
  - Synthetic: calls `simulate_mnar_oracle_data(...)`
  - Real: calls `prepare_real_benchmark_data(...)`
- **Trains** `MissBGM.fit(...)`
- **Computes MAP imputation** via `model.x_map_imputed_`
- **Draws posterior samples** via `model.predict(...)` to get:
  - `mcmc_imputed`: posterior mean imputation (from samples)
  - `intervals`: prediction intervals (controlled by `alpha`)
- **Reports metrics** such as RMSE on missing entries

## Project structure

```text
missbgm/
  datasets/        # synthetic and real dataset preparing / preprocessing
  models/          # MissBGM implementation
  utils/           # utility functions
configs/           # YAML configs for experiments
main.py            # main script to run the experiments
```

## Citation

If you use MissBGM in your research, please cite our paper:

```bibtex
@misc{missbgm2026,
  title        = {MissBGM: Missingness-aware Data Imputation via AI-powered Bayesian Generative Modeling with Uncertainty Quantification},
  author       = {Qiao Liu},
  year         = {2026},
  archivePrefix= {arXiv},
  primaryClass = {stat.ML}
}
```

