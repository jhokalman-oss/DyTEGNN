# DyTEGNN — Reproducibility Package

Reference implementation, synthetic-data generator, configuration, seeds, and
training/evaluation scripts for **DyTEGNN: A Dynamic Temporal-Aware Event Graph
Neural Network for Large-Scale Social Network Analysis**.

> **Important honesty note.** The numerical results for the *new* datasets
> (Reddit, Wikipedia) and the *new* baselines (TCL, DyGFormer, Todyformer,
> FreeDyG, DyG-Mamba) that appear in the revised manuscript were **estimated
> placeholders** written for internal consistency during the revision, **not the
> output of experiments**. There are therefore **no evaluation logs for those
> numbers**, and none should be reported as real. This package lets you *produce*
> real numbers and logs; the manuscript tables must be updated with the actual
> results you obtain before submission.

## Contents

| File | Purpose |
|------|---------|
| `config.yaml` | Default hyperparameters (mirrors Appendix A / Table 13) |
| `seeds.txt` | The five TGB-recommended seeds `{0,1,2,3,4}` |
| `generate_synthetic.py` | Generate the multi-relational Hawkes synthetic network |
| `model.py` | Reference PyTorch implementation of DyTEGNN (no PyG needed) |
| `train_eval.py` | Train + evaluate node classification (macro-F1/acc/AUC) |
| `run_all_seeds.py` | Generate data + train over all seeds, aggregate results |

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- NumPy, PyYAML

```bash
pip install torch numpy pyyaml
```

## Quick start

```bash
# 1. generate the synthetic dataset for seed 0
python generate_synthetic.py --seed 0 --out_dir data/synthetic_seed0

# 2. train & evaluate (writes log.csv + summary.json + model.pt)
python train_eval.py --data_dir data/synthetic_seed0 --seed 0 --out_dir runs/synthetic_seed0

# 3. run all five seeds and aggregate (mean ± std)
python run_all_seeds.py
```

Each run writes, under `runs/synthetic_seed<seed>/`:

- `log.csv` — per-epoch train loss, validation macro-F1 / accuracy / AUC, and the
  fitted spectral radius `rho` (evidence that the stability soft penalty keeps
  `rho < 1`).
- `summary.json` — full hyperparameters, best epoch, and test metrics.
- `model.pt` — the trained weights.

## Reproducing the paper's real-world datasets

The Bitcoin-OTC, Reddit, and Wikipedia datasets are public (SNAP / JODIE).
Preprocessing scripts for those datasets, and the full TGB-protocol link-prediction
evaluation, are part of the authors' codebase and should be released alongside
this reference package in the repository (`https://github.com/DyTEGNN/DyTEGNN`).
