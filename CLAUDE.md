# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

General Physics Transformer (GPhyT) — a foundation model for physics simulation that combines a transformer-based neural differentiator with numerical integration. Trains on HDF5 datasets (The Well format) spanning diverse physical systems (Navier-Stokes, heat transfer, two-phase flow, etc.).

- Paper: https://arxiv.org/abs/2509.13805
- Weights: https://huggingface.co/flwi/Physics-Foundation-Model
- Blog: https://flowsnr.github.io/blog/physics-foundation-model/

## Build & Install

```bash
# Uses uv with hatchling build backend. PyTorch from cu129 index.
# Shared venv in parent mechinterp/ workspace.
source ../.venv/bin/activate
uv pip install -e ".[dev]"
```

Requires Python >= 3.12.

## Running Tests

```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_models/test_transformer/test_model.py

# Single test
pytest tests/test_models/test_transformer/test_model.py::TestClassName::test_method -v
```

Known: ~21 data tests fail because the dummy HDF5 fixtures in `tests/conftest.py` use generic field names (e.g. "variable_field1") instead of physics field names like "pressure". These are pre-existing failures.

## Training & Evaluation

```bash
# Single-GPU training
python gphyt/train/run_training.py --config_path <path_to_config.yaml>

# Multi-GPU with torchrun
torchrun --standalone --nproc_per_node=N gphyt/train/run_training.py --config_path <config>

# Evaluation
python gphyt/train/model_eval.py --config_file <config> --sim_name <name> --log_dir <dir> --data_dir <dir> --checkpoint_name <best_model|epoch_num> --forecast_horizons 1 4 8
```

SLURM scripts in `gphyt/train/scripts/` (train_riv.sh, eval.sh).

## Architecture

### Full Repo Structure

```
gphyt/
├── data/
│   ├── dataloader.py          # get_dataloader() — sampler selection (DDP/random/sequential)
│   ├── dataset.py             # get_dataset() — high-level factory (used by run_training.py)
│   ├── dataset_utils.py       # get_datasets(), get_dt_datasets() — per-stride factories (used by model_eval.py)
│   ├── normalize.py           # Normalization utilities
│   ├── phys_dataset.py        # PhysicsDataset (wraps WellDataset), SuperDataset (multi-dataset concat)
│   └── well_dataset.py        # Modified copy of the_well's WellDataset (HDF5 loader, StrideError, include_fields)
├── models/
│   ├── fno.py                 # Fourier Neural Operator (wraps neuraloperator library)
│   ├── loss_fns.py            # Generic losses: MSE, MAE, RMSE, NRMSE, VRMSE
│   ├── model_specs.py         # Size configs: GPT_S/M/L/XL, FNO_S/M, UNet_S/M (dataclasses)
│   ├── model_utils.py         # get_model() — architecture dispatcher
│   ├── resnet.py              # ResNet model
│   ├── unet.py                # U-Net model
│   ├── tokenizer/
│   │   ├── tokenizer.py       # Tokenizer & Detokenizer (linear or conv_net patchify/unpatchify)
│   │   └── tokenizer_utils.py # Tokenizer helper functions
│   └── transformer/
│       ├── attention.py       # AttentionBlock (MHA + MLP + norms)
│       ├── ax_attention.py    # Axial attention variant
│       ├── derivatives.py     # FiniteDifference module (dt, dh, dw spatial/temporal gradients)
│       ├── loss_fns.py        # GPhyT-specific losses: NMSELoss, VMSELoss, RNMSELoss, RVMSELoss
│       ├── model.py           # PhysicsTransformer — main GPhyT model class & get_model()
│       ├── norms.py           # RMSNorm, LayerNorm variants
│       ├── num_integration.py # Euler, RK4, Heun numerical integrators
│       └── pos_encodings.py   # RotaryPositionalEmbedding (RoPE), AbsPositionalEmbedding
└── train/
    ├── eval.py                # Evaluator class — validation loop with AR rollout support
    ├── model_eval.py          # Standalone detailed eval CLI (per-dataset, per-horizon metrics + viz)
    ├── run_training.py        # Main CLI entry point — config parsing, model/data/optimizer setup → Trainer
    ├── train.yml              # Reference config (nested YAML format)
    ├── train_base.py          # Trainer class — full training loop (DDP, AMP, checkpointing, W&B)
    ├── scripts/
    │   ├── eval.sh            # SLURM eval launcher
    │   └── train_riv.sh       # SLURM training launcher
    └── utils/
        ├── checkpoint_utils.py # save/load checkpoints, find_checkpoint(), load_stored_model()
        ├── logger.py           # setup_logger() with rank-aware formatting
        ├── lr_scheduler.py     # get_lr_scheduler() — LinearLR warmup + CosineAnnealingLR
        ├── optimizer.py        # get_optimizer() — Adam/AdamW setup
        ├── rollout_video.py    # Generate rollout videos from predictions
        ├── run_utils.py        # compute_metrics(), reduce_all_losses() (DDP allreduce)
        ├── time_keeper.py      # TimeKeeper for training time limits
        ├── train_vis.py        # Training visualization utilities
        └── wandb_logger.py     # WandbLogger wrapper

tests/                         # Mirrors gphyt/ structure; conftest.py creates dummy HDF5 fixtures
```

### Key Entry Points & Factory Functions

- **`gphyt/models/model_utils.py`** — `get_model(model_config)`: dispatches to `gphyt`, `unet`, or `fno` based on `architecture` key
- **`gphyt/data/dataset.py`** — `get_dataset(config, split)`: builds `SuperDataset` of `PhysicsDataset`s
- **`gphyt/train/run_training.py`** — CLI entry point; parses YAML config, builds model/data/optimizer, creates `Trainer`

### Model Architectures

- **GPhyT** (`gphyt/models/transformer/`): Input → optional FiniteDifference (dt, dh, dw concatenated, 4x channels) → Tokenizer (patchify) → positional encoding → N AttentionBlocks → Detokenizer → residual add. The `forward()` outputs `out[:, -1, ...].unsqueeze(1)` (single next-step prediction). Size variants: GPT_S (192d/12L), GPT_M (768d/12L), GPT_L (1024d/24L), GPT_XL (1280d/32L) in `model_specs.py`.
- **FNO** (`gphyt/models/fno.py`): Fourier Neural Operator wrapper using `neuraloperator` library's `FNO` class with `n_modes=(t,h,w)` tuple.
- **UNet** (`gphyt/models/unet.py`): Convolutional U-Net. Variants: UNet_S, UNet_M.

### Data Pipeline

- **`WellDataset`** (`gphyt/data/well_dataset.py`): modified copy of `the_well.data.datasets.WellDataset` with custom extensions (e.g., `StrideError`, `include_fields` filtering). This is NOT the upstream version.
- **`PhysicsDataset`** (`gphyt/data/phys_dataset.py`): wraps `WellDataset`, handles input/output windowing (`n_steps_input`, `n_steps_output`), z-score normalization, dt_stride, and data augmentation (flips). Returns `(x, y)` tensors of shape `(T, H, W, C)`.
- **`SuperDataset`** (`gphyt/data/phys_dataset.py`): concatenates multiple `PhysicsDataset`s, supports `max_samples` per dataset, reshuffles indices each epoch.
- **Two dataset factory paths**: `gphyt/data/dataset.py` → `get_dataset()` (used by `run_training.py`) and `gphyt/data/dataset_utils.py` → `get_datasets()`/`get_dt_datasets()` (used by `model_eval.py`, supports per-dt-stride evaluation).
- Data format: HDF5 per The Well spec — fields stored in `t0_fields/`, `t1_fields/`, `t2_fields/` groups with shape `(n_trajectories, n_steps, x, y [, n_dim])`.
- **`include_fields`** config controls which physics fields are loaded (defaults to `t0_fields: [pressure, density, temperature]`, `t1_fields: [velocity]`). Different datasets use different subsets — see `DATASET_FIELDS` in `gphyt/train/model_eval.py` for the mapping (channel indices: 0=pressure, 1=density, 2=temp, 3=vel_x, 4=vel_y).

### Training

- **`Trainer`** (`gphyt/train/train_base.py`): full training loop with DDP support, AMP (bfloat16/float16), gradient checkpointing via `mem_budget`, autoregressive training (`n_ar_steps`), W&B logging.
- **`Evaluator`** (`gphyt/train/eval.py`): validation loop supporting AR rollout evaluation.
- **`model_eval.py`**: standalone detailed evaluation with per-dataset, per-horizon metrics and visualization.

### Config Format

Nested YAML with top-level keys: `wandb`, `logging`, `model`, `training`, `data`. See `gphyt/train/train.yml` for the reference config. Also supports a flat format for backward compatibility (detected via presence of `training` and `data` keys — see `_is_nested_config()` in `run_training.py`).

### Loss Functions

- Generic: MSE, MAE, RMSE, NRMSE, VRMSE in `gphyt/models/loss_fns.py`
- GPhyT-specific: NMSELoss, VMSELoss, RNMSELoss, RVMSELoss in `gphyt/models/transformer/loss_fns.py` (support `return_scalar` and dimension selection)
