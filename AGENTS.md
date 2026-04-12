# Current Steering Status

- Steering package implemented under `gphyt/steering/` with backend adapters, feature extraction, task registry, vector fitting, sweep evaluation, CLI entrypoints, and a sample YAML config.
- Default backend order is `pyvene` first, then hook fallback. The `pyvene` adapter uses custom `keep_last_dim` interventions so full 5D GP<sub>hy</sub>T block outputs can be collected and steered without rewriting the model.
- Validation completed on `tests/test_steering` plus focused existing transformer/data/eval tests under `uv run pytest`.
- The steering CLI has been hardened for real runs:
  - checkpoint auto-resolution now prefers downloaded public weights and falls back to `weights/`
  - The Well download path now derives a correct base path from `data.data_dir`
  - activation labels are taken from an unnormalized raw-data copy when the dataset supports it
  - sweep reports now include model/task metadata, baseline next-step MSE, and non-finite output rate
- Public bootstrap execution has now reached a real checkpoint/data/report path:
  - public checkpoints `GPT_S`, `GPT_M`, `GPT_L`, and `GPT_XL` were downloaded to `artifacts/checkpoints/`
  - the public `rayleigh_benard` dataset was downloaded under `data/datasets/`
  - the public `shear_flow` and `turbulent_radiative_layer_2D` datasets are also available locally under `data/datasets/`
  - `PhysicsDataset` now maps public field names into the canonical 5-channel checkpoint layout and resizes samples to `data.out_shape`
  - a real `GPT_S` mean-pressure steering report was generated at `artifacts/bootstrap/reports/GPT_S-mean_pressure.csv`
  - cross-model `GPT_S/M/L/XL` mean-pressure reports and a `GPT_S` shear-vs-rayleigh regime report are now present under `artifacts/bootstrap/reports/`
  - 20-step autoregressive rollout reports for `GPT_S/M/L/XL` mean-pressure steering are now present under `artifacts/bootstrap/rollouts/`
  - transfer reports on public `turbulent_radiative_layer_2D` are now present under `artifacts/bootstrap/transfer/`
  - manuscript-ready figure and table assets were generated under `manuscript/generated/`
- Immediate next execution steps are:
  - regenerate manuscript assets and rebuild the PDF after the rollout/transfer additions
  - run the broader targeted validation suite and commit/push the current public-only study state
  - revisit `euler_multi_quadrants_periodicBC` only with a more storage-aware bootstrap strategy, since the first attempted partial train download exceeded the writable quota and was removed

## Current Open Questions

- Public dataset resolutions and field names differ from the checkpoint's canonical training layout. The current repo now handles this for the public bootstrap path, but the same logic still needs to be stress-tested on all remaining public datasets.
- `euler_multi_quadrants_periodicBC` remains the one public dataset not yet usable locally for this repo because the first train shard is extremely large; the previous partial download was deleted after it exhausted the writable quota.

# Repository Guidelines

## Project Structure & Module Organization
`gphyt/` is the main package. Keep new code in the closest existing subpackage:

- `gphyt/data/`: dataset loading, normalization, The Well compatibility, and dataloader utilities.
- `gphyt/models/`: model implementations such as `fno.py`, `resnet.py`, `unet.py`, plus `transformer/` and `tokenizer/`.
- `gphyt/train/`: training entrypoints, evaluation code, utilities, YAML configs, and SLURM helper scripts in `gphyt/train/scripts/`.
- `tests/`: mirrors the package layout with `test_data/`, `test_models/`, `test_models/test_transformer/`, `test_models/test_tokenizer/`, and `test_train/test_utils/`.
- `images/`: README figures. `weights/`: reference checkpoint artifacts such as `weights/gphyt-S.pth`. `paper.pdf`: bundled paper copy.

Place tests next to the corresponding area, for example `gphyt/models/transformer/model.py` -> `tests/test_models/test_transformer/test_model.py`.

## Build, Test, and Development Commands
This repo uses `uv` and editable installs; there is no Makefile or npm workflow.

- `uv pip install -e ".[dev]"`: install the package and pytest.
- `pytest -q`: run the full test suite.
- `pytest tests/test_models/test_transformer -q`: run a focused subset while iterating.
- `python gphyt/train/run_training.py --config_path gphyt/train/train.yml`: start training from the nested YAML config.
- `python gphyt/train/model_eval.py --config_file <run_dir>/config_eval.yaml --log_dir results --sim_name <run> --data_dir data/datasets --checkpoint_name best_model`: run evaluation for a saved training run.

The shell scripts in `gphyt/train/scripts/` are HPC-oriented SLURM examples, not the primary local development path.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for modules/functions/config keys, PascalCase for classes, and UPPER_CASE for constants. Match surrounding import grouping and docstring style rather than reformatting unrelated code. No formatter or linter is configured in `pyproject.toml`, so keep edits minimal and consistent with nearby code.

## Testing Guidelines
Use `pytest` and add tests for each bug fix or behavior change. Name files `test_<unit>.py` and functions `test_<behavior>()`. Reuse shared fixtures from `tests/conftest.py`. GPU-specific coverage exists in several model tests; guard CUDA-only assertions with `pytest.skip(...)` when hardware is unavailable.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Remove stale venv activation docs` and `Make INCLUDE_FIELDS configurable instead of hardcoded`. Keep subjects concise, capitalized, and focused on one change. PRs should explain the affected module, summarize validation performed (`pytest -q`, targeted tests, or training/eval smoke tests), and link related issues or experiment context. Include figures only for visual or README changes.

## Configuration & Data Notes
Keep machine-specific paths, secrets, and W&B details out of committed configs when possible. Dataset locations are controlled through YAML under `data.data_dir`; large datasets and generated results should stay outside the repository.

# Towards a Physics Foundation Model

Paper: https://arxiv.org/abs/2509.13805

Blog post if you want to see some cool results: https://flowsnr.github.io/blog/physics-foundation-model/

Weights: https://huggingface.co/flwi/Physics-Foundation-Model

## Introduction

This repository contains the official implementation of the **General Physics Transformer (GP<sub>hy</sub>T)**, a foundation model for physics simulation presented in our paper (under review) "Towards a Physics Foundation Model."

### What is GP<sub>hy</sub>T?

<img src="images/arch.png" width="800" alt="GP<sub>hy</sub>T Architecture">

*Architecture overview: GP<sub>hy</sub>T combines a transformer-based neural differentiator with numerical integration, enabling robust and generalizable physics simulation.*

GP<sub>hy</sub>T represents a paradigm shift in physics-aware machine learning—moving from specialized, single-physics models to a unified "**train once, deploy anywhere**" approach. Our model demonstrates three groundbreaking capabilities:

**Multi-Physics Mastery**: A single model effectively simulates diverse physical systems including fluid-solid interactions, shock waves, thermal convection, and multi-phase dynamics—**without being explicitly told the governing equations**.

**Zero-Shot Generalization**: Through in-context learning, GP<sub>hy</sub>T adapts to entirely unseen physical systems and boundary conditions by inferring the underlying dynamics from input prompts alone.

**Long-Term Stability**: Maintains physically plausible predictions through extended 50-timestep autoregressive rollouts.

### Key Results

- **29× better performance** than specialized neural operators (FNO) on multi-physics benchmarks
- **Zero-shot adaptation** to new boundary conditions and entirely novel physics
- **1.8TB training corpus** spanning 8 distinct physical systems
- **Stable long-term predictions** with consistent physical behavior

The image below showcases GP<sub>hy</sub>T's ability to predict the evolution of physical systems it has never seen during training, purely from context.

<img src="images/result.png" width="800" alt="GP<sub>hy</sub>T generalization capabilities">


### Why This Matters

Current physics-aware ML models are fundamentally limited to single, narrow domains and require retraining for each new system. GP<sub>hy</sub>T breaks this barrier by learning generalizable physical principles from diverse simulation data, opening the path toward a universal physics foundation model that could:

- **Democratize** access to high-fidelity simulations
- **Accelerate** scientific discovery across disciplines
- **Eliminate** the need for specialized solver development for each new problem

## Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# From the project directory
uv pip install -e ".[dev]"
```

## Datasets

This study uses both self-made datasets and datasets from [The Well](https://polymathic-ai.org/the_well/).
The well-datasets can be downloaded like this:

```bash
the-well-download --base-path /home/gphyt --dataset turbulent_radiative_layer_2D
```

All datasets are formatted according to the-well [format](https://polymathic-ai.org/the_well/data_format/).
In general, the data are hdf5 files, one for each parameter set.
Inside the hdf5 files, the features are stored in the t0 (scalar), t1 (vector), and t2 (tensor) groups.
The arrays are shaped as (n_trajectories, n_steps, x, y) or for vector features as (n_trajectories, n_steps, x, y, 2).

### Physics

The datasets cover the following physics:

- Incompressible Navier-Stokes
- Compressible Navier-Stokes
- Flow with heat transfer
- Obstacles and wall interactions
- Two-phase flow
- Natural convection

# Repository Guidelines

## Project Structure & Module Organization
`gphyt/` is the main package. Keep new code in the closest existing subpackage:

- `gphyt/data/`: dataset loading, normalization, The Well compatibility, and dataloader utilities.
- `gphyt/models/`: model implementations such as `fno.py`, `resnet.py`, `unet.py`, plus `transformer/` and `tokenizer/`.
- `gphyt/train/`: training entrypoints, evaluation code, utilities, YAML configs, and SLURM helper scripts in `gphyt/train/scripts/`.
- `tests/`: mirrors the package layout with `test_data/`, `test_models/`, `test_models/test_transformer/`, `test_models/test_tokenizer/`, and `test_train/test_utils/`.
- `images/`: README figures. `weights/`: reference checkpoint artifacts such as `weights/gphyt-S.pth`. `paper.pdf`: bundled paper copy.

Place tests next to the corresponding area, for example `gphyt/models/transformer/model.py` -> `tests/test_models/test_transformer/test_model.py`.

## Build, Test, and Development Commands
This repo uses `uv` and editable installs; there is no Makefile or npm workflow.

- `uv pip install -e ".[dev]"`: install the package and pytest.
- `pytest -q`: run the full test suite.
- `pytest tests/test_models/test_transformer -q`: run a focused subset while iterating.
- `python gphyt/train/run_training.py --config_path gphyt/train/train.yml`: start training from the nested YAML config.
- `python gphyt/train/model_eval.py --config_file <run_dir>/config_eval.yaml --log_dir results --sim_name <run> --data_dir data/datasets --checkpoint_name best_model`: run evaluation for a saved training run.

The shell scripts in `gphyt/train/scripts/` are HPC-oriented SLURM examples, not the primary local development path.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for modules/functions/config keys, PascalCase for classes, and UPPER_CASE for constants. Match surrounding import grouping and docstring style rather than reformatting unrelated code. No formatter or linter is configured in `pyproject.toml`, so keep edits minimal and consistent with nearby code.

## Testing Guidelines
Use `pytest` and add tests for each bug fix or behavior change. Name files `test_<unit>.py` and functions `test_<behavior>()`. Reuse shared fixtures from `tests/conftest.py`. GPU-specific coverage exists in several model tests; guard CUDA-only assertions with `pytest.skip(...)` when hardware is unavailable.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Remove stale venv activation docs` and `Make INCLUDE_FIELDS configurable instead of hardcoded`. Keep subjects concise, capitalized, and focused on one change. PRs should explain the affected module, summarize validation performed (`pytest -q`, targeted tests, or training/eval smoke tests), and link related issues or experiment context. Include figures only for visual or README changes.

## Configuration & Data Notes
Keep machine-specific paths, secrets, and W&B details out of committed configs when possible. Dataset locations are controlled through YAML under `data.data_dir`; large datasets and generated results should stay outside the repository.

# Towards a Physics Foundation Model

Paper: https://arxiv.org/abs/2509.13805

Blog post if you want to see some cool results: https://flowsnr.github.io/blog/physics-foundation-model/

Weights: https://huggingface.co/flwi/Physics-Foundation-Model

## Introduction

This repository contains the official implementation of the **General Physics Transformer (GP<sub>hy</sub>T)**, a foundation model for physics simulation presented in our paper (under review) "Towards a Physics Foundation Model."

### What is GP<sub>hy</sub>T?

<img src="images/arch.png" width="800" alt="GP<sub>hy</sub>T Architecture">

*Architecture overview: GP<sub>hy</sub>T combines a transformer-based neural differentiator with numerical integration, enabling robust and generalizable physics simulation.*

GP<sub>hy</sub>T represents a paradigm shift in physics-aware machine learning—moving from specialized, single-physics models to a unified "**train once, deploy anywhere**" approach. Our model demonstrates three groundbreaking capabilities:

**Multi-Physics Mastery**: A single model effectively simulates diverse physical systems including fluid-solid interactions, shock waves, thermal convection, and multi-phase dynamics—**without being explicitly told the governing equations**.

**Zero-Shot Generalization**: Through in-context learning, GP<sub>hy</sub>T adapts to entirely unseen physical systems and boundary conditions by inferring the underlying dynamics from input prompts alone.

**Long-Term Stability**: Maintains physically plausible predictions through extended 50-timestep autoregressive rollouts.

### Key Results

- **29× better performance** than specialized neural operators (FNO) on multi-physics benchmarks
- **Zero-shot adaptation** to new boundary conditions and entirely novel physics
- **1.8TB training corpus** spanning 8 distinct physical systems
- **Stable long-term predictions** with consistent physical behavior

The image below showcases GP<sub>hy</sub>T's ability to predict the evolution of physical systems it has never seen during training, purely from context.

<img src="images/result.png" width="800" alt="GP<sub>hy</sub>T generalization capabilities">


### Why This Matters

Current physics-aware ML models are fundamentally limited to single, narrow domains and require retraining for each new system. GP<sub>hy</sub>T breaks this barrier by learning generalizable physical principles from diverse simulation data, opening the path toward a universal physics foundation model that could:

- **Democratize** access to high-fidelity simulations
- **Accelerate** scientific discovery across disciplines
- **Eliminate** the need for specialized solver development for each new problem

## Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# From the project directory
uv pip install -e ".[dev]"
```

## Datasets

This study uses both self-made datasets and datasets from [The Well](https://polymathic-ai.org/the_well/).
The well-datasets can be downloaded like this:

```bash
the-well-download --base-path /home/gphyt --dataset turbulent_radiative_layer_2D
```

All datasets are formatted according to the-well [format](https://polymathic-ai.org/the_well/data_format/).
In general, the data are hdf5 files, one for each parameter set.
Inside the hdf5 files, the features are stored in the t0 (scalar), t1 (vector), and t2 (tensor) groups.
The arrays are shaped as (n_trajectories, n_steps, x, y) or for vector features as (n_trajectories, n_steps, x, y, 2).

### Physics

The datasets cover the following physics:

- Incompressible Navier-Stokes
- Compressible Navier-Stokes
- Flow with heat transfer
- Obstacles and wall interactions
- Two-phase flow
- Natural convection

# Autonomous Research Agent: Activation Vector Steering on Physics Foundation Models

## Objective

Perform activation vector steering (mechanistic interpretability) on physics foundation models. Produce a **complete research paper** with an associated codebase, trained/steered models, and experimental results.

---

## 1. Inputs & Resources

- **Paper**: Read `paper.pdf` thoroughly before starting. This is your primary reference.
- **Codebase**: Read the entire existing codebase in this repo before writing any code.
- **Hardware**: RTX 4090 GPU, 200 GB on `/workspace`, 4 TB on Google Drive via rclone:
  ```
  rclone ls gd:projects/physics-foundation-model-mechinterp
  ```

## 2. Environment & Setup

- Use uv.
- Install all necessary dependencies through uv add, apt install, apt-get, etc. DO NOT FALLBACK to what is installed if you want to use something that isn't installed yet, and just install it!
- Work inside this (currently empty) repo. **Do not browse other files on this machine or other branches in GitHub.**
- Use **GitHub** for version control — commit and push regularly.

## 3. Development Principles

### Simplicity First
- Keep everything as simple as possible.
- Reuse existing libraries and GitHub repos. Search the web extensively before building anything from scratch. Only write custom code for functionality that truly doesn't exist elsewhere.

### Test-Driven Development
- Write tests before or alongside implementation.
- Write logs.
- Run the codebase in practice after each significant change. Inspect logs, validate outputs, fix issues, then push.

### Iterative Research Loop
- If an approach works, keep it. If it doesn't, discard it and try something else.
- Advance the GitHub branch incrementally so you can iterate (and theoretically rewind, though do this very sparingly if ever).
- Search the web often on how to do things to make sure that you're grounded.
- If stuck: browse the web for solutions, re-read the paper and referenced works, search for new libraries/codebases, re-examine in-scope files for new angles, combine previous near-misses, or try more radical changes.

### Continuous Planning

Write your current plan continually to AGENTS.md. Before starting any new phase, update AGENTS.md with your current understanding, next steps, open questions, and revised plan. This file should always reflect your latest thinking — treat it as a living document, not a one-time write.

## 4. Quality Checkpoints

After the initial implementation — and periodically throughout — verify:

- [ ] Do the current research results make sense?
- [ ] Does the code run end-to-end without errors?
- [ ] Do outputs make practical/physical sense?
- [ ] Is logging sufficient and informative? Do the logs tell a clear story?
- [ ] Is `README.md` accurate and up to date?
- [ ] Is `AGENTS.md` accurate and up to date?
- [ ] Is everything committed and pushed to GitHub?
- [ ] Are tests passing?

If any answer is **no**, fix it before moving on.

## 5. Deliverables

1. **Codebase**: Clean, well-documented, tested code in the GitHub repo.
2. **Experiments**: Completed experiments with logged results.
3. **Models**: Any trained or steered model artifacts (store large files on Google Drive).
4. **Research Paper**: A full paper written in LaTeX, including all results, analysis, and figures.

Record results and learnings **as you go** — do not defer writing until the end.

## 6. Autonomy Rules

> **NEVER STOP UNTIL YOU HAVE A FULL RESEARCH PAPER.**

- After initial setup, **do not** pause to ask the human if you should continue.
- **Do not** ask "should I keep going?" or "is this a good stopping point?"
- The human may be asleep or away and expects you to work **indefinitely** until the paper is complete or you are manually interrupted.
- You are fully autonomous. The loop runs until the research paper is finished, period.

**Example pacing**: If each experiment takes ~5 minutes, you can run ~12/hour — roughly 100 experiments over an average sleep cycle. The human wakes up to completed results.

## 7. Workflow Summary

```
1. Read paper.pdf and the full codebase
2. Search the web for relevant libraries, repos, and techniques
3. Set up environment (venv, dependencies, GitHub)
4. Implement (TDD, simple, reuse existing tools)
5. Run experiments → log results → inspect → fix (browse the web) → push
6. Repeat step 5, recording findings as you go
7. Write the LaTeX paper with all results
8. Final quality checkpoint (see §4)
9. Push everything to GitHub
```

# Library-First Activation Steering For Public GPhyT Models

  ## Summary

  - Build the project around existing libraries first, with custom code only for GP<sub>hy</sub>T-specific adapters, physics-feature extraction, and rollout evaluation.
  - Compare all public GPhyT checkpoints S/M/L/XL, with GPT_XL as the primary analysis model.
  - Run three steering tracks:
      - Regime steering
      - Simple feature steering
      - Complex derived-physics steering
  - Keep the project public-only unless a non-public dataset path is later provided.

  ## Library Reuse Decisions

  - Use pyvene as the default intervention engine.
      - Reason: its docs and repo state that it supports interventions on any PyTorch model and serializable intervention configs.
      - Use it for activation tracing, cached activations, and inference-time interventions on tokenizer output, selected transformer block outputs, and detokenizer-adjacent
        activations.
  - Use huggingface_hub for checkpoint download and provenance tracking.
      - Download the public GP<sub>hy</sub>T checkpoints directly rather than scripting manual HTTP fetches.
  - Use the_well for public dataset download and dataset access.
      - Keep the repo’s existing dataset wrappers, but do not replace the upstream The Well download/load path with custom download code.
  - Use scikit-learn for:
      - PCA
      - LogisticRegression
      - probe metrics such as roc_auc_score
  - Use scipy.stats.bootstrap for confidence intervals on steering effects.
  - Use Captum for attribution sanity-check experiments only.
      - This is an auxiliary analysis track, not the primary steering mechanism.
  - Use matplotlib and seaborn for plots.
  - Use tqdm for long-running collection and eval jobs.
  - Do not build on TransformerLens, AxBench, or SAELens in v1.
      - They are targeted at GPT/LLM workflows rather than arbitrary 5D physics models.
  - Do not use custom hooks unless the initial pyvene spike fails on GP<sub>hy</sub>T tensor shapes or module wrapping.
      - If fallback is needed, mirror the same public interfaces so the rest of the pipeline does not change.

  ## Minimal Custom Code

  - gphyt/steering/adapters.py
      - Map GP<sub>hy</sub>T module names and tensor conventions into a stable intervention spec usable by pyvene.
  - gphyt/steering/features.py
      - Compute physics-aware labels and metrics from (T,H,W,C) fields.
      - This remains custom because no existing steering library understands GP<sub>hy</sub>T’s channel semantics.
  - gphyt/steering/tasks.py
      - Registry of steering tasks, dataset eligibility, required channels, and held-out transfer sets.
  - gphyt/steering/eval.py
      - Bridge steered model outputs into the repo’s rollout evaluation and feature-shift reporting.
  - gphyt/steering/cli.py
      - Thin wrappers around library-backed workflows so experiments are reproducible from the command line.
  - manuscript/
      - LaTeX paper and generated figures.

  ## Steering Tasks

  - Regime steering:
      - shear_flow vs euler_multi_quadrants_periodicbc
      - shear_flow vs rayleigh_benard
      - transfer to turbulent_radiative_layer_2D
  - Simple feature steering:
      - mean velocity magnitude
      - high-velocity tail
      - mean pressure
      - pressure contrast
  - Complex feature steering:
      - enstrophy proxy
      - divergence magnitude
      - vortex intermittency
      - mean density
      - density contrast
      - shock score
      - stratification score
  - Feature labels:
      - compute on the last input frame
      - z-score within dataset
      - fit top-vs-bottom quartile contrastive directions
  - Steering methods:
      - main: difference-of-means
      - ablations: logistic-probe normal and PCA direction
      - if pyvene can host them directly, store them as intervention configs; otherwise store plain tensors plus metadata

  ## Experiment Protocol

  - Phase 0: library suitability spike
      - Verify pyvene can wrap GPT_XL and target GP<sub>hy</sub>T internal modules without changing outputs when steering is disabled.
      - If successful, freeze pyvene as the intervention backend.
      - If not, fall back to plain PyTorch hooks with the same CLI and artifact schema.
  - Phase 1: public asset bootstrap
      - Install missing dependencies with uv.
      - Download public checkpoints with huggingface_hub.
      - Download public The Well datasets with the_well.
  - Phase 2: baseline evaluation
      - Run unsteered next-step and rollout eval on S/M/L/XL.
  - Phase 4: vector fitting
      - Select the top 3 layers per task per model size by held-out probe quality.
  - Phase 5: steered evaluation
      - Run a fixed symmetric scale sweep across all selected tasks and sizes.
      - Report task-shift, off-target drift, and stability metrics.
  - Phase 6: attribution sanity checks
      - Use Captum on a small subset to compare whether salient inputs align with steered features.
  - Phase 7: manuscript
      - Write the full LaTeX paper with scaling tables, transfer results, and failure cases.

  ## Test Plan

  - Add a smoke test proving pyvene can attach to a small GP<sub>hy</sub>T instance and preserve outputs under no-op intervention.
  - Add a fallback smoke test for the hook backend only if Phase 0 fails.
  - Add unit tests for all custom feature extractors on analytic toy fields.
  - Add unit tests for vector serialization, dataset-z-scoring, and quartile label generation.
  - Add an end-to-end test for collect -> fit -> steer -> eval.
  - Add one-batch smoke runs for S/M/L/XL.
  - Add a manuscript build smoke test.

  ## Assumptions And Defaults

  - Default backend is pyvene, not hand-written hooks.
  - Captum is auxiliary, not the core steering engine.
  - Public assets only: GP<sub>hy</sub>T checkpoints from Hugging Face and datasets available through The Well.
  - Custom code is limited to GP<sub>hy</sub>T adapters, physics-feature definitions, rollout evaluation glue, and paper generation.
  - If pyvene fails on GP<sub>hy</sub>T in Phase 0, the fallback is custom hooks, but only after that failure is demonstrated concretely.

  
