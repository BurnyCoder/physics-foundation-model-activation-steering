# Repository Guidelines

## Project Structure & Module Organization
`gphyt/` is the main Python package and implements GP<sub>hy</sub>T, the repository's physics foundation model: a transformer-based neural differentiator paired with numerical integration for multiphysics simulation rollouts. The README frames the project as a "train once, deploy anywhere" physics model with three core goals to preserve in code changes: one model across multiple physics regimes, zero-shot or in-context adaptation to unseen settings, and stable long-horizon autoregressive prediction. Keep data loading and preprocessing in `gphyt/data/`, model code in `gphyt/models/`, and training or evaluation entry points in `gphyt/train/`.

`gphyt/data/` contains the trajectory ingestion stack. `well_dataset.py` handles The Well-style HDF5 format, metadata, normalization, and timestep windowing; `phys_dataset.py` wraps that base dataset with project-specific options such as stride control, field selection, NaN handling, and spatial flips; `dataset.py`, `dataloader.py`, `dataset_utils.py`, and `normalize.py` contain shared dataset assembly, loader, and normalization helpers. The datasets described in the README are multiphysics trajectories stored as HDF5 fields, so keep new data logic compatible with that layout rather than baking in ad hoc local formats. In the README, data comes from both self-made simulations and [The Well](https://polymathic-ai.org/the_well/); files are generally HDF5 per parameter set with scalar, vector, and tensor features grouped under `t0`, `t1`, and `t2`. The documented physics coverage includes incompressible and compressible Navier-Stokes, flow with heat transfer, obstacles and wall interactions, two-phase flow, and natural convection, and the README describes the broader training corpus as 1.8 TB spanning 8 physical systems.

`gphyt/models/` contains both the core GP<sub>hy</sub>T model and baseline architectures. `gphyt/models/transformer/` is the main GP<sub>hy</sub>T stack and includes attention blocks, axial/full attention variants, positional encodings, derivative estimation, normalization layers, and numerical integrators such as Euler, Heun, and RK4. `gphyt/models/tokenizer/` contains the patch tokenization and detokenization code that maps physics fields to transformer tokens and back. Top-level modules such as `fno.py`, `unet.py`, and `resnet.py` provide comparison models, while `model_specs.py`, `model_utils.py`, and `loss_fns.py` define shared model sizes, helpers, and losses.

`gphyt/train/` contains the executable training and evaluation pipeline. `run_training.py` is the main training entry point, `train_base.py` implements the trainer loop with checkpointing and distributed support, `eval.py` and `model_eval.py` handle validation and rollout evaluation, and `utils/` contains logging, optimizer and LR scheduler setup, checkpoint utilities, timing, visualization, rollout video generation, and Weights & Biases integration. The default experiment config is [`gphyt/train/train.yml`](/workspace/physics-foundation-model-activation-steering/gphyt/train/train.yml), and `gphyt/train/scripts/` contains cluster launch helpers.

Tests mirror the package layout in `tests/` (`tests/test_data/`, `tests/test_models/`, `tests/test_train/`) and cover dataset handling, tokenizer and transformer internals, baseline models, and trainer utilities. Reference assets are stored in `images/`; the README uses `images/arch.png` for the architecture overview and `images/result.png` for representative zero-shot or in-context prediction results. The repository root also includes [`paper.pdf`](/workspace/physics-foundation-model-activation-steering/paper.pdf), the bundled full paper "Towards a Physics Foundation Model." It contains the research motivation and related work, the General Physics Transformer architecture, dataset descriptions, reference model baselines, core results, long-horizon predictions on individual physics settings, in-context learning results, ablations on what makes a useful physics foundation model, limitations, conclusion, and appendix sections with dataset details and model and training hyperparameters. Use it when you need the intended scientific framing or to confirm how a code path maps back to the paper. The README also points to the live paper at <https://arxiv.org/abs/2509.13805>, a qualitative results blog post at <https://flowsnr.github.io/blog/physics-foundation-model/>, and pretrained weights at <https://huggingface.co/flwi/Physics-Foundation-Model>; these are the first places to look for claims, visual examples, and checkpoint provenance. The README's headline results are also useful orientation when evaluating regressions: it claims strong gains over FNO on the multiphysics benchmark, zero-shot adaptation to new boundary conditions and novel physics, and stable 50-step autoregressive predictions.

## Build, Test, and Development Commands
Use the shared virtual environment described in the repo:

```bash
source ../.venv/bin/activate
uv pip install -e ".[dev]"
```

For dataset setup, the README uses The Well downloader. Example:

```bash
the-well-download --base-path /home/gphyt --dataset turbulent_radiative_layer_2D
```

Keep new setup notes consistent with that workflow and with the documented HDF5 structure expected by `gphyt/data/`.

Key commands:

```bash
pytest tests/
python gphyt/train/run_training.py --config_path gphyt/train/train.yml
python gphyt/train/model_eval.py --config_file <config> --sim_name <name> --log_dir <dir> --data_dir <dir>
torchrun --standalone --nproc_per_node=4 gphyt/train/run_training.py --config_path <config>
```

Use `gphyt/train/scripts/` for cluster launch helpers such as `train_riv.sh` and `eval.sh`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and short docstrings on non-obvious public helpers. Prefer typed function signatures when adding new APIs. No formatter or linter is configured in `pyproject.toml`, so keep imports, spacing, and line wrapping consistent with nearby files instead of reformatting unrelated code.

## Testing Guidelines
Tests use `pytest` and should track the module layout they cover. Name files `test_<module>.py` and keep fixtures in `tests/conftest.py` when shared. Run a focused test first, for example:

```bash
pytest tests/test_models/test_transformer/test_model.py -v
```

Add regression tests for model shape changes, config parsing, and dataset edge cases. Do not rely on local dataset paths in tests; use fixtures or temporary HDF5 data.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Add paper PDF`, `fix bug`, and `Restructure to ML-Training-Suite architecture`. Keep commit titles concise and specific to one change. PRs should explain the motivation, list code/config/data impacts, and note the exact tests run. Include sample metrics or plots when changing training, evaluation, or visualization behavior. Avoid committing `data/`, `results/`, `.env`, or generated checkpoints.

# Autonomous Research Agent: Activation Vector Steering on Physics Foundation Models

## Objective

Perform activation vector steering (mechanistic interpretability) on physics foundation models. Produce a **complete research paper** with an associated codebase, trained/steered models, and experimental results.

---

## 1. Inputs & Resources

- **Paper**: Read `paper.pdf` thoroughly before starting. This is your primary reference.
- **Codebase**: Read the entire existing codebase in this repo before writing any code.
- **Hardware**: RTX 4090 GPU, 200 GB on `/workspace`, 4 TB on Google Drive via rclone:
  ```
  rclone ls gdrive:projects/physics-foundation-model-mechinterp
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

# Plan