# Repository Guidelines

## Project Structure & Module Organization
`gphyt/` is the main Python package. Keep data loading in `gphyt/data/`, model code in `gphyt/models/`, and training or evaluation entry points in `gphyt/train/`. Transformer-specific components live under `gphyt/models/transformer/`; tokenizer code lives in `gphyt/models/tokenizer/`. Tests mirror the package layout in `tests/` (`tests/test_data/`, `tests/test_models/`, `tests/test_train/`). Reference assets are stored in `images/`, and the default experiment config is [`gphyt/train/train.yml`](/workspace/physics-foundation-model-activation-steering/gphyt/train/train.yml).

## Build, Test, and Development Commands
Use the shared virtual environment described in the repo:

```bash
source ../.venv/bin/activate
uv pip install -e ".[dev]"
```

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
