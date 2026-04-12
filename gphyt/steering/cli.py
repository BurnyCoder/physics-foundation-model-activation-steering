"""Command-line entry points for GPhyT activation steering."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from huggingface_hub import hf_hub_download, model_info
import torch
import yaml

from gphyt.data.dataset_utils import get_datasets
from gphyt.models.model_utils import get_model
from gphyt.steering.adapters import get_backend, list_layer_ids
from gphyt.steering.eval import (
    aggregate_sweep_reports,
    collect_activation_dataset,
    fit_task_vectors,
    load_activation_collection,
    load_direction,
    run_scale_sweep,
    save_activation_collection,
    save_direction,
    write_sweep_report,
)
from gphyt.steering.tasks import get_task


WELL_DATASET_ALIASES = {
    "euler_multi_quadrants_openbc": "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicbc": "euler_multi_quadrants_periodicBC",
    "turbulent_radiative_layer_2d": "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3d": "turbulent_radiative_layer_3D",
}


def load_config(path: str | Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _device_from_config(config: dict) -> torch.device:
    device_name = config.get("steering", {}).get("device")
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _configured_model_size(config: dict) -> str:
    return config.get("model", {}).get("transformer", {}).get("model_size", "GPT_S")


def _model_sizes_from_config(config: dict) -> list[str]:
    model_sizes = config.get("steering", {}).get("model_sizes")
    if model_sizes:
        return list(model_sizes)
    return [_configured_model_size(config)]


def _resolve_well_base_path(config: dict) -> Path:
    configured = config.get("data", {}).get("well_base_path")
    if configured:
        return Path(configured)
    data_dir = Path(config["data"]["data_dir"])
    if data_dir.name == "datasets":
        return data_dir.parent
    return data_dir


def _well_registry_name(dataset_name: str) -> str:
    return WELL_DATASET_ALIASES.get(dataset_name, dataset_name)


def _ensure_canonical_dataset_path(base_path: Path, dataset_name: str, registry_name: str) -> None:
    datasets_root = base_path / "datasets"
    canonical_path = datasets_root / dataset_name
    registry_path = datasets_root / registry_name
    if dataset_name == registry_name or canonical_path.exists() or not registry_path.exists():
        return
    canonical_path.symlink_to(registry_path.name, target_is_directory=True)


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        while new_key.startswith("module.") or new_key.startswith("_orig_mod."):
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod.") :]
        cleaned[new_key] = value
    return cleaned


def _checkpoint_spec(config: dict, model_size: str) -> dict | None:
    return config.get("assets", {}).get("checkpoints", {}).get(model_size)


def resolve_checkpoint_path(
    config: dict,
    model_size: str | None = None,
) -> Path | None:
    model_size = model_size or _configured_model_size(config)

    explicit_paths = [
        config.get("checkpoint_path"),
        config.get("model", {}).get("checkpoint_path"),
        config.get("steering", {}).get("checkpoint_path"),
    ]
    for path in explicit_paths:
        if path and Path(path).exists():
            return Path(path)

    spec = _checkpoint_spec(config, model_size)
    if spec is None:
        return None

    candidate_paths = [
        Path(config["outputs"]["artifact_dir"]) / "checkpoints" / model_size / spec["filename"],
        Path("weights") / spec["filename"],
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def model_config_for_size(config: dict, model_size: str) -> dict:
    updated = deepcopy(config)
    updated.setdefault("model", {})
    updated["model"].setdefault("transformer", {})
    updated["model"]["transformer"]["model_size"] = model_size
    data_config = updated.get("data", {})
    if "n_steps_input" in data_config and "out_shape" in data_config:
        updated["model"]["img_size"] = (
            data_config["n_steps_input"],
            data_config["out_shape"][0],
            data_config["out_shape"][1],
        )
    return updated


def build_model_from_config(
    config: dict,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    model_size: str | None = None,
) -> torch.nn.Module:
    if model_size is not None:
        config = model_config_for_size(config, model_size)
    model_config = config["model"] if "model" in config else config
    model = get_model(model_config)
    checkpoint_path = checkpoint_path or resolve_checkpoint_path(config, model_size=model_size)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(_strip_state_dict_prefixes(state_dict), strict=False)
    device = device or _device_from_config(config)
    model.to(device)
    model.eval()
    return model


def _feature_names_from_config(config: dict) -> list[str]:
    feature_names = config.get("steering", {}).get("feature_names")
    if feature_names:
        return list(feature_names)
    task_ids = config.get("steering", {}).get("task_ids", [])
    task_feature_names = []
    for task_id in task_ids:
        task = get_task(task_id)
        if task.feature_name is not None:
            task_feature_names.append(task.feature_name)
    return sorted(set(task_feature_names))


def download_assets_command(config: dict) -> Path:
    output_dir = Path(config["outputs"]["artifact_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    selected_model_sizes = set(_model_sizes_from_config(config))
    checkpoint_specs = config.get("assets", {}).get("checkpoints", {})
    repo_revisions: dict[str, str | None] = {}

    for model_size, checkpoint_spec in checkpoint_specs.items():
        if model_size not in selected_model_sizes:
            continue
        repo_id = checkpoint_spec["repo_id"]
        if repo_id not in repo_revisions:
            repo_revisions[repo_id] = model_info(repo_id).sha
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_spec["filename"],
            local_dir=checkpoint_dir / model_size,
        )
        local_path = Path(local_path)
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        manifest.append(
            {
                "model_size": model_size,
                "repo_id": repo_id,
                "revision": repo_revisions[repo_id],
                "filename": checkpoint_spec["filename"],
                "local_path": str(local_path),
                "sha256": digest,
                "size_bytes": local_path.stat().st_size,
            }
        )

    well_base_path = _resolve_well_base_path(config)
    well_download_command = [
        "the-well-download",
        "--base-path",
        str(well_base_path),
    ]
    if config.get("assets", {}).get("first_only", False):
        well_download_command.append("--first-only")
    if config.get("assets", {}).get("download_parallel") is False:
        well_download_command.append("--no-parallel")
    split = config.get("assets", {}).get("well_split")
    for dataset_name in config.get("assets", {}).get("well_datasets", []):
        registry_name = _well_registry_name(dataset_name)
        dataset_command = well_download_command + ["--dataset", registry_name]
        if split:
            dataset_command.extend(["--split", split])
        subprocess.run(dataset_command, check=True)
        _ensure_canonical_dataset_path(well_base_path, dataset_name, registry_name)

    manifest_path = output_dir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def collect_activations_command(
    config: dict,
    model: torch.nn.Module | None = None,
    datasets: dict | None = None,
    output_path: str | Path | None = None,
) -> Path:
    device = _device_from_config(config)
    model_size = _configured_model_size(config)
    model = (
        build_model_from_config(config, device=device, model_size=model_size)
        if model is None
        else model.to(device)
    )
    datasets = (
        get_datasets(config["data"], split=config.get("split", "train"))
        if datasets is None
        else datasets
    )
    backend = get_backend(config.get("steering", {}).get("backend", "auto"))
    layer_ids = config.get("steering", {}).get("layer_ids") or list_layer_ids(model)
    collection = collect_activation_dataset(
        model=model,
        backend=backend,
        datasets=datasets,
        batch_size=int(config.get("steering", {}).get("batch_size", 4)),
        feature_names=_feature_names_from_config(config),
        layer_ids=layer_ids,
        max_batches=config.get("steering", {}).get("max_batches"),
        device=device,
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(config["outputs"]["activation_collection"])
    )
    return save_activation_collection(output_path, collection)


def fit_vectors_command(
    config: dict,
    train_collection_path: str | Path | None = None,
    val_collection_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    train_collection = load_activation_collection(
        train_collection_path or config["outputs"]["train_collection"]
    )
    val_collection = load_activation_collection(
        val_collection_path or config["outputs"]["val_collection"]
    )
    output_dir = Path(output_dir or config["outputs"]["vector_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = config.get("steering", {}).get("methods", ["mean_diff"])
    top_k = int(config.get("steering", {}).get("top_k_layers", 3))
    model_size = _configured_model_size(config)
    checkpoint_path = resolve_checkpoint_path(config, model_size=model_size)
    vector_paths = []
    summary_rows = []

    for task_id in config.get("steering", {}).get("task_ids", []):
        task = get_task(task_id)
        records = fit_task_vectors(train_collection, val_collection, task, methods=methods, top_k=top_k)
        for record in records:
            record.metadata["model_size"] = model_size
            record.metadata["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
            vector_path = output_dir / task.task_id / f"{record.layer_id}-{record.method}.safetensors"
            save_direction(vector_path, record)
            vector_paths.append(vector_path)
            summary_rows.append(
                {
                    "task_id": record.task_id,
                    "layer_id": record.layer_id,
                    "method": record.method,
                    "score": record.score,
                    "vector_path": str(vector_path),
                }
            )

    summary_path = output_dir / "vector_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    return vector_paths


def sweep_command(
    config: dict,
    model: torch.nn.Module | None = None,
    datasets: dict | None = None,
    vector_paths: Iterable[str | Path] | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    device = _device_from_config(config)
    model_size = _configured_model_size(config)
    model = (
        build_model_from_config(config, device=device, model_size=model_size)
        if model is None
        else model.to(device)
    )
    datasets = (
        get_datasets(config["data"], split=config.get("sweep_split", "valid"))
        if datasets is None
        else datasets
    )
    output_dir = Path(output_dir or config["outputs"]["sweep_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(config.get("steering", {}).get("backend", "auto"))
    vector_paths = list(vector_paths or Path(config["outputs"]["vector_dir"]).glob("**/*.safetensors"))
    written_paths = []

    for vector_path in vector_paths:
        direction, metadata = load_direction(vector_path)
        auxiliary_features = config.get("steering", {}).get("auxiliary_features")
        results = run_scale_sweep(
            model=model,
            backend=backend,
            datasets=datasets,
            layer_id=metadata["layer_id"],
            direction=direction,
            scales=config.get("steering", {}).get("scales", [-4, -2, -1, 0, 1, 2, 4]),
            target_feature=metadata["feature_name"] or config["steering"]["default_target_feature"],
            auxiliary_features=auxiliary_features,
            batch_size=int(config.get("steering", {}).get("batch_size", 4)),
            max_batches=config.get("steering", {}).get("max_batches"),
            device=device,
        )
        results.insert(0, "task_id", metadata["task_id"])
        results.insert(1, "method", metadata["method"])
        results.insert(2, "model_size", metadata["metadata"].get("model_size", model_size))
        results.insert(3, "vector_path", str(vector_path))
        report_name = f"{metadata['task_id']}-{metadata['layer_id']}-{metadata['method']}.csv"
        report_path = output_dir / report_name
        write_sweep_report(results, report_path)
        written_paths.append(report_path)

    return written_paths


def report_command(
    config: dict,
    sweep_paths: Iterable[str | Path] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    sweep_paths = list(sweep_paths or Path(config["outputs"]["sweep_dir"]).glob("*.csv"))
    combined = aggregate_sweep_reports(sweep_paths)
    output_path = Path(output_path or config["outputs"]["combined_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


def mirror_artifacts_command(config: dict) -> None:
    source = config["outputs"]["artifact_dir"]
    destination = config["outputs"]["remote_artifact_dir"]
    subprocess.run(["rclone", "copy", source, destination], check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activation steering utilities for GPhyT")
    parser.add_argument("command", choices=[
        "download-assets",
        "collect-activations",
        "fit-vectors",
        "sweep",
        "report",
        "mirror-artifacts",
    ])
    parser.add_argument("--config", required=True, help="Path to steering YAML config")
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "download-assets":
        return download_assets_command(config)
    if args.command == "collect-activations":
        return collect_activations_command(config)
    if args.command == "fit-vectors":
        return fit_vectors_command(config)
    if args.command == "sweep":
        return sweep_command(config)
    if args.command == "report":
        return report_command(config)
    if args.command == "mirror-artifacts":
        return mirror_artifacts_command(config)

    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
