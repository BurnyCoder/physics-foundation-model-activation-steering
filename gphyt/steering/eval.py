"""Activation collection, vector fitting, and sweep evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file
from scipy.stats import bootstrap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

from gphyt.steering.adapters import SteeringBackend, list_layer_ids
from gphyt.steering.features import (
    FEATURE_NAMES,
    compute_feature_batch,
    fit_quartile_split,
    fit_zscore_stats,
    quartile_contrast_labels,
    zscore,
)
from gphyt.steering.tasks import SteeringTask


@dataclass
class ActivationCollection:
    layer_activations: dict[str, np.ndarray]
    feature_values: dict[str, np.ndarray]
    dataset_names: np.ndarray


@dataclass
class DirectionRecord:
    layer_id: str
    method: str
    direction: torch.Tensor
    score: float
    task_id: str
    feature_name: str | None
    train_size: int
    val_size: int
    metadata: dict


def mean_pool_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim < 2:
        raise ValueError(f"Expected activation with batch dimension, got {activation.shape}")
    reduce_dims = tuple(range(1, activation.ndim - 1))
    if not reduce_dims:
        return activation
    return activation.mean(dim=reduce_dims)


def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _make_dataloader(
    dataset_or_loader: torch.utils.data.Dataset | DataLoader,
    batch_size: int,
) -> DataLoader:
    if isinstance(dataset_or_loader, DataLoader):
        return dataset_or_loader
    return DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)


def _raw_feature_dataloader(
    dataset_or_loader: torch.utils.data.Dataset | DataLoader,
    batch_size: int,
) -> DataLoader | None:
    dataset = (
        dataset_or_loader.dataset
        if isinstance(dataset_or_loader, DataLoader)
        else dataset_or_loader
    )
    if not hasattr(dataset, "copy"):
        return None
    dataset_config = getattr(dataset, "config", {})
    if not dataset_config.get("use_normalization", False):
        return None
    raw_dataset = dataset.copy({"use_normalization": False})
    if raw_dataset is None:
        return None
    return DataLoader(raw_dataset, batch_size=batch_size, shuffle=False)


def _denormalize_prediction(
    dataset_or_loader: torch.utils.data.Dataset | DataLoader,
    fields: torch.Tensor,
) -> torch.Tensor:
    dataset = (
        dataset_or_loader.dataset
        if isinstance(dataset_or_loader, DataLoader)
        else dataset_or_loader
    )
    if not getattr(dataset, "config", {}).get("use_normalization", False):
        return fields
    if hasattr(dataset, "denormalize_variable_fields"):
        return dataset.denormalize_variable_fields(fields)
    norm = getattr(dataset, "norm", None)
    if norm is None or not hasattr(norm, "denormalize_flattened"):
        return fields
    return norm.denormalize_flattened(fields, "variable")


def collect_activation_dataset(
    model: torch.nn.Module,
    backend: SteeringBackend,
    datasets: dict[str, torch.utils.data.Dataset | DataLoader],
    batch_size: int = 4,
    feature_names: Iterable[str] | None = None,
    layer_ids: Iterable[str] | None = None,
    max_batches: int | None = None,
    device: torch.device | None = None,
) -> ActivationCollection:
    feature_names = tuple(FEATURE_NAMES if feature_names is None else feature_names)
    layer_ids = list(list_layer_ids(model) if layer_ids is None else layer_ids)
    device = device or next(model.parameters()).device

    pooled_by_layer = {layer_id: [] for layer_id in layer_ids}
    feature_values = {feature_name: [] for feature_name in feature_names}
    dataset_names: list[str] = []

    for dataset_name, dataset in datasets.items():
        dataloader = _make_dataloader(dataset, batch_size=batch_size)
        raw_feature_loader = _raw_feature_dataloader(dataset, batch_size=batch_size)
        raw_iter = iter(raw_feature_loader) if raw_feature_loader is not None else None

        for batch_idx, batch in enumerate(dataloader):
            x = batch[0].to(device)
            if raw_iter is not None:
                raw_batch = next(raw_iter)
                feature_source = raw_batch[0].detach().cpu()
            else:
                feature_source = batch[0].detach().cpu()
            batch_size_actual = x.shape[0]

            for layer_id in layer_ids:
                activation = backend.collect_activation(model, layer_id, x)
                pooled = mean_pool_activation(activation).detach().cpu().numpy()
                pooled_by_layer[layer_id].append(pooled)

            for feature_name in feature_names:
                feature_values[feature_name].append(
                    compute_feature_batch(feature_source, feature_name).detach().cpu().numpy()
                )

            dataset_names.extend([dataset_name] * batch_size_actual)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    return ActivationCollection(
        layer_activations={
            layer_id: np.concatenate(values, axis=0) if values else np.zeros((0, 0))
            for layer_id, values in pooled_by_layer.items()
        },
        feature_values={
            feature_name: np.concatenate(values, axis=0) if values else np.zeros((0,))
            for feature_name, values in feature_values.items()
        },
        dataset_names=np.asarray(dataset_names),
    )


def save_activation_collection(path: str | Path, collection: ActivationCollection) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "layer_ids": list(collection.layer_activations),
        "feature_names": list(collection.feature_values),
    }
    np.savez_compressed(
        path,
        dataset_names=collection.dataset_names,
        **{f"layer::{key}": value for key, value in collection.layer_activations.items()},
        **{f"feature::{key}": value for key, value in collection.feature_values.items()},
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return path


def load_activation_collection(path: str | Path) -> ActivationCollection:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(path.with_suffix(".json").read_text())
    return ActivationCollection(
        layer_activations={
            layer_id: data[f"layer::{layer_id}"] for layer_id in metadata["layer_ids"]
        },
        feature_values={
            feature_name: data[f"feature::{feature_name}"]
            for feature_name in metadata["feature_names"]
        },
        dataset_names=data["dataset_names"],
    )


def fit_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    method: str,
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(f"Expected 2D activations, got {activations.shape}")
    positive = activations[labels == 1]
    negative = activations[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Need both positive and negative labels to fit direction")

    if method == "mean_diff":
        direction = positive.mean(axis=0) - negative.mean(axis=0)
    elif method == "logistic":
        probe = LogisticRegression(max_iter=500)
        probe.fit(activations, labels)
        direction = probe.coef_[0]
    elif method == "pca":
        pair_count = min(len(positive), len(negative))
        diffs = positive[:pair_count] - negative[:pair_count]
        pca = PCA(n_components=1)
        pca.fit(diffs)
        direction = pca.components_[0]
    else:
        raise ValueError(f"Unknown fit method {method}")

    direction = torch.as_tensor(direction, dtype=torch.float32)
    norm = torch.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Fitted zero-norm direction")
    return direction / norm


def _probe_auc(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> float:
    probe = LogisticRegression(max_iter=500)
    probe.fit(train_x, train_y)
    val_scores = probe.predict_proba(val_x)[:, 1]
    return float(roc_auc_score(val_y, val_scores))


def _per_dataset_zscore(
    values: np.ndarray,
    dataset_names: np.ndarray,
    stats_map: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    values = np.asarray(values, dtype=np.float64)
    dataset_names = np.asarray(dataset_names)
    if stats_map is None:
        stats_map = {}
        for dataset_name in np.unique(dataset_names):
            stats_map[str(dataset_name)] = fit_zscore_stats(
                values[dataset_names == dataset_name]
            )

    z_values = np.zeros_like(values, dtype=np.float64)
    for dataset_name, stats in stats_map.items():
        mask = dataset_names == dataset_name
        if np.any(mask):
            z_values[mask] = zscore(values[mask], stats)
    return z_values, stats_map


def _per_dataset_quartile_labels(
    values: np.ndarray,
    dataset_names: np.ndarray,
    split_map: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    values = np.asarray(values, dtype=np.float64)
    dataset_names = np.asarray(dataset_names)
    if split_map is None:
        split_map = {}
        for dataset_name in np.unique(dataset_names):
            split_map[str(dataset_name)] = fit_quartile_split(
                values[dataset_names == dataset_name]
            )

    labels = np.full(values.shape[0], -1, dtype=np.int64)
    for dataset_name, split in split_map.items():
        mask = dataset_names == dataset_name
        if np.any(mask):
            dataset_labels, _ = quartile_contrast_labels(values[mask], split)
            labels[mask] = dataset_labels
    return labels, labels >= 0, split_map


def _feature_task_masks(
    train_collection: ActivationCollection,
    val_collection: ActivationCollection,
    task: SteeringTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    train_values = train_collection.feature_values[task.feature_name]
    train_z, stats_map = _per_dataset_zscore(
        train_values,
        train_collection.dataset_names,
    )
    train_labels, train_mask, split_map = _per_dataset_quartile_labels(
        train_z,
        train_collection.dataset_names,
    )

    val_z, _ = _per_dataset_zscore(
        val_collection.feature_values[task.feature_name],
        val_collection.dataset_names,
        stats_map=stats_map,
    )
    val_labels, val_mask, _ = _per_dataset_quartile_labels(
        val_z,
        val_collection.dataset_names,
        split_map=split_map,
    )
    return train_labels, train_mask, val_labels, val_mask, stats_map, split_map


def _regime_task_masks(
    train_collection: ActivationCollection,
    val_collection: ActivationCollection,
    task: SteeringTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_labels = np.full(train_collection.dataset_names.shape, -1, dtype=np.int64)
    val_labels = np.full(val_collection.dataset_names.shape, -1, dtype=np.int64)

    train_names = np.asarray(train_collection.dataset_names)
    val_names = np.asarray(val_collection.dataset_names)

    train_labels[np.isin(train_names, task.negative_datasets)] = 0
    train_labels[np.isin(train_names, task.positive_datasets)] = 1
    val_labels[np.isin(val_names, task.negative_datasets)] = 0
    val_labels[np.isin(val_names, task.positive_datasets)] = 1

    return train_labels, train_labels >= 0, val_labels, val_labels >= 0


def fit_task_vectors(
    train_collection: ActivationCollection,
    val_collection: ActivationCollection,
    task: SteeringTask,
    methods: Iterable[str] = ("mean_diff",),
    top_k: int = 3,
) -> list[DirectionRecord]:
    methods = tuple(methods)
    if task.task_type == "feature":
        train_labels, train_mask, val_labels, val_mask, stats, split = _feature_task_masks(
            train_collection, val_collection, task
        )
        task_metadata = {"zscore_stats": stats, "quartile_split": split}
    else:
        train_labels, train_mask, val_labels, val_mask = _regime_task_masks(
            train_collection, val_collection, task
        )
        task_metadata = {}

    ranked_layers = []
    for layer_id, train_acts in train_collection.layer_activations.items():
        val_acts = val_collection.layer_activations[layer_id]
        train_x = train_acts[train_mask]
        train_y = train_labels[train_mask]
        val_x = val_acts[val_mask]
        val_y = val_labels[val_mask]
        if train_x.shape[0] < 4 or val_x.shape[0] < 2:
            continue
        if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            continue
        score = _probe_auc(train_x, train_y, val_x, val_y)
        ranked_layers.append((layer_id, score, train_x, train_y, val_x, val_y))

    ranked_layers.sort(key=lambda item: item[1], reverse=True)
    selected_layers = ranked_layers[:top_k]

    records: list[DirectionRecord] = []
    for layer_id, score, train_x, train_y, _val_x, _val_y in selected_layers:
        for method in methods:
            direction = fit_direction(train_x, train_y, method)
            records.append(
                DirectionRecord(
                    layer_id=layer_id,
                    method=method,
                    direction=direction,
                    score=float(score),
                    task_id=task.task_id,
                    feature_name=task.feature_name,
                    train_size=int(train_x.shape[0]),
                    val_size=int(val_collection.layer_activations[layer_id][val_mask].shape[0]),
                    metadata=task_metadata.copy(),
                )
            )
    return records


def save_direction(path: str | Path, record: DirectionRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"direction": record.direction.detach().cpu()}, str(path))
    metadata = asdict(record)
    metadata["direction"] = None
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return path


def load_direction(path: str | Path) -> tuple[torch.Tensor, dict]:
    path = Path(path)
    tensors = load_file(str(path))
    metadata = json.loads(path.with_suffix(".json").read_text())
    return tensors["direction"], metadata


def bootstrap_confidence_interval(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        scalar = float(values.mean()) if values.size else 0.0
        return scalar, scalar
    result = bootstrap(
        (values,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=1000,
        method="basic",
        random_state=0,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1).detach().cpu().numpy()


def run_scale_sweep(
    model: torch.nn.Module,
    backend: SteeringBackend,
    datasets: dict[str, torch.utils.data.Dataset | DataLoader],
    layer_id: str,
    direction: torch.Tensor,
    scales: Iterable[float],
    target_feature: str,
    auxiliary_features: Iterable[str] | None = None,
    batch_size: int = 4,
    max_batches: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    scales = tuple(scales)
    auxiliary_features = tuple(
        feature_name
        for feature_name in (FEATURE_NAMES if auxiliary_features is None else auxiliary_features)
        if feature_name != target_feature
    )

    baseline_target_samples: list[float] = []
    baseline_aux_samples = {feature_name: [] for feature_name in auxiliary_features}
    rows = []

    cached_batches = []
    for dataset_name, dataset in datasets.items():
        dataloader = _make_dataloader(dataset, batch_size=batch_size)
        for batch_idx, batch in enumerate(dataloader):
            x = batch[0].to(device)
            y = batch[1].to(device) if len(batch) > 1 else None
            baseline = model(x).detach().cpu()
            baseline_denorm = _denormalize_prediction(dataset, baseline)
            baseline_target = compute_feature_batch(
                baseline_denorm,
                target_feature,
            ).detach().cpu().numpy()
            baseline_target_samples.extend(baseline_target.tolist())
            for feature_name in auxiliary_features:
                baseline_aux = compute_feature_batch(
                    baseline_denorm,
                    feature_name,
                ).detach().cpu().numpy()
                baseline_aux_samples[feature_name].extend(baseline_aux.tolist())
            cached_batches.append((dataset_name, dataset, x, y, baseline, baseline_denorm))
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    baseline_target_std = max(np.std(baseline_target_samples), 1e-6)
    aux_stds = {
        feature_name: max(np.std(values), 1e-6)
        for feature_name, values in baseline_aux_samples.items()
    }

    for scale in scales:
        target_shift_samples = []
        off_target_samples = []
        mse_delta_samples = []
        baseline_loss_samples = []
        steered_loss_samples = []
        nonfinite_samples = []

        for dataset_name, dataset, x, y, baseline, baseline_denorm in cached_batches:
            steered = backend.run_with_direction(model, layer_id, x, direction, scale).detach().cpu()
            steered_denorm = _denormalize_prediction(dataset, steered)
            finite_mask = torch.isfinite(steered_denorm).reshape(steered_denorm.shape[0], -1).all(dim=1)
            finite_mask_np = finite_mask.detach().cpu().numpy()
            nonfinite_samples.extend((~finite_mask_np).astype(np.float64).tolist())

            baseline_target = compute_feature_batch(
                baseline_denorm,
                target_feature,
            ).detach().cpu().numpy()
            steered_target = compute_feature_batch(
                torch.nan_to_num(steered_denorm),
                target_feature,
            ).detach().cpu().numpy()
            target_shift = (steered_target - baseline_target) / baseline_target_std
            target_shift_samples.extend(target_shift[finite_mask_np].tolist())

            if auxiliary_features:
                aux_drifts = []
                for feature_name in auxiliary_features:
                    baseline_aux = compute_feature_batch(
                        baseline_denorm,
                        feature_name,
                    ).detach().cpu().numpy()
                    steered_aux = compute_feature_batch(
                        torch.nan_to_num(steered_denorm),
                        feature_name,
                    ).detach().cpu().numpy()
                    aux_drifts.append(np.abs(steered_aux - baseline_aux) / aux_stds[feature_name])
                aux_drift = np.stack(aux_drifts, axis=1).mean(axis=1)
                off_target_samples.extend(aux_drift[finite_mask_np].tolist())
            else:
                off_target_samples.extend(np.zeros_like(target_shift[finite_mask_np]))

            if y is not None:
                baseline_loss = _per_sample_mse(baseline, y.detach().cpu())
                steered_loss = _per_sample_mse(steered, y.detach().cpu())
                baseline_loss_samples.extend(baseline_loss.tolist())
                steered_loss_samples.extend(steered_loss.tolist())
                mse_delta_samples.extend((steered_loss - baseline_loss).tolist())

        target_shift_array = np.asarray(target_shift_samples, dtype=np.float64)
        ci_low, ci_high = bootstrap_confidence_interval(target_shift_array)
        rows.append(
            {
                "layer_id": layer_id,
                "scale": float(scale),
                "target_feature": target_feature,
                "target_shift_mean": float(np.mean(target_shift_array)) if target_shift_samples else np.nan,
                "target_shift_ci_low": ci_low,
                "target_shift_ci_high": ci_high,
                "off_target_drift_mean": float(np.mean(off_target_samples)) if off_target_samples else np.nan,
                "baseline_next_step_mse_mean": float(np.mean(baseline_loss_samples)) if baseline_loss_samples else np.nan,
                "steered_next_step_mse_mean": float(np.mean(steered_loss_samples)) if steered_loss_samples else np.nan,
                "next_step_mse_delta_mean": float(np.mean(mse_delta_samples)) if mse_delta_samples else np.nan,
                "nonfinite_fraction": float(np.mean(nonfinite_samples)) if nonfinite_samples else 0.0,
                "num_samples": int(len(target_shift_samples)),
            }
        )

    return pd.DataFrame(rows)


def write_sweep_report(results: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def aggregate_sweep_reports(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
