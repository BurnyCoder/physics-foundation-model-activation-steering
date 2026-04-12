"""Physics-aware feature extraction for activation steering."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


FIELD_INDEX = {
    "pressure": 0,
    "density": 1,
    "temperature": 2,
    "velocity_x": 3,
    "velocity_y": 4,
}

FEATURE_NAMES = (
    "mean_velocity_magnitude",
    "high_velocity_tail",
    "mean_pressure",
    "pressure_contrast",
    "enstrophy_proxy",
    "divergence_magnitude",
    "vortex_intermittency",
    "mean_density",
    "density_contrast",
    "shock_score",
    "stratification_score",
)

FEATURE_REQUIRED_FIELDS = {
    "mean_velocity_magnitude": ("velocity_x", "velocity_y"),
    "high_velocity_tail": ("velocity_x", "velocity_y"),
    "mean_pressure": ("pressure",),
    "pressure_contrast": ("pressure",),
    "enstrophy_proxy": ("velocity_x", "velocity_y"),
    "divergence_magnitude": ("velocity_x", "velocity_y"),
    "vortex_intermittency": ("velocity_x", "velocity_y"),
    "mean_density": ("density",),
    "density_contrast": ("density",),
    "shock_score": ("density", "velocity_x", "velocity_y"),
    "stratification_score": ("density",),
}


def _ensure_batched_fields(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(0)
    if x.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")
    return x


def _last_frame(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_batched_fields(x)
    return x[:, -1, ...]


def _diff_axis(field: torch.Tensor, dim: int) -> torch.Tensor:
    if field.shape[dim] < 2:
        return torch.zeros_like(field)

    out = torch.zeros_like(field)

    interior = [slice(None)] * field.ndim
    prev_slice = [slice(None)] * field.ndim
    next_slice = [slice(None)] * field.ndim
    interior[dim] = slice(1, -1)
    prev_slice[dim] = slice(0, -2)
    next_slice[dim] = slice(2, None)
    out[tuple(interior)] = (
        field[tuple(next_slice)] - field[tuple(prev_slice)]
    ) / 2.0

    first = [slice(None)] * field.ndim
    second = [slice(None)] * field.ndim
    first[dim] = 0
    second[dim] = 1
    out[tuple(first)] = field[tuple(second)] - field[tuple(first)]

    last = [slice(None)] * field.ndim
    penultimate = [slice(None)] * field.ndim
    last[dim] = -1
    penultimate[dim] = -2
    out[tuple(last)] = field[tuple(last)] - field[tuple(penultimate)]

    return out


def _flatten_samples(field: torch.Tensor) -> torch.Tensor:
    return field.reshape(field.shape[0], -1)


def _percentile(field: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(_flatten_samples(field), q, dim=1)


def _field(frame: torch.Tensor, field_name: str) -> torch.Tensor:
    return frame[..., FIELD_INDEX[field_name]]


def required_fields_for_feature(feature_name: str) -> tuple[str, ...]:
    if feature_name not in FEATURE_REQUIRED_FIELDS:
        raise KeyError(f"Unknown feature {feature_name}")
    return FEATURE_REQUIRED_FIELDS[feature_name]


def compute_feature_batch(x: torch.Tensor, feature_name: str) -> torch.Tensor:
    frame = _last_frame(x)
    frame = frame.to(dtype=torch.float32)

    if feature_name == "mean_velocity_magnitude":
        vx = _field(frame, "velocity_x")
        vy = _field(frame, "velocity_y")
        speed = torch.sqrt(vx.square() + vy.square())
        return speed.mean(dim=(1, 2))

    if feature_name == "high_velocity_tail":
        vx = _field(frame, "velocity_x")
        vy = _field(frame, "velocity_y")
        speed = torch.sqrt(vx.square() + vy.square())
        flat = _flatten_samples(speed)
        k = max(1, math.ceil(flat.shape[1] * 0.1))
        topk = torch.topk(flat, k=k, dim=1).values
        return topk.mean(dim=1)

    if feature_name == "mean_pressure":
        pressure = _field(frame, "pressure")
        return pressure.mean(dim=(1, 2))

    if feature_name == "pressure_contrast":
        pressure = _field(frame, "pressure")
        return _percentile(pressure, 0.95) - _percentile(pressure, 0.05)

    if feature_name == "enstrophy_proxy":
        vx = _field(frame, "velocity_x")
        vy = _field(frame, "velocity_y")
        curl = _diff_axis(vy, 1) - _diff_axis(vx, 2)
        return curl.square().mean(dim=(1, 2))

    if feature_name == "divergence_magnitude":
        vx = _field(frame, "velocity_x")
        vy = _field(frame, "velocity_y")
        divergence = _diff_axis(vx, 1) + _diff_axis(vy, 2)
        return divergence.abs().mean(dim=(1, 2))

    if feature_name == "vortex_intermittency":
        vx = _field(frame, "velocity_x")
        vy = _field(frame, "velocity_y")
        curl = (_diff_axis(vy, 1) - _diff_axis(vx, 2)).abs()
        return _percentile(curl, 0.95) / (curl.mean(dim=(1, 2)) + 1e-6)

    if feature_name == "mean_density":
        density = _field(frame, "density")
        return density.mean(dim=(1, 2))

    if feature_name == "density_contrast":
        density = _field(frame, "density")
        return _percentile(density, 0.95) - _percentile(density, 0.05)

    if feature_name == "shock_score":
        density = _field(frame, "density")
        vx = _field(frame, "velocity_x")
        vy = _field(frame, "velocity_y")
        divergence = _diff_axis(vx, 1) + _diff_axis(vy, 2)
        compression = torch.relu(-divergence)
        density_grad = torch.sqrt(_diff_axis(density, 1).square() + _diff_axis(density, 2).square())
        return (compression * density_grad).mean(dim=(1, 2))

    if feature_name == "stratification_score":
        density = _field(frame, "density")
        quarter = max(1, density.shape[1] // 4)
        top_mean = density[:, :quarter, :].mean(dim=(1, 2))
        bottom_mean = density[:, -quarter:, :].mean(dim=(1, 2))
        return (top_mean - bottom_mean).abs()

    raise KeyError(f"Unknown feature {feature_name}")


def compute_features(
    x: torch.Tensor,
    feature_names: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    feature_names = FEATURE_NAMES if feature_names is None else tuple(feature_names)
    return {
        feature_name: compute_feature_batch(x, feature_name)
        for feature_name in feature_names
    }


def fit_zscore_stats(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std < 1e-12:
        std = 1.0
    return {"mean": float(values.mean()), "std": std}


def zscore(
    values: torch.Tensor | np.ndarray,
    stats: dict[str, float],
) -> torch.Tensor | np.ndarray:
    if isinstance(values, torch.Tensor):
        return (values - stats["mean"]) / stats["std"]
    values = np.asarray(values, dtype=np.float64)
    return (values - stats["mean"]) / stats["std"]


def fit_quartile_split(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    return {
        "low": float(np.quantile(values, 0.25)),
        "high": float(np.quantile(values, 0.75)),
    }


def quartile_contrast_labels(
    values: torch.Tensor | np.ndarray,
    split: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    split = fit_quartile_split(values) if split is None else split
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    labels[values <= split["low"]] = 0
    labels[values >= split["high"]] = 1
    mask = labels >= 0
    return labels, mask
