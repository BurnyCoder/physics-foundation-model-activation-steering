from pathlib import Path

import numpy as np
import torch

from gphyt.steering.eval import (
    ActivationCollection,
    bootstrap_confidence_interval,
    fit_direction,
    fit_task_vectors,
    load_direction,
    save_direction,
)
from gphyt.steering.tasks import get_task


def test_fit_direction_returns_unit_norm_vectors():
    activations = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.1],
            [1.0, -0.1],
            [2.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    for method in ("mean_diff", "logistic", "pca"):
        direction = fit_direction(activations, labels, method)
        assert torch.allclose(torch.linalg.norm(direction), torch.tensor(1.0), atol=1e-6)
        assert direction[0] > 0


def test_fit_task_vectors_and_save(tmp_path: Path):
    rng = np.random.default_rng(0)
    train_feature = np.linspace(-2.0, 2.0, 40)
    val_feature = np.linspace(-1.5, 1.5, 24)

    train_collection = ActivationCollection(
        layer_activations={
            "block_out:0": np.stack([train_feature, rng.normal(scale=0.05, size=40)], axis=1),
            "block_out:1": rng.normal(size=(40, 2)),
        },
        feature_values={"mean_pressure": train_feature},
        dataset_names=np.asarray(["shear_flow"] * 40),
    )
    val_collection = ActivationCollection(
        layer_activations={
            "block_out:0": np.stack([val_feature, rng.normal(scale=0.05, size=24)], axis=1),
            "block_out:1": rng.normal(size=(24, 2)),
        },
        feature_values={"mean_pressure": val_feature},
        dataset_names=np.asarray(["shear_flow"] * 24),
    )

    records = fit_task_vectors(
        train_collection,
        val_collection,
        get_task("mean_pressure"),
        methods=("mean_diff", "logistic", "pca"),
        top_k=1,
    )

    assert len(records) == 3
    assert all(record.layer_id == "block_out:0" for record in records)
    assert all(record.score > 0.95 for record in records)

    vector_path = tmp_path / "direction.safetensors"
    save_direction(vector_path, records[0])
    direction, metadata = load_direction(vector_path)

    assert direction.shape == (2,)
    assert metadata["task_id"] == "mean_pressure"


def test_bootstrap_confidence_interval_is_ordered():
    low, high = bootstrap_confidence_interval(np.array([0.0, 1.0, 2.0, 3.0]))
    assert low <= high
