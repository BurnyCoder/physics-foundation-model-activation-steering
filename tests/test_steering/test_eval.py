from pathlib import Path

import numpy as np
import torch

from gphyt.steering.adapters import SteeringBackend
from gphyt.steering.eval import (
    ActivationCollection,
    bootstrap_confidence_interval,
    fit_direction,
    fit_task_vectors,
    load_direction,
    run_rollout_sweep,
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


class DummyRolloutBackend(SteeringBackend):
    name = "dummy"

    def collect_activation(self, model, layer_id, x):
        return x[:, -1, ...]

    def run_with_direction(self, model, layer_id, x, direction, scale):
        direction = direction.to(device=x.device, dtype=x.dtype)
        delta = direction.view(1, 1, 1, 1, -1) * float(scale)
        return model(x) + delta

    def run_with_noop(self, model, layer_id, x):
        return model(x)


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: list[torch.Tensor],
        *,
        full_trajectory_mode: bool = False,
        max_rollout_steps: int = 4,
    ):
        self.trajectories = trajectories
        self.dataset_name = "rayleigh_benard"
        self.config = {
            "use_normalization": False,
            "full_trajectory_mode": full_trajectory_mode,
            "max_rollout_steps": max_rollout_steps,
        }
        self.full_trajectory_mode = full_trajectory_mode
        self.max_rollout_steps = max_rollout_steps

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        traj = self.trajectories[index]
        x = traj[:4].clone()
        y = traj[4 : 4 + self.max_rollout_steps].clone()
        return x, y

    def copy(self, overwrites=None):
        overwrites = overwrites or {}
        return RolloutDataset(
            self.trajectories,
            full_trajectory_mode=overwrites.get("full_trajectory_mode", self.full_trajectory_mode),
            max_rollout_steps=overwrites.get("max_rollout_steps", self.max_rollout_steps),
        )


def test_run_rollout_sweep_returns_rollout_metrics():
    base_traj = torch.zeros(8, 6, 6, 5)
    trajectories = [base_traj.clone(), base_traj.clone()]
    dataset = RolloutDataset(trajectories, max_rollout_steps=4)

    class LastFrameModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return x[:, -1:, ...] + self.bias

    model = LastFrameModel()
    backend = DummyRolloutBackend()
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    results = run_rollout_sweep(
        model=model,
        backend=backend,
        datasets={"rayleigh_benard": dataset},
        layer_id="block_out:0",
        direction=direction,
        scales=[0.0, 1.0],
        target_feature="mean_pressure",
        auxiliary_features=("mean_velocity_magnitude",),
        num_timesteps=4,
        num_samples=2,
        device=torch.device("cpu"),
    )

    assert {"rollout_target_shift_mean", "rollout_nonfinite_fraction"}.issubset(results.columns)
    zero_scale = results[results["scale"] == 0.0].iloc[0]
    pos_scale = results[results["scale"] == 1.0].iloc[0]

    assert zero_scale["rollout_target_shift_mean"] == 0.0
    assert zero_scale["rollout_nonfinite_fraction"] == 0.0
    assert zero_scale["rollout_valid_steps_mean"] == 4.0
    assert pos_scale["rollout_target_shift_mean"] > 0.0
    assert pos_scale["steered_rollout_mse_final_mean"] > zero_scale["steered_rollout_mse_final_mean"]
