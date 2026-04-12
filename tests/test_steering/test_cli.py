from pathlib import Path

import pandas as pd
import torch

from gphyt.steering import cli


class ToyPressureDataset(torch.utils.data.Dataset):
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = float(self.values[index])
        x = torch.zeros(2, 4, 4, 5)
        x[..., 0] = value
        x[..., 3] = value / 2.0
        y = x[-1:, ...].clone()
        return x, y


class ToyBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_dim))

    def forward(self, x):
        return self.proj(x)


class ToySteeringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = torch.nn.Linear(5, 4, bias=False)
        with torch.no_grad():
            self.tokenizer.weight.zero_()
            self.tokenizer.weight[0, 0] = 1.0
            self.tokenizer.weight[1, 3] = 1.0

        self.attention_blocks = torch.nn.Sequential(ToyBlock(4))
        self.detokenizer = torch.nn.Linear(4, 5, bias=False)
        with torch.no_grad():
            self.detokenizer.weight.zero_()
            self.detokenizer.weight[0, 0] = 1.0
            self.detokenizer.weight[3, 1] = 1.0

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.attention_blocks(x)
        return self.detokenizer(x[:, -1:, ...])


def test_cli_smoke_pipeline(tmp_path: Path):
    model = ToySteeringModel()
    config = {
        "steering": {
            "backend": "hook",
            "device": "cpu",
            "task_ids": ["mean_pressure"],
            "methods": ["mean_diff"],
            "scales": [0.0, 1.0],
            "batch_size": 4,
            "top_k_layers": 1,
        },
        "outputs": {
            "train_collection": str(tmp_path / "train_collection.npz"),
            "val_collection": str(tmp_path / "val_collection.npz"),
            "vector_dir": str(tmp_path / "vectors"),
            "sweep_dir": str(tmp_path / "sweeps"),
            "combined_report": str(tmp_path / "combined_report.csv"),
        },
    }

    train_path = cli.collect_activations_command(
        config,
        model=model,
        datasets={"shear_flow": ToyPressureDataset(torch.linspace(-2.0, 2.0, 24))},
        output_path=tmp_path / "train_collection.npz",
    )
    val_path = cli.collect_activations_command(
        config,
        model=model,
        datasets={"shear_flow": ToyPressureDataset(torch.linspace(-1.5, 1.5, 16))},
        output_path=tmp_path / "val_collection.npz",
    )
    vector_paths = cli.fit_vectors_command(
        config,
        train_collection_path=train_path,
        val_collection_path=val_path,
        output_dir=tmp_path / "vectors",
    )
    sweep_paths = cli.sweep_command(
        config,
        model=model,
        datasets={"eval": ToyPressureDataset(torch.linspace(-1.0, 1.0, 12))},
        vector_paths=vector_paths,
        output_dir=tmp_path / "sweeps",
    )
    report_path = cli.report_command(
        config,
        sweep_paths=sweep_paths,
        output_path=tmp_path / "combined_report.csv",
    )

    assert Path(train_path).exists()
    assert Path(val_path).exists()
    assert len(vector_paths) == 1
    assert len(sweep_paths) == 1
    assert report_path.exists()

    report = pd.read_csv(report_path)
    assert set(report["scale"].tolist()) == {0.0, 1.0}
    assert "model_size" in report.columns
    assert "baseline_next_step_mse_mean" in report.columns
    assert "target_shift_mean" in report.columns


def test_cli_checkpoint_resolution_and_well_base_path(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    checkpoint_path = artifact_dir / "checkpoints" / "GPT_S" / "gphyt-S.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")

    config = {
        "assets": {
            "checkpoints": {
                "GPT_S": {
                    "repo_id": "flwi/Physics-Foundation-Model",
                    "filename": "gphyt-S.pth",
                }
            }
        },
        "data": {
            "data_dir": str(tmp_path / "data" / "datasets"),
        },
        "model": {
            "transformer": {
                "model_size": "GPT_S",
            }
        },
        "outputs": {
            "artifact_dir": str(artifact_dir),
        },
    }

    assert cli.resolve_checkpoint_path(config) == checkpoint_path
    assert cli._resolve_well_base_path(config) == tmp_path / "data"
