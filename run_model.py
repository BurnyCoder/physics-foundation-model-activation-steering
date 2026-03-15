"""
Run ALL GPhyT model sizes (S/M/L/XL) on real shear_flow data from The Well.

Downloads all checkpoints, runs autoregressive rollouts, and generates
one big comparison GIF: rows = fields, columns = GT + S + M + L + XL.
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from gphyt.models.transformer.model import get_model

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_SIZES = ["GPT_S", "GPT_M", "GPT_L", "GPT_XL"]

N_STEPS_INPUT = 4
OUT_SHAPE = (256, 128)
INPUT_CHANNELS = 5

ACTIVE_FIELDS = (0, 3, 4)  # shear_flow: pressure, vel_x, vel_y

HDF5_PATH = Path("data/datasets/shear_flow/data/test/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
TRAJ_IDX = 0
START_T = 50
NUM_ROLLOUT_STEPS = 30

# Fields to visualize (channel_idx, name, colormap)
VIS_FIELDS = [
    (0, "Pressure", "inferno"),
    (3, "Velocity X", "coolwarm"),
    (4, "Velocity Y", "coolwarm"),
]

def make_model_config(model_size):
    return {
        "img_size": (N_STEPS_INPUT, *OUT_SHAPE),
        "transformer": {
            "input_channels": INPUT_CHANNELS,
            "model_size": model_size,
            "att_mode": "full",
            "dropout": 0.0,
            "pos_enc_mode": "absolute",
            "patch_size": [1, 16, 16],
            "stochastic_depth_rate": 0.0,
            "use_derivatives": True,
            "integrator": "Euler",
        },
        "tokenizer": {
            "tokenizer_mode": "linear",
            "detokenizer_mode": "linear",
            "tokenizer_overlap": 0,
            "detokenizer_overlap": 0,
        },
    }

# ── Load real data ─────────────────────────────────────────────────────────
print(f"Loading data from {HDF5_PATH}...")
with h5py.File(HDF5_PATH, "r") as f:
    pressure = f["t0_fields/pressure"][TRAJ_IDX]
    velocity = f["t1_fields/velocity"][TRAJ_IDX]

n_steps = pressure.shape[0]
H_raw, W_raw = pressure.shape[1], pressure.shape[2]
print(f"Raw data: {n_steps} timesteps, {H_raw}x{W_raw}")

full_data = np.zeros((n_steps, H_raw, W_raw, INPUT_CHANNELS), dtype=np.float32)
full_data[..., 0] = pressure
full_data[..., 3] = velocity[..., 0]
full_data[..., 4] = velocity[..., 1]

# Instance normalize
norm_window = full_data[START_T : START_T + N_STEPS_INPUT]
for c in ACTIVE_FIELDS:
    mean = norm_window[..., c].mean()
    std = norm_window[..., c].std() + 1e-6
    full_data[..., c] = (full_data[..., c] - mean) / std

full_data_t = torch.from_numpy(full_data)
full_data_t = full_data_t.permute(0, 3, 1, 2)
full_data_t = F.interpolate(full_data_t, size=OUT_SHAPE, mode="bilinear", align_corners=False)
full_data_t = full_data_t.permute(0, 2, 3, 1)

end_t = START_T + N_STEPS_INPUT + NUM_ROLLOUT_STEPS
x = full_data_t[START_T : START_T + N_STEPS_INPUT].unsqueeze(0)
gt = full_data_t[START_T + N_STEPS_INPUT : end_t]

print(f"Input: t={START_T}-{START_T + N_STEPS_INPUT - 1}, "
      f"GT: t={START_T + N_STEPS_INPUT}-{end_t - 1}")

# ── Run all model sizes ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

all_rollouts = {}  # model_size → (num_steps, H, W, 5) numpy

for model_size in MODEL_SIZES:
    size_letter = model_size.split("_")[1]
    hf_filename = f"gphyt-{size_letter}.pth"

    print(f"── {model_size} ──")
    print(f"  Downloading {hf_filename}...")
    checkpoint_path = hf_hub_download(
        repo_id="flwi/Physics-Foundation-Model",
        filename=hf_filename,
    )

    model = get_model(make_model_config(model_size))
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {num_params:,} parameters")

    # Rollout
    rollout_outputs = []
    current_input = x.to(device).clone()
    for step in range(NUM_ROLLOUT_STEPS):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = model(current_input)
        rollout_outputs.append(pred.cpu().float())
        current_input = torch.cat([current_input[:, 1:, ...], pred], dim=1)

    rollout = torch.cat(rollout_outputs, dim=1)[0].numpy()  # (num_steps, H, W, 5)
    all_rollouts[model_size] = rollout
    print(f"  Rollout done ({NUM_ROLLOUT_STEPS} steps)")

    # Free GPU memory
    del model, state_dict
    torch.cuda.empty_cache()

# ── Generate big comparison GIF ────────────────────────────────────────────
# Layout per frame:
#   Rows: pressure, velocity_x, velocity_y, |velocity|
#   Columns: GT, GPT_S, GPT_M, GPT_L, GPT_XL

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

gt_np = gt.numpy()

# Add velocity magnitude as a virtual field
VIS_FIELDS_FULL = VIS_FIELDS + [(-1, "|Velocity|", "viridis")]  # -1 = velocity magnitude

col_labels = ["Ground Truth"] + [s.replace("GPT_", "GPT-") for s in MODEL_SIZES]
n_rows = len(VIS_FIELDS_FULL)
n_cols = len(col_labels)

# Precompute color ranges per field across all models and GT
color_ranges = {}
for ch, name, _ in VIS_FIELDS:
    vals = [gt_np[..., ch].ravel()]
    for ms in MODEL_SIZES:
        vals.append(all_rollouts[ms][..., ch].ravel())
    all_vals = np.concatenate(vals)
    color_ranges[ch] = np.nanpercentile(all_vals, [2, 98])

# Velocity magnitude range
vel_vals = []
for data in [gt_np] + [all_rollouts[ms] for ms in MODEL_SIZES]:
    vel_vals.append(np.sqrt(data[..., 3]**2 + data[..., 4]**2).ravel())
color_ranges[-1] = np.nanpercentile(np.concatenate(vel_vals), [2, 98])


def get_field_data(data, ch):
    """Get field data; ch=-1 means velocity magnitude."""
    if ch == -1:
        return np.sqrt(data[..., 3]**2 + data[..., 4]**2)
    return data[..., ch]


print("\nGenerating comparison GIF...")
frames = []
for t in range(NUM_ROLLOUT_STEPS):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows))

    for row, (ch, name, cmap_name) in enumerate(VIS_FIELDS_FULL):
        vmin, vmax = color_ranges[ch]

        # Column 0: Ground truth
        ax = axes[row, 0]
        ax.imshow(get_field_data(gt_np, ch)[t].T, cmap=cmap_name,
                  vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0 and row == 0:
            pass  # titles set below
        ax.set_ylabel(name, fontsize=9, fontweight="bold")

        # Columns 1-4: model predictions
        for col_idx, ms in enumerate(MODEL_SIZES, start=1):
            ax = axes[row, col_idx]
            ax.imshow(get_field_data(all_rollouts[ms], ch)[t].T, cmap=cmap_name,
                      vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])

    # Column titles (top row only)
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=10, fontweight="bold")

    fig.suptitle(f"Shear Flow — Autoregressive Rollout  t+{t + 1}/{NUM_ROLLOUT_STEPS}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    frames.append(frame.copy())
    plt.close(fig)

    if (t + 1) % 10 == 0:
        print(f"  Frame {t + 1}/{NUM_ROLLOUT_STEPS}")

gif_path = output_dir / "all_models_comparison.gif"
iio.imwrite(gif_path, frames, loop=0, duration=250)  # 4 fps
print(f"\nSaved: {gif_path}")

# ── Also save a static comparison at key timesteps ─────────────────────────
key_steps = [0, 4, 9, 19, 29]  # t+1, t+5, t+10, t+20, t+30
n_key = len(key_steps)

for ch, name, cmap_name in VIS_FIELDS_FULL:
    vmin, vmax = color_ranges[ch]
    fig, axes = plt.subplots(n_key, n_cols, figsize=(2.5 * n_cols, 2.2 * n_key), squeeze=False)

    for row_idx, t in enumerate(key_steps):
        # GT
        ax = axes[row_idx][0]
        ax.imshow(get_field_data(gt_np, ch)[t].T, cmap=cmap_name,
                  vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"t+{t + 1}", fontsize=9, fontweight="bold")

        # Models
        for col_idx, ms in enumerate(MODEL_SIZES, start=1):
            ax = axes[row_idx][col_idx]
            ax.imshow(get_field_data(all_rollouts[ms], ch)[t].T, cmap=cmap_name,
                      vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])

    for col_idx, label in enumerate(col_labels):
        axes[0][col_idx].set_title(label, fontsize=9, fontweight="bold")

    safe_name = name.replace("|", "").replace(" ", "_").lower()
    fig.suptitle(f"{name} — All Model Sizes", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    static_path = output_dir / f"comparison_{safe_name}.png"
    fig.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {static_path}")

# ── Save all rollouts ──────────────────────────────────────────────────────
torch.save({
    "ground_truth": gt.cpu().float(),
    "input": x.cpu().float(),
    "rollouts": {ms: torch.from_numpy(r) for ms, r in all_rollouts.items()},
    "start_t": START_T,
    "traj_idx": TRAJ_IDX,
    "dataset": "shear_flow",
}, output_dir / "all_models_results.pt")

print(f"\nAll results saved to {output_dir}/")
print("Done!")
