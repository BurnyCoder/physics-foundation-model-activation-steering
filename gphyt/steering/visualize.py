"""Visualization utilities for steering rollouts and manuscript GIF assets."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


FIELD_INDEX = {
    "pressure": 0,
    "density": 1,
    "temperature": 2,
    "velocity_x": 3,
    "velocity_y": 4,
}


def extract_display_field(fields: np.ndarray, field_name: str) -> np.ndarray:
    """Extract a display-ready field from `(T, H, W, C)` arrays."""
    if field_name == "velocity_magnitude":
        velocity = fields[..., [FIELD_INDEX["velocity_x"], FIELD_INDEX["velocity_y"]]]
        return np.linalg.norm(velocity, axis=-1)
    if field_name not in FIELD_INDEX:
        raise KeyError(f"Unsupported field {field_name}")
    return fields[..., FIELD_INDEX[field_name]]


def render_comparison_gif(
    ground_truth: np.ndarray,
    baseline: np.ndarray,
    steered: np.ndarray,
    *,
    field_name: str,
    output_path: str | Path,
    title: str,
    fps: int = 3,
) -> Path:
    """Render a 4-panel GIF with ground truth, baseline, steered, and delta."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_field = extract_display_field(ground_truth, field_name)
    baseline_field = extract_display_field(baseline, field_name)
    steered_field = extract_display_field(steered, field_name)
    delta_field = steered_field - baseline_field

    combined = np.stack([gt_field, baseline_field, steered_field], axis=0)
    finite_values = combined[np.isfinite(combined)]
    if finite_values.size == 0:
        raise ValueError("Cannot render GIF from all-nonfinite fields")
    vmin = float(np.percentile(finite_values, 1))
    vmax = float(np.percentile(finite_values, 99))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    delta_values = delta_field[np.isfinite(delta_field)]
    if delta_values.size == 0:
        delta_bound = 1e-6
    else:
        delta_bound = float(np.percentile(np.abs(delta_values), 99))
        delta_bound = max(delta_bound, 1e-6)

    frames = []
    num_frames = min(gt_field.shape[0], baseline_field.shape[0], steered_field.shape[0])
    for frame_idx in range(num_frames):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), constrained_layout=True)
        panels = [
            ("Ground Truth", gt_field[frame_idx], "viridis", vmin, vmax),
            ("Baseline", baseline_field[frame_idx], "viridis", vmin, vmax),
            ("Steered", steered_field[frame_idx], "viridis", vmin, vmax),
            ("Steered - Baseline", delta_field[frame_idx], "coolwarm", -delta_bound, delta_bound),
        ]
        colorbars = []
        for ax, (panel_title, field, cmap, panel_vmin, panel_vmax) in zip(axes, panels, strict=True):
            image = ax.imshow(field.T, cmap=cmap, vmin=panel_vmin, vmax=panel_vmax, origin="lower")
            ax.set_title(panel_title)
            ax.set_xticks([])
            ax.set_yticks([])
            colorbars.append(image)
        fig.suptitle(f"{title} | {field_name.replace('_', ' ')} | t={frame_idx}")
        fig.colorbar(colorbars[0], ax=axes[:3], shrink=0.75, location="bottom", pad=0.04)
        fig.colorbar(colorbars[-1], ax=axes[3], shrink=0.75, location="bottom", pad=0.04)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3].copy())
        plt.close(fig)

    iio.imwrite(output_path, frames, duration=1000 / fps, loop=0)
    return output_path
