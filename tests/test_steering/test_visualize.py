from pathlib import Path

import imageio.v3 as iio
import numpy as np

from gphyt.steering.visualize import extract_display_field, render_comparison_gif


def test_extract_display_field_velocity_magnitude():
    fields = np.zeros((2, 3, 4, 5), dtype=np.float32)
    fields[..., 3] = 3.0
    fields[..., 4] = 4.0
    magnitude = extract_display_field(fields, "velocity_magnitude")
    assert magnitude.shape == (2, 3, 4)
    assert np.allclose(magnitude, 5.0)


def test_render_comparison_gif_writes_frames(tmp_path: Path):
    base = np.zeros((3, 6, 5, 5), dtype=np.float32)
    base[..., 0] = np.linspace(0.0, 1.0, 3)[:, None, None]
    baseline = base.copy()
    steered = base.copy()
    steered[..., 0] += 0.25

    output_path = tmp_path / "comparison.gif"
    render_comparison_gif(
        base,
        baseline,
        steered,
        field_name="pressure",
        output_path=output_path,
        title="Test GIF",
        fps=2,
    )

    assert output_path.exists()
    frames = iio.imread(output_path)
    assert frames.shape[0] == 3
