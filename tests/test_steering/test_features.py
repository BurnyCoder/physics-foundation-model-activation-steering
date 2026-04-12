import numpy as np
import torch

from gphyt.steering.features import (
    compute_feature_batch,
    fit_zscore_stats,
    quartile_contrast_labels,
    zscore,
)


def test_divergence_magnitude_on_linear_velocity_field():
    coords = torch.arange(4, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    x = torch.zeros(1, 1, 4, 4, 5)
    x[0, 0, ..., 3] = xx
    x[0, 0, ..., 4] = yy

    divergence = compute_feature_batch(x, "divergence_magnitude")
    assert torch.allclose(divergence, torch.tensor([2.0]))


def test_enstrophy_and_vortex_intermittency_on_constant_curl_field():
    coords = torch.arange(4, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    x = torch.zeros(1, 1, 4, 4, 5)
    x[0, 0, ..., 3] = -yy
    x[0, 0, ..., 4] = xx

    enstrophy = compute_feature_batch(x, "enstrophy_proxy")
    intermittency = compute_feature_batch(x, "vortex_intermittency")

    assert torch.allclose(enstrophy, torch.tensor([4.0]))
    assert torch.allclose(intermittency, torch.tensor([1.0]))


def test_shock_and_stratification_scores():
    coords = torch.arange(4, dtype=torch.float32)
    xx, _ = torch.meshgrid(coords, coords, indexing="ij")

    shock_input = torch.zeros(1, 1, 4, 4, 5)
    shock_input[0, 0, ..., 1] = xx
    shock_input[0, 0, ..., 3] = -xx
    shock = compute_feature_batch(shock_input, "shock_score")
    assert torch.allclose(shock, torch.tensor([1.0]))

    strat_input = torch.zeros(1, 1, 4, 4, 5)
    strat_input[0, 0, :2, :, 1] = 3.0
    strat_input[0, 0, 2:, :, 1] = 1.0
    stratification = compute_feature_batch(strat_input, "stratification_score")
    assert torch.allclose(stratification, torch.tensor([2.0]))


def test_pressure_statistics_and_quartile_labels():
    x = torch.zeros(4, 1, 2, 2, 5)
    x[:, 0, ..., 0] = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1)

    mean_pressure = compute_feature_batch(x, "mean_pressure")
    contrast = compute_feature_batch(x, "pressure_contrast")
    stats = fit_zscore_stats(mean_pressure)
    scores = zscore(mean_pressure, stats)
    labels, mask = quartile_contrast_labels(scores)

    assert torch.allclose(mean_pressure, torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(contrast.numpy(), np.zeros(4))
    assert labels.tolist() == [0, -1, -1, 1]
    assert mask.tolist() == [True, False, False, True]
