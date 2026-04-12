Generate the current manuscript assets with:

```bash
uv run python manuscript/generate_bootstrap_assets.py
```

Build the draft with `latexmk -pdf main.tex` from `manuscript/` after a LaTeX
distribution is installed.

Render the current steering GIFs with:

```bash
python -m gphyt.steering.cli render-gifs --config gphyt/steering/steering.yml
```

The default public-result animations are written to `manuscript/generated/gifs/`.

Current tracked GIF assets:

- [`bootstrap_gpt_xl_rollout_pressure.gif`](generated/gifs/bootstrap_gpt_xl_rollout_pressure.gif)
- [`bootstrap_gpt_xl_rollout_velocity.gif`](generated/gifs/bootstrap_gpt_xl_rollout_velocity.gif)
- [`bootstrap_gpt_xl_rollout_pressure_neg.gif`](generated/gifs/bootstrap_gpt_xl_rollout_pressure_neg.gif)
- [`bootstrap_gpt_xl_rollout_velocity_neg.gif`](generated/gifs/bootstrap_gpt_xl_rollout_velocity_neg.gif)
- [`bootstrap_gpt_xl_mean_velocity_rollout.gif`](generated/gifs/bootstrap_gpt_xl_mean_velocity_rollout.gif)
- [`bootstrap_gpt_xl_mean_velocity_pressure.gif`](generated/gifs/bootstrap_gpt_xl_mean_velocity_pressure.gif)
- [`bootstrap_gpt_xl_regime_shear_velocity.gif`](generated/gifs/bootstrap_gpt_xl_regime_shear_velocity.gif)
- [`bootstrap_gpt_xl_regime_shear_velocity_flip.gif`](generated/gifs/bootstrap_gpt_xl_regime_shear_velocity_flip.gif)
- [`bootstrap_gpt_xl_regime_shear_pressure.gif`](generated/gifs/bootstrap_gpt_xl_regime_shear_pressure.gif)
- [`bootstrap_gpt_xl_enstrophy_rollout.gif`](generated/gifs/bootstrap_gpt_xl_enstrophy_rollout.gif)
- [`bootstrap_gpt_s_rollout_pressure.gif`](generated/gifs/bootstrap_gpt_s_rollout_pressure.gif)
- [`bootstrap_gpt_s_rollout_velocity.gif`](generated/gifs/bootstrap_gpt_s_rollout_velocity.gif)
- [`bootstrap_gpt_l_rollout_pressure_neg.gif`](generated/gifs/bootstrap_gpt_l_rollout_pressure_neg.gif)
- [`bootstrap_gpt_m_rollout_pressure_failure.gif`](generated/gifs/bootstrap_gpt_m_rollout_pressure_failure.gif)
- [`bootstrap_gpt_s_transfer_pressure.gif`](generated/gifs/bootstrap_gpt_s_transfer_pressure.gif)
- [`bootstrap_gpt_s_transfer_velocity.gif`](generated/gifs/bootstrap_gpt_s_transfer_velocity.gif)
