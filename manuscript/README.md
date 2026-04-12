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
- [`bootstrap_gpt_s_transfer_pressure.gif`](generated/gifs/bootstrap_gpt_s_transfer_pressure.gif)
