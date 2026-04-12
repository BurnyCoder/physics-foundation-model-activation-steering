Generate the current manuscript assets with:

```bash
uv run python manuscript/generate_bootstrap_assets.py
```

Build the draft with `latexmk -pdf main.tex` from `manuscript/` after a LaTeX
distribution is installed.
