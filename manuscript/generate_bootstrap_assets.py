from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPORT_PATH = Path("artifacts/bootstrap/reports/GPT_S-mean_pressure.csv")
OUTPUT_DIR = Path("manuscript/generated")


def write_table(report: pd.DataFrame, output_path: Path) -> None:
    table = report[
        [
            "method",
            "scale",
            "target_shift_mean",
            "off_target_drift_mean",
            "next_step_mse_delta_mean",
        ]
    ].copy()
    table["target_shift_mean"] = table["target_shift_mean"].map(lambda x: f"{x:.3f}")
    table["off_target_drift_mean"] = table["off_target_drift_mean"].map(
        lambda x: f"{x:.3f}"
    )
    table["next_step_mse_delta_mean"] = table["next_step_mse_delta_mean"].map(
        lambda x: f"{x:.6f}"
    )
    output_path.write_text(table.to_latex(index=False, escape=False))


def write_plot(report: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method, frame in report.groupby("method"):
        frame = frame.sort_values("scale")
        ax.plot(
            frame["scale"],
            frame["target_shift_mean"],
            marker="o",
            linewidth=2,
            label=method,
        )
        ax.fill_between(
            frame["scale"],
            frame["target_shift_ci_low"],
            frame["target_shift_ci_high"],
            alpha=0.15,
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Steering Scale")
    ax.set_ylabel("Mean Pressure Shift (z)")
    ax.set_title("GPT_S Mean-Pressure Steering on Rayleigh-Benard")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing report at {REPORT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = pd.read_csv(REPORT_PATH)
    write_table(report, OUTPUT_DIR / "bootstrap_gpt_s_mean_pressure_table.tex")
    write_plot(report, OUTPUT_DIR / "bootstrap_gpt_s_mean_pressure.png")


if __name__ == "__main__":
    main()
