from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "artifacts/bootstrap/reports/GPT_S-mean_pressure.csv"
OUTPUT_DIR = REPO_ROOT / "manuscript/generated"
SCALING_REPORT_GLOB = "GPT_*-mean_pressure.csv"
REGIME_REPORT_PATH = REPO_ROOT / "artifacts/bootstrap/reports/GPT_S-regime-shear-vs-rayleigh.csv"
REGIME_VECTOR_SUMMARY = REPO_ROOT / "artifacts/bootstrap/vectors/GPT_S_regime/vector_summary.json"
ROLLOUT_REPORT_GLOB = "*/mean_pressure-block_out:0-logistic.csv"
TRANSFER_REPORT_GLOB = "*/mean_pressure-block_out:0-logistic.csv"
SUMMARY_PATH = REPO_ROOT / "artifacts/bootstrap/reports/bootstrap_rollout_transfer_summary.csv"


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
    table = table.rename(
        columns={
            "method": "Method",
            "scale": "Scale",
            "target_shift_mean": "Target Shift",
            "off_target_drift_mean": "Off-Target Drift",
            "next_step_mse_delta_mean": "Next-Step MSE Delta",
        }
    )
    output_path.write_text(table.to_latex(index=False, escape=True))


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


def write_scaling_table(report_dir: Path, output_path: Path) -> None:
    frames = []
    for report_path in sorted(report_dir.glob(SCALING_REPORT_GLOB)):
        report = pd.read_csv(report_path)
        subset = report[(report["method"] == "logistic") & (report["scale"] == 2.0)].copy()
        frames.append(
            subset[
                [
                    "model_size",
                    "target_shift_mean",
                    "off_target_drift_mean",
                    "next_step_mse_delta_mean",
                ]
            ]
        )
    table = pd.concat(frames, ignore_index=True)
    table["target_shift_mean"] = table["target_shift_mean"].map(lambda x: f"{x:.3f}")
    table["off_target_drift_mean"] = table["off_target_drift_mean"].map(
        lambda x: f"{x:.3f}"
    )
    table["next_step_mse_delta_mean"] = table["next_step_mse_delta_mean"].map(
        lambda x: f"{x:.6f}"
    )
    table = table.rename(
        columns={
            "model_size": "Model",
            "target_shift_mean": "Target Shift @ +2",
            "off_target_drift_mean": "Off-Target Drift @ +2",
            "next_step_mse_delta_mean": "Next-Step MSE Delta @ +2",
        }
    )
    output_path.write_text(table.to_latex(index=False, escape=True))


def write_scaling_plot(report_dir: Path, output_path: Path) -> None:
    frames = []
    for report_path in sorted(report_dir.glob(SCALING_REPORT_GLOB)):
        frames.append(pd.read_csv(report_path))
    report = pd.concat(frames, ignore_index=True)
    report = report[report["method"] == "logistic"].copy()
    order = ["GPT_S", "GPT_M", "GPT_L", "GPT_XL"]
    report["model_size"] = pd.Categorical(report["model_size"], categories=order, ordered=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for scale, frame in report.groupby("scale"):
        frame = frame.sort_values("model_size")
        ax.plot(
            frame["model_size"],
            frame["target_shift_mean"],
            marker="o",
            linewidth=2,
            label=f"scale={scale:g}",
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Mean Pressure Shift (z)")
    ax.set_title("Scaling of Logistic Mean-Pressure Steering")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_regime_table(report_path: Path, summary_path: Path, output_path: Path) -> None:
    report = pd.read_csv(report_path)
    summary = pd.read_json(summary_path)
    auc_map = {
        row["method"]: row["score"]
        for _, row in summary.iterrows()
    }
    plus_two = report[report["scale"] == 2.0][
        ["method", "target_shift_mean", "off_target_drift_mean", "next_step_mse_delta_mean"]
    ].copy()
    plus_two["auc"] = plus_two["method"].map(auc_map)
    plus_two["auc"] = plus_two["auc"].map(lambda x: f"{x:.3f}")
    plus_two["target_shift_mean"] = plus_two["target_shift_mean"].map(lambda x: f"{x:.4f}")
    plus_two["off_target_drift_mean"] = plus_two["off_target_drift_mean"].map(lambda x: f"{x:.4f}")
    plus_two["next_step_mse_delta_mean"] = plus_two["next_step_mse_delta_mean"].map(
        lambda x: f"{x:.6f}"
    )
    plus_two = plus_two.rename(
        columns={
            "method": "Method",
            "auc": "Validation AUC",
            "target_shift_mean": "Velocity Shift @ +2",
            "off_target_drift_mean": "Off-Target Drift @ +2",
            "next_step_mse_delta_mean": "Next-Step MSE Delta @ +2",
        }
    )
    output_path.write_text(plus_two.to_latex(index=False, escape=True))


def write_rollout_table(summary: pd.DataFrame, output_path: Path) -> None:
    table = summary[
        [
            "model_size",
            "rollout_target_shift_mean_at_2",
            "steered_rollout_vmse_final_mean_at_2",
            "rollout_nonfinite_fraction_at_2",
        ]
    ].copy()
    table["rollout_target_shift_mean_at_2"] = table["rollout_target_shift_mean_at_2"].map(
        lambda x: f"{x:.3f}"
    )
    table["steered_rollout_vmse_final_mean_at_2"] = table[
        "steered_rollout_vmse_final_mean_at_2"
    ].map(lambda x: f"{x:.3f}")
    table["rollout_nonfinite_fraction_at_2"] = table[
        "rollout_nonfinite_fraction_at_2"
    ].map(lambda x: f"{x:.3f}")
    table = table.rename(
        columns={
            "model_size": "Model",
            "rollout_target_shift_mean_at_2": "20-Step Final Shift @ +2",
            "steered_rollout_vmse_final_mean_at_2": "20-Step Final VMSE @ +2",
            "rollout_nonfinite_fraction_at_2": "Non-Finite Fraction",
        }
    )
    output_path.write_text(table.to_latex(index=False, escape=True))


def write_transfer_table(summary: pd.DataFrame, output_path: Path) -> None:
    table = summary[
        [
            "model_size",
            "transfer_target_shift_mean_at_2",
            "transfer_off_target_drift_mean_at_2",
            "transfer_next_step_mse_delta_mean_at_2",
        ]
    ].copy()
    table["transfer_target_shift_mean_at_2"] = table["transfer_target_shift_mean_at_2"].map(
        lambda x: f"{x:.3f}"
    )
    table["transfer_off_target_drift_mean_at_2"] = table[
        "transfer_off_target_drift_mean_at_2"
    ].map(lambda x: f"{x:.3f}")
    table["transfer_next_step_mse_delta_mean_at_2"] = table[
        "transfer_next_step_mse_delta_mean_at_2"
    ].map(lambda x: f"{x:.6f}")
    table = table.rename(
        columns={
            "model_size": "Model",
            "transfer_target_shift_mean_at_2": "Transfer Shift @ +2",
            "transfer_off_target_drift_mean_at_2": "Transfer Drift @ +2",
            "transfer_next_step_mse_delta_mean_at_2": "Transfer MSE Delta @ +2",
        }
    )
    output_path.write_text(table.to_latex(index=False, escape=True))


def _load_family_reports(report_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(report_path) for report_path in sorted(report_dir.glob(ROLLOUT_REPORT_GLOB))]
    return pd.concat(frames, ignore_index=True)


def write_rollout_plot(report_dir: Path, output_path: Path) -> None:
    report = _load_family_reports(report_dir)
    order = ["GPT_S", "GPT_M", "GPT_L", "GPT_XL"]
    report["model_size"] = pd.Categorical(report["model_size"], categories=order, ordered=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for model_size, frame in report.groupby("model_size"):
        frame = frame.sort_values("scale")
        ax.plot(
            frame["scale"],
            frame["rollout_target_shift_mean"],
            marker="o",
            linewidth=2,
            label=model_size,
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Steering Scale")
    ax.set_ylabel("20-Step Final Pressure Shift (z)")
    ax.set_title("Autoregressive Rollout Steering on Rayleigh-Benard")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_transfer_plot(report_dir: Path, output_path: Path) -> None:
    frames = [pd.read_csv(report_path) for report_path in sorted(report_dir.glob(TRANSFER_REPORT_GLOB))]
    report = pd.concat(frames, ignore_index=True)
    order = ["GPT_S", "GPT_M", "GPT_L", "GPT_XL"]
    report["model_size"] = pd.Categorical(report["model_size"], categories=order, ordered=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for model_size, frame in report.groupby("model_size"):
        frame = frame.sort_values("scale")
        ax.plot(
            frame["scale"],
            frame["target_shift_mean"],
            marker="o",
            linewidth=2,
            label=model_size,
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Steering Scale")
    ax.set_ylabel("Transfer Pressure Shift (z)")
    ax.set_title("Transfer to Turbulent Radiative Layer")
    ax.legend(frameon=False, ncol=2)
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
    write_scaling_table(
        REPO_ROOT / "artifacts/bootstrap/reports",
        OUTPUT_DIR / "bootstrap_scaling_table.tex",
    )
    write_scaling_plot(
        REPO_ROOT / "artifacts/bootstrap/reports",
        OUTPUT_DIR / "bootstrap_scaling.png",
    )
    if REGIME_REPORT_PATH.exists() and REGIME_VECTOR_SUMMARY.exists():
        write_regime_table(
            REGIME_REPORT_PATH,
            REGIME_VECTOR_SUMMARY,
            OUTPUT_DIR / "bootstrap_regime_table.tex",
        )
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        write_rollout_table(summary, OUTPUT_DIR / "bootstrap_rollout_table.tex")
        write_transfer_table(summary, OUTPUT_DIR / "bootstrap_transfer_table.tex")
    rollout_dir = REPO_ROOT / "artifacts/bootstrap/rollouts"
    if rollout_dir.exists():
        write_rollout_plot(
            rollout_dir,
            OUTPUT_DIR / "bootstrap_rollout.png",
        )
    transfer_dir = REPO_ROOT / "artifacts/bootstrap/transfer"
    if transfer_dir.exists():
        write_transfer_plot(
            transfer_dir,
            OUTPUT_DIR / "bootstrap_transfer.png",
        )


if __name__ == "__main__":
    main()
