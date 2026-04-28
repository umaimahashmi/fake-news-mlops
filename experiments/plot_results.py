"""Generate paper figures from comparison_results.csv."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_results():
    path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Run experiments/run_comparison.py first. Missing: {path}"
        )
    return pd.read_csv(path)


def fig1_accuracy_over_windows(df: pd.DataFrame):
    """Figure 1: Accuracy degradation curves for Setup A / B / C."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"A": "#e74c3c", "B": "#f39c12", "C": "#27ae60"}
    labels = {
        "A": "Setup A: Static",
        "B": "Setup B: Periodic",
        "C": "Setup C: Drift-Triggered",
    }

    for setup in ["A", "B", "C"]:
        sub = df[df["setup"] == setup].sort_values("window")
        ax.plot(
            sub["window"],
            sub["accuracy"],
            marker="o",
            color=colors[setup],
            label=labels[setup],
            linewidth=2,
        )
        retrain_wins = sub[sub["retrained"]]
        if len(retrain_wins):
            ax.scatter(
                retrain_wins["window"],
                retrain_wins["accuracy"],
                color=colors[setup],
                marker="*",
                s=200,
                zorder=5,
            )

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, label="Drift onset")
    ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Data Window (time →)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Figure 1: Model Accuracy Under Distributional Shift\n(★ = retraining event)",
        fontsize=13,
    )
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig1_accuracy_over_windows.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def fig2_drift_vs_accuracy(df: pd.DataFrame):
    """Figure 2: Drift score vs accuracy (Setup C only)."""
    sub = df[df["setup"] == "C"].sort_values("window")
    if "drift_score" not in sub.columns:
        print("No drift_score column in results — skipping figure 2")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(sub["window"], sub["accuracy"], "g-o", label="Accuracy", linewidth=2)
    ax2.plot(
        sub["window"], sub["drift_score"], "b--s", label="Drift Score (KS)", linewidth=2
    )
    ax2.axhline(y=0.1, color="red", linestyle=":", alpha=0.7, label="Drift Threshold")
    ax1.set_xlabel("Data Window", fontsize=12)
    ax1.set_ylabel("Accuracy", color="green", fontsize=12)
    ax2.set_ylabel("KS Drift Score", color="blue", fontsize=12)
    ax1.set_title("Figure 2: Drift Score vs Accuracy (Setup C)", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig2_drift_vs_accuracy.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def fig3_retraining_efficiency(df: pd.DataFrame):
    """Figure 3: Cumulative retrains vs average accuracy — B vs C bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    setups = ["B", "C"]
    colors = ["#f39c12", "#27ae60"]
    retrain_counts = [df[df["setup"] == s]["retrained"].sum() for s in setups]
    avg_accs = [df[df["setup"] == s]["accuracy"].mean() for s in setups]

    axes[0].bar(setups, retrain_counts, color=colors)
    axes[0].set_title("Total Retraining Cycles", fontsize=12)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(retrain_counts):
        axes[0].text(i, v + 0.1, str(int(v)), ha="center", fontweight="bold")

    axes[1].bar(setups, avg_accs, color=colors)
    axes[1].set_title("Average Accuracy Under Drift", fontsize=12)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(avg_accs):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    fig.suptitle("Figure 3: Retraining Efficiency — Setup B vs C", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig3_retraining_efficiency.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    df = load_results()
    fig1_accuracy_over_windows(df)
    fig2_drift_vs_accuracy(df)
    fig3_retraining_efficiency(df)
    print("\nAll figures saved to results/")
