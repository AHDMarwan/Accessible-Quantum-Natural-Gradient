"""Regenerate the manuscript figures from the paper-scale summary tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "paper" / "paper_scale_v2" / "summaries"
OUT = ROOT / "paper" / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def accessible_geometry():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.20, 0.24, "Measurement score\n$S$"),
        (0.30, 0.58, 0.22, 0.24, "Retained readout span\n$P_B$"),
        (0.59, 0.58, 0.25, 0.24, "Accessible metric\n$G_{\\rm acc}=S^T P_B S$"),
        (0.59, 0.12, 0.25, 0.24, "Preconditioned step\n$(G_{\\rm acc}+\\lambda I)d=g$"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for x1, y1, x2, y2 in [
        (0.23, 0.70, 0.30, 0.70),
        (0.52, 0.70, 0.59, 0.70),
        (0.715, 0.58, 0.715, 0.36),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", linewidth=1.3))
    ax.text(0.405, 0.88, "measurement interface", ha="center", fontsize=9)
    ax.text(0.10, 0.33, "Full measurement geometry:\n$G_C=S^TS$",
            ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.16, 0.58), xytext=(0.13, 0.40),
                arrowprops=dict(arrowstyle="->", linewidth=1.0))
    ax.text(0.94, 0.70, "$G_{\\rm acc}\\preceq G_C\\preceq G_Q$",
            ha="center", va="center", fontsize=9)
    save(fig, "accessible_geometry.pdf")


def scaling_geometry():
    df = pd.read_csv(SUMMARY / "scaling_geometry_summary.csv")
    df = df[df["family"] == "su2_haar"].sort_values("n_qubits")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(df.n_qubits, df.aligned_crossfit_heldout_rho, marker="o", label="aligned")
    ax.plot(df.n_qubits, df.physical_heldout_rho, marker="s", label="physical")
    ax.plot(df.n_qubits, df.random_rank_heldout_rho, marker="^", label="random rank")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="rank baseline")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("normalized retention $\\rho$")
    ax.set_xticks(df.n_qubits)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    save(fig, "scaling_geometry.pdf")


def readout_order_tradeoff():
    geom = pd.read_csv(SUMMARY / "readout_order_geometry_summary.csv")
    res = pd.read_csv(SUMMARY / "readout_order_summary.csv")
    geom = geom[geom["family"] == "su2_haar"]
    res = res[(res["family"] == "su2_haar") & (res["method"] == "AQNG-aligned")]
    df = geom.merge(res, on=["dataset", "family", "readout_order"])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for dataset, grp in df.groupby("dataset"):
        grp = grp.sort_values("readout_order")
        label = "Iris" if dataset == "iris01" else "Digits"
        ax.plot(grp.aligned_crossfit_heldout_retention, grp.test_loss_mean,
                marker="o", label=label)
        for _, row in grp.iterrows():
            ax.annotate(f"order {int(row.readout_order)}",
                        (row.aligned_crossfit_heldout_retention, row.test_loss_mean),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("aligned retained mass $R$")
    ax.set_ylabel("mean test loss")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    save(fig, "readout_order_tradeoff.pdf")


def u1_signal():
    geom = pd.read_csv(SUMMARY / "scaling_geometry_summary.csv")
    res = pd.read_csv(SUMMARY / "scaling_summary.csv")
    geom = geom[geom["family"] == "u1_rzxy"].sort_values("n_qubits")
    res = res[(res["family"] == "u1_rzxy") & (res["method"] == "AQNG-aligned")]
    res = res.sort_values("n_qubits")
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(geom.n_qubits, 1 - geom.aligned_crossfit_heldout_retention,
            marker="o", label="$1-R_{\\rm aligned}$")
    ax.plot(res.n_qubits, res.first_g_dot_d_mean, marker="s",
            label="first-step $g^T d$")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("diagnostic magnitude (log scale)")
    ax.set_xticks(geom.n_qubits)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    save(fig, "u1_accessibility_vs_signal.pdf")


if __name__ == "__main__":
    accessible_geometry()
    scaling_geometry()
    readout_order_tradeoff()
    u1_signal()
