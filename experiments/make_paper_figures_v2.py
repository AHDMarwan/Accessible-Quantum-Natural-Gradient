"""Generate the main AQNG v2 paper figures from frozen result tables.

The script is reporting-only. It reads the committed paper_scale_v2 package and
writes vector PDF plus PNG previews under paper/aqng/figures/generated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper" / "paper_scale_v2"
OUT = ROOT / "paper" / "aqng" / "figures" / "generated"

METHOD_ORDER = [
    "AQNG-aligned",
    "AQNG-random",
    "AQNG-physical",
    "AQNG-Z0",
    "Adam",
    "SGD",
    "Full-QNG",
    "Block-QNG",
]

METHOD_LABEL = {
    "AQNG-aligned": "AQNG aligned",
    "AQNG-random": "AQNG random",
    "AQNG-physical": "AQNG physical",
    "AQNG-Z0": "AQNG Z0",
    "Adam": "Adam",
    "SGD": "SGD",
    "Full-QNG": "Full QNG",
    "Block-QNG": "Block QNG",
}


def _finish(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _method_subset(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    out = df[df["method"].isin(methods)].copy()
    out["method"] = pd.Categorical(out["method"], methods, ordered=True)
    return out.sort_values("method")


def fig02_generic_primary() -> None:
    df = pd.read_csv(RESULTS / "summaries" / "generic_primary_summary.csv")
    df = _method_subset(df, METHOD_ORDER)
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    axes[0].scatter(df["test_loss_mean"], y, s=34)
    axes[0].set_xlabel("Mean terminal test loss (lower is better)")
    axes[0].set_yticks(y, [METHOD_LABEL[str(m)] for m in df["method"]])
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].scatter(df["test_acc_mean"], y, s=34)
    axes[1].set_xlabel("Mean terminal test accuracy")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_xlim(0.74, 0.81)

    fig.suptitle("Frozen generic primary benchmark: 240 matched dataset-architecture cases")
    fig.tight_layout()
    _finish(fig, "fig02_generic_primary")


def fig03_scaling_geometry() -> None:
    df = pd.read_csv(RESULTS / "summaries" / "scaling_geometry_summary.csv")
    d = df[df["family"] == "su2_haar"].sort_values("n_qubits")
    n = d["n_qubits"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    axes[0].plot(n, d["aligned_crossfit_heldout_retention"], marker="o", label="aligned")
    axes[0].plot(n, d["physical_heldout_retention"], marker="s", label="physical")
    axes[0].plot(n, d["random_rank_heldout_retention"], marker="^", label="random")
    axes[0].plot(n, d["rank_baseline"], marker="x", linestyle="--", label="rank baseline")
    axes[0].set_xlabel("Qubits n")
    axes[0].set_ylabel("Held-out tangent retention R")
    axes[0].set_yscale("log")
    axes[0].set_xticks(n)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(n, d["aligned_crossfit_heldout_rho"], marker="o", label="aligned")
    axes[1].plot(n, d["physical_heldout_rho"], marker="s", label="physical")
    axes[1].plot(n, d["random_rank_heldout_rho"], marker="^", label="random")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Qubits n")
    axes[1].set_ylabel("Rank-normalized retention rho")
    axes[1].set_yscale("log")
    axes[1].set_xticks(n)
    axes[1].grid(alpha=0.25)

    fig.suptitle("SU(2)-Haar: rank law remains clean while orientation gain grows")
    fig.tight_layout()
    _finish(fig, "fig03_scaling_geometry")


def fig04_large_deep() -> None:
    df = pd.read_csv(RESULTS / "followup" / "large_deep_summary.csv")
    methods = [
        "AQNG-aligned",
        "AQNG-random",
        "AQNG-physical",
        "Adam",
        "SGD",
        "Full-QNG",
    ]
    settings = ["n8_L3", "n10_L3", "n6_L5"]
    titles = ["n=8, L=3", "n=10, L=3", "n=6, L=5"]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.8), sharey=True)
    for ax, setting, title in zip(axes, settings, titles):
        d = _method_subset(df[df["setting"] == setting], methods)
        y = np.arange(len(d))
        ax.errorbar(
            d["test_loss_mean"],
            y,
            xerr=d["test_loss_std"],
            fmt="o",
            capsize=2.5,
            linewidth=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Terminal test loss")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y, [METHOD_LABEL[str(m)] for m in d["method"]])
        ax.invert_yaxis()

    fig.suptitle(
        "Matched larger/deeper SU(2)-Haar follow-up (5 paired seeds per setting)\n"
        "Pooled aligned vs SGD: Holm p=0.00122; aligned vs Full QNG: p=0.00580; aligned vs Adam: n.s."
    )
    fig.tight_layout()
    _finish(fig, "fig04_large_deep")


def fig05_accessibility_not_trainability() -> None:
    scaling = pd.read_csv(RESULTS / "summaries" / "scaling_geometry_summary.csv")
    train = pd.read_csv(RESULTS / "summaries" / "scaling_summary.csv")
    order_g = pd.read_csv(RESULTS / "summaries" / "readout_order_geometry_summary.csv")
    order_t = pd.read_csv(RESULTS / "summaries" / "readout_order_summary.csv")

    ug = scaling[scaling["family"] == "u1_rzxy"].sort_values("n_qubits")
    ut = train[(train["family"] == "u1_rzxy") & (train["method"] == "AQNG-aligned")].sort_values("n_qubits")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))

    ax = axes[0]
    ax.plot(ug["n_qubits"], ug["aligned_crossfit_heldout_retention"], marker="o", label="aligned retention")
    ax.plot(ug["n_qubits"], ug["physical_heldout_retention"], marker="s", label="physical retention")
    ax.set_xlabel("Qubits n")
    ax.set_ylabel("Held-out retention R")
    ax.set_ylim(0.85, 1.02)
    ax.set_xticks(ug["n_qubits"])
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(ut["n_qubits"], ut["first_g_dot_d_mean"], marker="^", linestyle="--", label="first g.d")
    ax2.set_yscale("log")
    ax2.set_ylabel("First-step g.d (log scale)")
    ax.set_title("U(1): accessible geometry, collapsing descent signal")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    for dataset, label in [("iris01", "Iris"), ("digits01", "Digits")]:
        g = order_g[(order_g["dataset"] == dataset) & (order_g["family"] == "su2_haar")].sort_values("readout_order")
        t = order_t[(order_t["dataset"] == dataset) & (order_t["family"] == "su2_haar") & (order_t["method"] == "AQNG-aligned")].sort_values("readout_order")
        x = g["aligned_crossfit_heldout_retention"].to_numpy()
        y = t["test_loss_mean"].to_numpy()
        ax.plot(x, y, marker="o", label=label)
        if len(x) == 2:
            ax.annotate("order 1", (x[0], y[0]), xytext=(4, -12), textcoords="offset points", fontsize=8)
            ax.annotate("order 2", (x[1], y[1]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Aligned held-out retention R")
    ax.set_ylabel("Aligned AQNG terminal test loss")
    ax.set_title("More retained tangent mass can worsen task loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    _finish(fig, "fig05_accessibility_not_trainability")


def fig06_finite_shot() -> None:
    df = pd.read_csv(RESULTS / "summaries" / "finite_shot_summary.csv")
    d = df[df["family"] == "su2_haar"].copy()
    agg = (
        d.groupby(["shots", "method"], as_index=False)
        .agg(test_loss_mean=("test_loss_mean", "mean"))
    )
    methods = ["AQNG-aligned", "AQNG-random", "AQNG-physical", "Full-QNG"]
    shots = [1000.0, 10000.0]

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for method in methods:
        m = agg[agg["method"] == method].set_index("shots").reindex(shots)
        ax.plot([1, 2], m["test_loss_mean"], marker="o", label=METHOD_LABEL[method])
    ax.set_xticks([1, 2], ["1k shots", "10k shots"])
    ax.set_ylabel("Mean terminal test loss (Iris + Digits)")
    ax.set_title("Finite-shot SU(2)-Haar robustness\nFull QNG uses an analytic metric oracle in this protocol")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _finish(fig, "fig06_finite_shot")


def main() -> None:
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig02_generic_primary()
    fig03_scaling_geometry()
    fig04_large_deep()
    fig05_accessibility_not_trainability()
    fig06_finite_shot()
    print(f"Wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
