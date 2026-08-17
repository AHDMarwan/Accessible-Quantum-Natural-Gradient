"""Aggregate and analyze the frozen AQNG paper-scale experiment matrix.

The analysis is deliberately post-hoc only with respect to *reporting*: optimizer
hyperparameters are frozen before this workflow runs.  The script produces
machine-readable summaries for the primary benchmark, finite-shot robustness,
qubit scaling, depth scaling, readout-order ablations, paired tests, and the
relationship between tangent retention and task loss improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


CORE_METHODS = [
    "AQNG-physical",
    "AQNG-random",
    "AQNG-aligned",
    "AQNG-Z0",
    "Full-QNG",
    "Block-QNG",
    "SGD",
    "Adam",
]


def _block_from_tag(tag: str) -> str:
    tag = str(tag)
    for name in ("primary", "finite", "scaling", "depth", "order"):
        if f"paper-run-{name}-" in tag or f"paper-{name}-" in tag:
            return name
    return "unknown"


def _holm(p_values: list[float]) -> np.ndarray:
    """Holm step-down adjusted p values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    vals = p[idx]
    order = np.argsort(vals)
    m = len(vals)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, pos in enumerate(order):
        candidate = (m - rank) * vals[pos]
        running = max(running, candidate)
        adjusted_sorted[pos] = min(1.0, running)
    out[idx] = adjusted_sorted
    return out


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if len(d) == 0:
        return np.nan
    if np.allclose(d, 0.0, atol=1e-14, rtol=0.0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _paired_rows(primary: pd.DataFrame) -> pd.DataFrame:
    """Planned paired comparisons against AQNG-aligned in each task/family cell."""
    rows: list[dict] = []
    for (dataset, family), g in primary.groupby(["dataset", "family"], sort=True):
        aligned = g[g.method == "AQNG-aligned"].set_index("seed")
        if aligned.empty:
            continue
        for ref in [m for m in CORE_METHODS if m != "AQNG-aligned"]:
            other = g[g.method == ref].set_index("seed")
            seeds = aligned.index.intersection(other.index)
            if len(seeds) == 0:
                continue
            a = aligned.loc[seeds, "test_loss"].to_numpy(float)
            b = other.loc[seeds, "test_loss"].to_numpy(float)
            delta = b - a  # positive means aligned has lower loss
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "method_a": "AQNG-aligned",
                    "method_b": ref,
                    "n_pairs": int(len(seeds)),
                    "mean_loss_a": float(np.mean(a)),
                    "mean_loss_b": float(np.mean(b)),
                    "mean_loss_advantage_a": float(np.mean(delta)),
                    "median_loss_advantage_a": float(np.median(delta)),
                    "wins_a": int(np.sum(delta > 1e-12)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "wins_b": int(np.sum(delta < -1e-12)),
                    "p_raw": _safe_wilcoxon(a, b),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = _holm(out["p_raw"].tolist())
    return out


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("seed", "nunique"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            train_loss_mean=("train_loss", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
            first_g_dot_d_mean=("first_g_dot_d", "mean"),
            first_actual_decrease_mean=("first_actual_same_batch_decrease", "mean"),
        )
        .reset_index()
    )
    return agg


def _cal_summary(cal: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if cal.empty:
        return pd.DataFrame()
    cols = [
        "physical_rank",
        "centered_score_dimension",
        "rank_baseline",
        "reference_support_size",
        "physical_heldout_retention",
        "physical_heldout_rho",
        "aligned_crossfit_heldout_retention",
        "aligned_crossfit_heldout_rho",
        "random_rank_heldout_retention",
        "random_rank_heldout_rho",
    ]
    aggs = {c: (c, "mean") for c in cols if c in cal.columns}
    aggs["n"] = ("seed", "nunique")
    return cal.groupby(group_cols, dropna=False).agg(**aggs).reset_index()


def _orientation_task_relation(primary: pd.DataFrame, cal_primary: pd.DataFrame) -> dict:
    if primary.empty or cal_primary.empty:
        return {}
    piv = primary.pivot_table(
        index=["dataset", "family", "seed"],
        columns="method",
        values="test_loss",
        aggfunc="first",
    ).reset_index()
    keep = [
        "dataset",
        "family",
        "seed",
        "physical_heldout_retention",
        "aligned_crossfit_heldout_retention",
        "random_rank_heldout_retention",
    ]
    merged = cal_primary[keep].merge(piv, on=["dataset", "family", "seed"], how="inner")
    needed = {"AQNG-physical", "AQNG-aligned", "AQNG-random"}
    if not needed.issubset(merged.columns):
        return {}
    merged["aligned_retention_gain"] = (
        merged["aligned_crossfit_heldout_retention"] - merged["physical_heldout_retention"]
    )
    merged["aligned_loss_gain"] = merged["AQNG-physical"] - merged["AQNG-aligned"]
    merged["random_retention_gain"] = (
        merged["random_rank_heldout_retention"] - merged["physical_heldout_retention"]
    )
    merged["random_loss_gain"] = merged["AQNG-physical"] - merged["AQNG-random"]

    def corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
            return {"rho": np.nan, "p": np.nan, "n": int(mask.sum())}
        r = spearmanr(x[mask], y[mask])
        return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(mask.sum())}

    by_family = {}
    for fam, g in merged.groupby("family"):
        by_family[str(fam)] = corr(g.aligned_retention_gain, g.aligned_loss_gain)
    return {
        "aligned_retention_vs_loss_gain": corr(
            merged.aligned_retention_gain, merged.aligned_loss_gain
        ),
        "random_retention_vs_loss_gain": corr(
            merged.random_retention_gain, merged.random_loss_gain
        ),
        "aligned_by_family": by_family,
        "n_rows": int(len(merged)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()

    outdir = a.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(a.results)
    cal = pd.read_csv(a.calibration)
    for df in (results, cal):
        if "paper_tag" not in df.columns:
            raise ValueError("aggregated input is missing paper_tag")
        df["paper_block"] = df["paper_tag"].map(_block_from_tag)

    primary = results[results.paper_block == "primary"].copy()
    finite = results[results.paper_block == "finite"].copy()
    scaling = results[results.paper_block == "scaling"].copy()
    depth = results[results.paper_block == "depth"].copy()
    order = results[results.paper_block == "order"].copy()
    cal_primary = cal[cal.paper_block == "primary"].copy()

    files = {}
    summaries = {
        "primary_summary.csv": _summary(primary, ["dataset", "family", "method"]),
        "finite_shot_summary.csv": _summary(finite, ["dataset", "family", "shots", "method"]),
        "scaling_summary.csv": _summary(scaling, ["dataset", "family", "n_qubits", "method"]),
        "depth_summary.csv": _summary(depth, ["dataset", "family", "n_layers", "method"]),
        "readout_order_summary.csv": _summary(order, ["dataset", "family", "readout_order", "method"]),
        "primary_geometry_summary.csv": _cal_summary(cal_primary, ["dataset", "family"]),
        "finite_geometry_summary.csv": _cal_summary(
            cal[cal.paper_block == "finite"], ["dataset", "family", "shots"]
        ),
        "scaling_geometry_summary.csv": _cal_summary(
            cal[cal.paper_block == "scaling"], ["dataset", "family", "n_qubits"]
        ),
        "depth_geometry_summary.csv": _cal_summary(
            cal[cal.paper_block == "depth"], ["dataset", "family", "n_layers"]
        ),
        "readout_order_geometry_summary.csv": _cal_summary(
            cal[cal.paper_block == "order"], ["dataset", "family", "readout_order"]
        ),
        "paired_primary_tests.csv": _paired_rows(primary),
    }
    for name, df in summaries.items():
        path = outdir / name
        df.to_csv(path, index=False)
        files[name] = int(len(df))

    # Resource summary is meaningful mainly in the finite-shot block.
    if len(finite):
        resource_cols = [
            c
            for c in [
                "training_shots_total",
                "calibration_shots_shared",
                "end_to_end_shots_including_calibration",
                "training_executions_total",
                "calibration_executions_shared",
                "end_to_end_executions_including_calibration",
            ]
            if c in finite.columns
        ]
        resource = (
            finite.groupby(["dataset", "family", "shots", "method"], dropna=False)[resource_cols]
            .mean()
            .reset_index()
        )
    else:
        resource = pd.DataFrame()
    resource.to_csv(outdir / "finite_resource_summary.csv", index=False)
    files["finite_resource_summary.csv"] = int(len(resource))

    # Basic completeness counts and planned core interpretation diagnostics.
    expected_primary_cells = 4 * 4 * 20
    observed_primary_cells = int(
        primary[["dataset", "family", "seed"]].drop_duplicates().shape[0]
    )
    method_counts = primary.groupby("method")["seed"].count().to_dict() if len(primary) else {}
    relation = _orientation_task_relation(primary, cal_primary)

    report = {
        "protocol_frozen": {
            "lr": 0.06,
            "lam": 0.003,
            "metric_normalization": "trace",
            "damping_mode": "absolute",
            "max_direction_norm": 8.0,
            "max_metric_step": 0.25,
            "steps": 20,
            "primary_n_qubits": 6,
            "primary_n_layers": 3,
            "primary_readout_order": 1,
        },
        "completeness": {
            "expected_primary_dataset_family_seed_cells": expected_primary_cells,
            "observed_primary_dataset_family_seed_cells": observed_primary_cells,
            "primary_complete": observed_primary_cells == expected_primary_cells,
            "primary_method_rows": {str(k): int(v) for k, v in method_counts.items()},
            "results_rows_total": int(len(results)),
            "calibration_rows_total": int(len(cal)),
        },
        "orientation_task_relation": relation,
        "output_rows": files,
    }
    (outdir / "paper_scale_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
