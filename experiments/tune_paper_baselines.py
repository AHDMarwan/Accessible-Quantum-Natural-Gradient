"""Tune the additional paper baselines without touching the published test split.

Only the baselines that were not part of the pre-scale four-method validation gate
are tuned here.  The data source is Iris01, but the *existing training partition*
is split again deterministically into nested train/validation subsets; the original
test partition and the U(1) control are never used for selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import aqng_v2_benchmark as v2
import aqng_validation_gate as gate


METRIC_GRID = {
    "lr": [0.015, 0.03, 0.06],
    "lam": [3e-4, 1e-3, 3e-3],
}
EUCLIDEAN_GRID = {
    "SGD": [0.005, 0.015, 0.03, 0.06, 0.10],
    "Adam": [0.001, 0.003, 0.01, 0.03, 0.06],
}


def nested_data() -> dict:
    base = v2.v1.make_data("iris01", 60, 6)
    ids = np.arange(len(base["X_train"]))
    tr, va = train_test_split(
        ids,
        test_size=0.25,
        random_state=271828,
        stratify=np.asarray(base["y_train"], dtype=int),
    )
    return {
        "X_train": np.asarray(base["X_train"])[tr],
        "y_train": np.asarray(base["y_train"])[tr],
        "y_train_pm": np.asarray(base["y_train_pm"])[tr],
        "X_test": np.asarray(base["X_train"])[va],
        "y_test": np.asarray(base["y_train"])[va],
        "y_test_pm": np.asarray(base["y_train_pm"])[va],
    }


def cfg_for(*, lr: float, lam: float) -> v2.Config:
    return v2.Config(
        dataset="iris01",
        seed=0,
        n_qubits=6,
        n_layers=3,
        n_samples=60,
        steps=8,
        lr=float(lr),
        lam=float(lam),
        loss_batch=4,
        metric_batch=2,
        metric_every=2,
        cov_lam=0.0,
        suite="full",
        readout_order=1,
        alignment_batch=4,
        alignment_tangents=64,
        alignment_eval_tangents=64,
        metric_normalization="trace",
        damping_mode="absolute",
        max_direction_norm=8.0,
        max_metric_step=0.25,
    )


def run_candidate(method: str, lr: float, lam: float, data: dict) -> dict:
    cfg = cfg_for(lr=lr, lam=lam)
    batches = v2._schedule(len(data["X_train"]), cfg)
    theta0 = v2.v1.init_theta(
        v2.v1.parameter_count("su2_haar", cfg.n_qubits, cfg.n_layers), cfg.seed
    )
    calibration_bundle = gate._ORIGINAL_BUNDLE(
        "su2_haar", cfg, stream_tag="nested-tuning-calibration"
    )
    designs, _ = gate.stable_builder(
        calibration_bundle, theta0, data["X_train"], cfg
    )
    bundle = gate._ORIGINAL_BUNDLE(
        "su2_haar", cfg, stream_tag=f"nested-tuning-{method}-{lr}-{lam}"
    )
    out = gate._ORIGINAL_TRAIN(bundle, method, cfg, data, batches, designs)
    r = out["result"]
    return {
        "method": method,
        "lr": float(lr),
        "lam": float(lam),
        "validation_loss": float(r["test_loss"]),
        "validation_acc": float(r["test_acc"]),
        "train_loss": float(r["train_loss"]),
        "seconds": float(r["total_seconds"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--candidates-csv", type=Path, default=None)
    a = p.parse_args()

    data = nested_data()
    rows: list[dict] = []
    for method in ("AQNG-Z0", "Block-QNG"):
        for lr in METRIC_GRID["lr"]:
            for lam in METRIC_GRID["lam"]:
                rows.append(run_candidate(method, lr, lam, data))
    for method, lrs in EUCLIDEAN_GRID.items():
        for lr in lrs:
            rows.append(run_candidate(method, lr, 3e-3, data))

    df = pd.DataFrame(rows)
    selected = {}
    for method, g in df.groupby("method"):
        # Primary selection: minimum validation loss; deterministic ties prefer
        # higher accuracy, then smaller LR, then larger damping for metric methods.
        g = g.sort_values(
            ["validation_loss", "validation_acc", "lr", "lam"],
            ascending=[True, False, True, False],
        )
        best = g.iloc[0]
        selected[str(method)] = {
            "lr": float(best.lr),
            "lam": float(best.lam),
            "validation_loss": float(best.validation_loss),
            "validation_acc": float(best.validation_acc),
        }

    payload = {
        "protocol": (
            "Nested validation on Iris01/SU2-Haar only; seed 0; the existing training "
            "partition is split with random_state=271828. The published test split and "
            "U(1) control are not used for hyperparameter selection."
        ),
        "fixed_orientation_hparams": {"lr": 0.06, "lam": 0.003},
        "metric_grid": METRIC_GRID,
        "euclidean_lr_grid": EUCLIDEAN_GRID,
        "selected": selected,
        "freeze_rule": "Selected values are frozen before paper-scale outcome inspection.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = a.candidates_csv or a.output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
