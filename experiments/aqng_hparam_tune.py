"""Small pre-registered hyperparameter grid for the AQNG validation gate.

Tuning is intentionally restricted to SU(2)-Haar, seed 0, and a nested holdout
from the training partition.  The test split and U(1) structured control are never
used for selection.  The selected per-method (lr, lambda) values are then frozen
for the medium pilot.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as onp
from sklearn.model_selection import train_test_split

import aqng_v2_benchmark as v2
from aqng_validation_gate import stable_builder

METHODS = ("AQNG-physical", "AQNG-random", "AQNG-aligned", "Full-QNG")
LR_GRID = (0.015, 0.03, 0.06)
LAM_GRID = (3e-4, 1e-3, 3e-3)


def nested_data(dataset: str, n_samples: int, n_qubits: int):
    d = v2.v1.make_data(dataset, n_samples, n_qubits)
    idx = onp.arange(len(d["X_train"]))
    it, iv = train_test_split(
        idx, test_size=0.25, random_state=271828, stratify=d["y_train"]
    )
    return {
        "X_train": d["X_train"][it], "y_train": d["y_train"][it], "y_train_pm": d["y_train_pm"][it],
        "X_test": d["X_train"][iv], "y_test": d["y_train"][iv], "y_test_pm": d["y_train_pm"][iv],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dataset", default="iris01")
    p.add_argument("--steps", type=int, default=8)
    a = p.parse_args()

    family, seed, nq, nl, ns = "su2_haar", 0, 6, 3, 60
    data = nested_data(a.dataset, ns, nq)
    base = v2.Config(
        dataset=a.dataset, seed=seed, n_qubits=nq, n_layers=nl, n_samples=ns,
        steps=a.steps, lr=0.03, lam=1e-3, loss_batch=4, metric_batch=2,
        metric_every=2, suite="orientation", readout_order=1,
        alignment_batch=4, alignment_tangents=64, alignment_eval_tangents=64,
        metric_normalization="trace", damping_mode="absolute",
        max_direction_norm=8.0, max_metric_step=0.25,
    )
    theta0 = v2.v1.init_theta(v2.v1.parameter_count(family, nq, nl), seed)
    cal_bundle = v2.V2CircuitBundle(family, base, stream_tag="tune-calibration")
    designs, calibration = stable_builder(cal_bundle, theta0, data["X_train"], base)

    rows = []
    for lr in LR_GRID:
        for lam in LAM_GRID:
            cfg = v2.Config(**{**base.__dict__, "lr": lr, "lam": lam})
            batches = v2._schedule(len(data["X_train"]), cfg)
            b = v2.V2CircuitBundle(family, cfg, stream_tag=f"tune-{a.method}-{lr}-{lam}")
            out = v2.train_method(b, a.method, cfg, data, batches, designs)
            r = out["result"]
            rows.append({"method": a.method, "lr": lr, "lam": lam, "validation_loss": float(r["test_loss"]), "validation_acc": float(r["test_acc"])})

    rows.sort(key=lambda r: (r["validation_loss"], -r["validation_acc"], r["lr"], r["lam"]))
    payload = {
        "selection_protocol": "SU2-Haar seed0 nested training holdout; test and U1 unused",
        "method": a.method,
        "best": rows[0],
        "grid": rows,
        "calibration": calibration,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["best"], indent=2))


if __name__ == "__main__":
    main()
