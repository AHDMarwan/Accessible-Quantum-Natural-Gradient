"""Small targeted AQNG-v2 pilot for one circuit family.

This wrapper deliberately reuses ``aqng_v2_benchmark`` unchanged and restricts the
run to a single architecture.  It is intended for the first scientific pilot:
SU(2)-Haar as the generic positive case and half-filled U(1) as the structured
control, with physical/random/aligned same-rank readouts and Full-QNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import aqng_v2_benchmark as v2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=v2.v1.FAMILIES)
    p.add_argument("--dataset", default="iris01", choices=v2.v1.DATASETS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--finite-shot-calibration", action="store_true")
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=60)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--loss-batch", type=int, default=4)
    p.add_argument("--metric-batch", type=int, default=2)
    p.add_argument("--metric-every", type=int, default=2)
    p.add_argument("--readout-order", type=int, default=1)
    p.add_argument("--alignment-batch", type=int, default=4)
    p.add_argument("--alignment-tangents", type=int, default=32)
    p.add_argument("--alignment-eval-tangents", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--cov-lam", type=float, default=None)
    p.add_argument(
        "--metric-normalization",
        choices=("none", "trace", "maxeig"),
        default="trace",
    )
    p.add_argument(
        "--damping-mode",
        choices=("absolute", "mean_eig", "maxeig"),
        default="absolute",
    )
    p.add_argument("--max-direction-norm", type=float, default=8.0)
    p.add_argument("--max-metric-step", type=float, default=0.25)
    a = p.parse_args()

    cov_lam = (1e-3 if a.shots is not None else 0.0) if a.cov_lam is None else a.cov_lam

    # The v2 runner iterates ``v1.FAMILIES``.  Restricting that tuple here keeps
    # the benchmark implementation itself frozen while making the pilot cheap.
    v2.v1.FAMILIES = (a.family,)

    cfg = v2.Config(
        dataset=a.dataset,
        seed=a.seed,
        n_qubits=a.n_qubits,
        n_layers=a.n_layers,
        n_samples=a.n_samples,
        steps=a.steps,
        lr=a.lr,
        lam=a.lam,
        loss_batch=a.loss_batch,
        metric_batch=a.metric_batch,
        metric_every=a.metric_every,
        cov_lam=cov_lam,
        suite="orientation",
        readout_order=a.readout_order,
        alignment_batch=a.alignment_batch,
        alignment_tangents=a.alignment_tangents,
        alignment_eval_tangents=a.alignment_eval_tangents,
        metric_normalization=a.metric_normalization,
        damping_mode=a.damping_mode,
        max_direction_norm=a.max_direction_norm,
        max_metric_step=a.max_metric_step,
        shots=a.shots,
        finite_shot_calibration=bool(a.finite_shot_calibration),
    )
    df = v2.run_job(cfg, a.output_dir)
    cols = [
        "dataset",
        "family",
        "method",
        "seed",
        "test_loss",
        "test_acc",
        "initial_direction_cosine_to_full_qng",
        "loss_shots_total",
        "metric_shots_total",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
