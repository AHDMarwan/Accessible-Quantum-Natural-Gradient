"""Pre-large-scale AQNG validation gate.

Adds fixed support/pseudocount calibration, optional paired finite-shot RNG streams,
first-step task diagnostics, and explicit calibration+training resource accounting
without changing the frozen AQNG-v2 benchmark implementation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np

import aqng_v2_benchmark as v2
from aqng_readouts import readout_retention
from aqng_validation import (
    fixed_support_indices,
    fit_stable_rank_matched_readouts,
    normalized_reference_score_rows_fixed_support,
    stabilize_reference_probabilities,
)

_ORIGINAL_BUNDLE = v2.V2CircuitBundle
_ORIGINAL_TRAIN = v2.train_method


def stable_builder(bundle, theta0, X_train, cfg):
    ids = v2._alignment_ids(len(X_train), cfg)
    use_shots = bool(cfg.finite_shot_calibration)
    tracker = qml.Tracker(bundle.prob_shot_device) if use_shots else None
    with (tracker if tracker is not None else nullcontext()):
        probs, jacs = bundle.probability_batch_and_jacobian(
            theta0, X_train[ids], use_shots=use_shots
        )
    support = fixed_support_indices(cfg.n_qubits, bundle.family)
    raw_ref = onp.mean(probs, axis=0)
    p_ref = stabilize_reference_probabilities(
        raw_ref,
        support,
        shots=cfg.shots if use_shots else None,
        pseudocount=float(getattr(cfg, "support_pseudocount", 0.5)),
    )
    rng_fit = onp.random.default_rng(51003 + int(cfg.seed))
    rng_eval = onp.random.default_rng(61003 + int(cfg.seed))
    fit_dirs = v2._unit_directions(rng_fit, cfg.alignment_tangents, bundle.p)
    eval_dirs = v2._unit_directions(rng_eval, cfg.alignment_eval_tangents, bundle.p)
    fit_rows = normalized_reference_score_rows_fixed_support(
        probs, jacs, fit_dirs,
        reference_probabilities=p_ref,
        support_indices=support,
    )
    designs = fit_stable_rank_matched_readouts(
        p_ref, fit_rows,
        n_qubits=cfg.n_qubits,
        readout_order=cfg.readout_order,
        seed=71003 + int(cfg.seed),
    )
    eval_rows = normalized_reference_score_rows_fixed_support(
        probs, jacs, eval_dirs,
        reference_probabilities=p_ref,
        support_indices=support,
    )
    rank = designs["physical"].rank
    N = designs["physical"].centered_dimension
    baseline = rank / N
    diag = {
        "alignment_ids": ids.tolist(),
        "alignment_tangents": int(cfg.alignment_tangents),
        "alignment_eval_tangents": int(cfg.alignment_eval_tangents),
        "physical_rank": int(rank),
        "centered_score_dimension": int(N),
        "rank_baseline": float(baseline),
        "reference_support_size": int(len(support)),
        "empirical_nonzero_support_size": int(onp.sum(raw_ref > 0)),
        "support_policy": "half_filled_sector" if bundle.family == "u1_rzxy" else "full_basis",
        "support_pseudocount": float(getattr(cfg, "support_pseudocount", 0.5)),
        "calibration_finite_shot": bool(use_shots),
        "calibration_executions": v2._tracker_value(tracker, "executions"),
        "calibration_shots": v2._tracker_value(tracker, "shots"),
    }
    for name, design in designs.items():
        retention = readout_retention(eval_rows, design)
        diag[f"{name}_heldout_retention"] = float(retention)
        diag[f"{name}_heldout_rho"] = float(retention / baseline)
    return designs, diag


def _first_step_task_diag(method, cfg, data, batches, designs, family):
    """Compute the first-step task diagnostic for both metric and Euclidean baselines."""
    # Separate bundle: diagnostics do not consume the training RNG stream.
    b = _ORIGINAL_BUNDLE(family, cfg, stream_tag=f"diag-{method}")
    theta = v2.v1.init_theta(b.p, cfg.seed)
    ids = batches[0]
    mids = ids[: min(cfg.metric_batch, len(ids))]
    cost_train = v2._make_cost(b.train_pred, data["X_train"][ids], data["y_train_pm"][ids])
    grad = v2.v1.tonp(qml.grad(cost_train)(theta)).reshape(-1)
    lr = float(cfg.method_lr(method))

    if method in v2.METRIC_METHODS:
        z0 = v2._z0_features(cfg.n_qubits)
        G, _, _ = v2._metric_for_method(
            b, method, theta, data["X_train"][mids], designs, z0, cfg
        )
        direction, controls = v2.solve_controlled_direction(
            G, grad,
            lam=float(cfg.method_lam(method)),
            stepsize=lr,
            rcond=cfg.rcond,
            metric_normalization=cfg.metric_normalization,
            damping_mode=cfg.damping_mode,
            max_direction_norm=cfg.max_direction_norm,
            max_metric_step=cfg.max_metric_step,
        )
        spec = v2._spectral_summary(G, cfg.rcond)
    elif method == "SGD":
        direction = onp.array(grad, copy=True)
        controls = {"trust_region_clipped": False}
        spec = {"metric_rank": 0, "metric_condition": onp.nan}
    elif method == "Adam":
        state = {
            "m": onp.zeros(b.p, dtype=float),
            "v": onp.zeros(b.p, dtype=float),
            "t": 0,
        }
        direction = v2._adam_direction(
            grad,
            state,
            beta1=cfg.adam_beta1,
            beta2=cfg.adam_beta2,
            eps=cfg.adam_eps,
        )
        controls = {"trust_region_clipped": False}
        spec = {"metric_rank": 0, "metric_condition": onp.nan}
    else:  # pragma: no cover
        raise ValueError(f"unknown method {method!r}")

    theta1 = np.array(v2.v1.tonp(theta) - lr * direction, requires_grad=True)
    # Evaluate actual same-minibatch loss analytically to avoid evaluation-shot noise.
    c0 = v2._make_cost(b.pred, data["X_train"][ids], data["y_train_pm"][ids])
    loss0 = float(c0(theta))
    loss1 = float(c0(theta1))
    return {
        "first_g_dot_d": float(onp.dot(grad, direction)),
        "first_predicted_decrease": float(lr * onp.dot(grad, direction)),
        "first_actual_same_batch_decrease": float(loss0 - loss1),
        "first_actual_same_batch_loss_before": loss0,
        "first_actual_same_batch_loss_after": loss1,
        "first_diag_metric_rank": int(spec["metric_rank"]),
        "first_diag_metric_condition": float(spec["metric_condition"]),
        "first_diag_trust_clipped": bool(controls["trust_region_clipped"]),
    }


def install_validation_hooks(*, paired_shot_streams: bool):
    v2.build_rank_matched_readouts = stable_builder

    class GateBundle(_ORIGINAL_BUNDLE):
        def __init__(self, family, cfg, *, stream_tag):
            if paired_shot_streams and cfg.shots is not None and stream_tag not in {"calibration", "shared-initial"}:
                stream_tag = "paired-training"
            super().__init__(family, cfg, stream_tag=stream_tag)

    v2.V2CircuitBundle = GateBundle

    def train_with_diag(bundle, method, cfg, data, batches, designs):
        diag = _first_step_task_diag(method, cfg, data, batches, designs, bundle.family)
        out = _ORIGINAL_TRAIN(bundle, method, cfg, data, batches, designs)
        out["result"].update(diag)
        return out

    v2.train_method = train_with_diag


def postprocess_resources(output_dir: Path, dataset: str, seed: int) -> pd.DataFrame:
    stem = f"{dataset}_seed{seed:03d}"
    rp = output_dir / f"results_{stem}.csv"
    cp = output_dir / f"calibration_diags_{stem}.csv"
    df = pd.read_csv(rp)
    cal = pd.read_csv(cp)
    cal_by_family = cal.set_index("family")
    df["training_shots_total"] = df["loss_shots_total"].fillna(0) + df["metric_shots_total"].fillna(0)
    df["calibration_shots_shared"] = [float(cal_by_family.loc[f, "calibration_shots"]) for f in df.family]
    df["end_to_end_shots_including_calibration"] = df.training_shots_total + df.calibration_shots_shared
    df["training_executions_total"] = df["loss_executions"].fillna(0) + df["metric_executions"].fillna(0)
    df["calibration_executions_shared"] = [float(cal_by_family.loc[f, "calibration_executions"]) for f in df.family]
    df["end_to_end_executions_including_calibration"] = df.training_executions_total + df.calibration_executions_shared
    df.to_csv(rp, index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=v2.v1.FAMILIES)
    p.add_argument("--dataset", default="iris01", choices=v2.v1.DATASETS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--suite", choices=("orientation", "full"), default="orientation")
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--finite-shot-calibration", action="store_true")
    p.add_argument("--paired-shot-streams", action="store_true")
    p.add_argument("--support-pseudocount", type=float, default=0.5)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=60)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--loss-batch", type=int, default=4)
    p.add_argument("--metric-batch", type=int, default=2)
    p.add_argument("--metric-every", type=int, default=2)
    p.add_argument("--readout-order", type=int, default=1)
    p.add_argument("--alignment-batch", type=int, default=4)
    p.add_argument("--alignment-tangents", type=int, default=64)
    p.add_argument("--alignment-eval-tangents", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--cov-lam", type=float, default=None)
    p.add_argument("--metric-normalization", choices=("none","trace","maxeig"), default="trace")
    p.add_argument("--damping-mode", choices=("absolute","mean_eig","maxeig"), default="absolute")
    p.add_argument("--max-direction-norm", type=float, default=8.0)
    p.add_argument("--max-metric-step", type=float, default=0.25)
    p.add_argument("--lr-override", action="append", default=[])
    p.add_argument("--lam-override", action="append", default=[])
    a = p.parse_args()

    cov_lam = (1e-3 if a.shots is not None else 0.0) if a.cov_lam is None else a.cov_lam
    v2.v1.FAMILIES = (a.family,)
    cfg = v2.Config(
        dataset=a.dataset, seed=a.seed, n_qubits=a.n_qubits, n_layers=a.n_layers,
        n_samples=a.n_samples, steps=a.steps, lr=a.lr, lam=a.lam,
        loss_batch=a.loss_batch, metric_batch=a.metric_batch, metric_every=a.metric_every,
        cov_lam=cov_lam, suite=a.suite, readout_order=a.readout_order,
        alignment_batch=a.alignment_batch, alignment_tangents=a.alignment_tangents,
        alignment_eval_tangents=a.alignment_eval_tangents,
        metric_normalization=a.metric_normalization, damping_mode=a.damping_mode,
        max_direction_norm=a.max_direction_norm, max_metric_step=a.max_metric_step,
        shots=a.shots, finite_shot_calibration=bool(a.finite_shot_calibration),
        lr_overrides=v2._parse_overrides(a.lr_override),
        lam_overrides=v2._parse_overrides(a.lam_override),
    )
    # frozen dataclass: validation-only attributes are attached through class defaults
    type(cfg).support_pseudocount = a.support_pseudocount
    install_validation_hooks(paired_shot_streams=bool(a.paired_shot_streams))
    v2.run_job(cfg, a.output_dir)
    df = postprocess_resources(a.output_dir, a.dataset, a.seed)
    cols = ["family","method","seed","test_loss","test_acc","first_g_dot_d","first_actual_same_batch_decrease","training_shots_total","end_to_end_shots_including_calibration"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
