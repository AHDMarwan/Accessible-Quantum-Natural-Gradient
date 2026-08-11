"""Small executable correctness gate for the AQNG v2 benchmark."""

from dataclasses import replace

import numpy as np

import aqng_v2_benchmark as v2
import paper_classification as v1


def _assert_finite_result(out):
    result = out["result"]
    for key in ("train_loss", "test_loss", "train_acc", "test_acc", "total_seconds"):
        if not np.isfinite(float(result[key])):
            raise AssertionError(f"non-finite {key}: {result[key]}")
    if not np.all(np.isfinite(out["first_grad"])):
        raise AssertionError("non-finite initial gradient")
    if not np.all(np.isfinite(out["first_direction"])):
        raise AssertionError("non-finite initial direction")


def analytic_smoke():
    cfg = v2.Config(
        dataset="iris01",
        seed=0,
        n_qubits=4,
        n_layers=1,
        n_samples=20,
        steps=1,
        lr=0.02,
        lam=1e-3,
        loss_batch=2,
        metric_batch=1,
        metric_every=1,
        rcond=1e-8,
        suite="full",
        readout_order=1,
        alignment_batch=2,
        alignment_tangents=8,
        alignment_eval_tangents=8,
        metric_normalization="trace",
        max_direction_norm=5.0,
        max_metric_step=0.2,
    )
    data = v1.make_data(cfg.dataset, cfg.n_samples, cfg.n_qubits)
    family = "ryrz_cz"
    p = v1.parameter_count(family, cfg.n_qubits, cfg.n_layers)
    theta0 = v1.init_theta(p, cfg.seed)

    calibration_bundle = v2.V2CircuitBundle(family, cfg, stream_tag="smoke-calibration")
    designs, diag = v2.build_rank_matched_readouts(
        calibration_bundle, theta0, data["X_train"], cfg
    )
    ranks = {d.rank for d in designs.values()}
    if len(ranks) != 1:
        raise AssertionError(f"rank-matched readouts disagree: {ranks}")
    if diag["aligned_crossfit_heldout_retention"] < 0.0:
        raise AssertionError("invalid aligned retention")

    batches = v2._schedule(len(data["X_train"]), cfg)
    for method in cfg.methods():
        bundle = v2.V2CircuitBundle(family, cfg, stream_tag=f"smoke-{method}")
        out = v2.train_method(bundle, method, cfg, data, batches, designs)
        _assert_finite_result(out)
    print("analytic AQNG-v2 smoke: ok")


def finite_shot_smoke():
    base = v2.Config(
        dataset="iris01",
        seed=1,
        n_qubits=2,
        n_layers=1,
        n_samples=20,
        steps=1,
        lr=0.01,
        lam=1e-2,
        loss_batch=2,
        metric_batch=1,
        metric_every=1,
        rcond=1e-6,
        cov_lam=1e-2,
        suite="orientation",
        readout_order=1,
        alignment_batch=2,
        alignment_tangents=8,
        alignment_eval_tangents=8,
        metric_normalization="trace",
        max_direction_norm=5.0,
        max_metric_step=0.2,
        shots=200,
        finite_shot_calibration=True,
    )
    # Only the finite-shot AQNG path is needed in this gate.  Full-QNG is an
    # analytic oracle in shot mode and is already covered by the analytic smoke.
    cfg = replace(base, suite="orientation")
    data = v1.make_data(cfg.dataset, cfg.n_samples, cfg.n_qubits)
    family = "ryrz_cz"
    p = v1.parameter_count(family, cfg.n_qubits, cfg.n_layers)
    theta0 = v1.init_theta(p, cfg.seed)
    calibration_bundle = v2.V2CircuitBundle(family, cfg, stream_tag="shot-calibration")
    designs, _ = v2.build_rank_matched_readouts(
        calibration_bundle, theta0, data["X_train"], cfg
    )
    batches = v2._schedule(len(data["X_train"]), cfg)
    bundle = v2.V2CircuitBundle(family, cfg, stream_tag="shot-physical")
    out = v2.train_method(
        bundle, "AQNG-physical", cfg, data, batches, designs
    )
    _assert_finite_result(out)
    if out["result"]["shots"] != 200:
        raise AssertionError("finite-shot configuration was not preserved")
    print("finite-shot AQNG-v2 smoke: ok")


if __name__ == "__main__":
    analytic_smoke()
    finite_shot_smoke()
    print("AQNG V2 SMOKE PASSED")
