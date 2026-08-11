"""AQNG v2 benchmark: orientation controls, scale controls, baselines, finite shots.

This experiment is intentionally separate from ``paper_classification.py`` so the
published v1 benchmark remains frozen.  It implements the four follow-up controls
motivated by the spectral-geometry paper:

1. same-rank ``physical`` vs ``random_rank`` vs ``aligned_crossfit`` AQNG;
2. metric-scale normalization, relative damping, per-method LR/lambda overrides,
   and Euclidean / accessible-metric trust radii;
3. SGD, Adam, block-QNG, and task-head-only (Z0) AQNG baselines;
4. end-to-end finite-shot AQNG for both the loss gradient and accessible metric.

The rank-matched readouts are fitted at the common initial parameter point from a
label-free calibration set.  ``aligned_crossfit`` uses one set of random tangent
directions to fit its score subspace and an independent set for orientation
diagnostics.  The learned outcome functions are then frozen during training.

When ``--shots`` is supplied, the AQNG metric uses finite-shot ``qml.probs`` and a
parameter-shift probability Jacobian; the loss gradient uses a finite-shot Z0
expectation QNode and parameter shift.  Full-QNG and block-QNG remain analytic
oracle baselines in this mode and are labeled as such.  Analytic evaluation is used
for terminal train/test metrics so shot noise in evaluation does not obscure the
optimizer comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from tqdm.auto import tqdm

try:  # running as ``python experiments/aqng_v2_benchmark.py``
    import paper_classification as v1
except ImportError:  # pragma: no cover - useful for interactive imports
    from experiments import paper_classification as v1

from aqng_readouts import (
    accessible_metric_from_probability_jacobians,
    fit_rank_matched_readouts,
    normalized_reference_score_rows,
    readout_retention,
    solve_controlled_direction,
)


ORIENTATION_METHODS = (
    "AQNG-physical",
    "AQNG-random",
    "AQNG-aligned",
    "Full-QNG",
)
FULL_METHODS = (
    "AQNG-physical",
    "AQNG-random",
    "AQNG-aligned",
    "AQNG-Z0",
    "Full-QNG",
    "Block-QNG",
    "SGD",
    "Adam",
)
AQNG_METHODS = {
    "AQNG-physical": "physical",
    "AQNG-random": "random_rank",
    "AQNG-aligned": "aligned_crossfit",
    "AQNG-Z0": "Z0",
}
METRIC_METHODS = set(AQNG_METHODS) | {"Full-QNG", "Block-QNG"}


def _stable_text_seed(text: str, base: int) -> int:
    acc = int(base) & 0x7FFFFFFF
    for i, ch in enumerate(text):
        acc = (acc * 1664525 + (i + 1) * ord(ch) + 1013904223) & 0x7FFFFFFF
    return int(acc)


def _parse_overrides(items: Iterable[str]) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be METHOD=value, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in FULL_METHODS:
            raise ValueError(f"unknown method in override: {name!r}")
        out.append((name, float(value)))
    return tuple(out)


@dataclass(frozen=True)
class Config:
    dataset: str
    seed: int
    n_qubits: int = 6
    n_layers: int = 3
    n_samples: int = 80
    steps: int = 20
    lr: float = 0.03
    lam: float = 1e-3
    loss_batch: int = 8
    metric_batch: int = 4
    metric_every: int = 2
    rcond: float = 1e-8
    cov_lam: float = 0.0
    suite: str = "full"
    readout_order: int = 2
    alignment_batch: int = 4
    alignment_tangents: int = 64
    alignment_eval_tangents: int = 64
    metric_normalization: str = "trace"
    damping_mode: str = "absolute"
    max_direction_norm: Optional[float] = 8.0
    max_metric_step: Optional[float] = 0.25
    shots: Optional[int] = None
    finite_shot_calibration: bool = False
    lr_overrides: tuple[tuple[str, float], ...] = ()
    lam_overrides: tuple[tuple[str, float], ...] = ()
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    def methods(self) -> tuple[str, ...]:
        if self.suite == "orientation":
            return ORIENTATION_METHODS
        if self.suite == "full":
            return FULL_METHODS
        raise ValueError("suite must be 'orientation' or 'full'")

    def method_lr(self, method: str) -> float:
        return dict(self.lr_overrides).get(method, self.lr)

    def method_lam(self, method: str) -> float:
        return dict(self.lam_overrides).get(method, self.lam)


def _validate_config(cfg: Config) -> None:
    if cfg.dataset not in v1.DATASETS:
        raise ValueError(f"unknown dataset {cfg.dataset!r}")
    if cfg.n_qubits % 2:
        raise ValueError("n_qubits must be even for the half-filled U(1) control")
    if cfg.steps < 1 or cfg.loss_batch < 1 or cfg.metric_batch < 1:
        raise ValueError("steps and batch sizes must be positive")
    if cfg.metric_every < 1:
        raise ValueError("metric_every must be >= 1")
    if cfg.alignment_batch < 1:
        raise ValueError("alignment_batch must be positive")
    if cfg.alignment_tangents < 1 or cfg.alignment_eval_tangents < 1:
        raise ValueError("alignment tangent counts must be positive")
    if cfg.readout_order < 1:
        raise ValueError("readout_order must be >= 1")
    if cfg.cov_lam < 0:
        raise ValueError("cov_lam must be nonnegative")
    if cfg.shots is not None and cfg.shots < 1:
        raise ValueError("shots must be positive or None")
    if cfg.finite_shot_calibration and cfg.shots is None:
        raise ValueError("finite_shot_calibration requires --shots")
    if cfg.metric_normalization not in ("none", "trace", "maxeig"):
        raise ValueError("invalid metric_normalization")
    if cfg.damping_mode not in ("absolute", "mean_eig", "maxeig"):
        raise ValueError("invalid damping_mode")


def _unit_directions(rng: onp.random.Generator, count: int, p: int) -> onp.ndarray:
    v = rng.normal(size=(int(count), int(p)))
    norms = onp.linalg.norm(v, axis=1, keepdims=True)
    bad = norms[:, 0] <= 1e-15
    while onp.any(bad):
        v[bad] = rng.normal(size=(int(onp.sum(bad)), p))
        norms = onp.linalg.norm(v, axis=1, keepdims=True)
        bad = norms[:, 0] <= 1e-15
    return v / norms


def _z0_features(n_qubits: int) -> onp.ndarray:
    states = onp.arange(2**n_qubits, dtype=onp.int64)
    bit0 = (states >> (n_qubits - 1)) & 1
    return (1.0 - 2.0 * bit0.astype(float))[:, None]


class V2CircuitBundle(v1.CircuitBundle):
    """v1 circuits plus differentiable full probabilities and finite-shot paths."""

    def __init__(self, family: str, cfg: Config, *, stream_tag: str):
        super().__init__(family, cfg)
        self.v2cfg = cfg
        n = cfg.n_qubits

        def circuit(theta, x):
            v1.apply_family(
                family,
                theta,
                x,
                cfg.n_qubits,
                cfg.n_layers,
                self.haar_blocks,
            )

        # A differentiable full-probability path is needed for arbitrary outcome
        # functions (physical/random/aligned all use this same record).
        self.prob_analytic_device = qml.device("default.qubit", wires=n, shots=None)

        @qml.qnode(
            self.prob_analytic_device,
            interface="autograd",
            diff_method="backprop",
            cache=False,
        )
        def probs_diff(theta, x):
            circuit(theta, x)
            return qml.probs(wires=range(n))

        self.probs_diff = probs_diff

        # Analytic block-QNG comparator.  qml.metric_tensor is the FS metric, so
        # we multiply by 4 below to match the QFIM convention used by v1.
        self.block_metric_device = qml.device("default.qubit", wires=n, shots=None)

        @qml.qnode(self.block_metric_device, interface="autograd", cache=False)
        def block_base(theta, x):
            circuit(theta, x)
            return qml.expval(qml.PauliZ(0))

        self.block_metric_one = qml.metric_tensor(block_base, approx="block-diag")

        self.loss_shot_device = None
        self.prob_shot_device = None
        self.pred_shot = None
        self.probs_shot = None
        if cfg.shots is not None:
            base = 800003 + 1009 * int(cfg.seed)
            shot_seed = _stable_text_seed(stream_tag, base)
            self.loss_shot_device = qml.device(
                "default.qubit", wires=n, shots=int(cfg.shots), seed=shot_seed
            )
            self.prob_shot_device = qml.device(
                "default.qubit", wires=n, shots=int(cfg.shots), seed=shot_seed + 17
            )

            @qml.qnode(
                self.loss_shot_device,
                interface="autograd",
                diff_method="parameter-shift",
                cache=False,
            )
            def pred_shot(theta, x):
                circuit(theta, x)
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(
                self.prob_shot_device,
                interface="autograd",
                diff_method="parameter-shift",
                cache=False,
            )
            def probs_shot(theta, x):
                circuit(theta, x)
                return qml.probs(wires=range(n))

            self.pred_shot = pred_shot
            self.probs_shot = probs_shot

    @property
    def train_pred(self):
        return self.pred if self.v2cfg.shots is None else self.pred_shot

    def probability_qnode(self, *, use_shots: bool):
        if use_shots:
            if self.probs_shot is None:
                raise ValueError("finite-shot probabilities requested without shots")
            return self.probs_shot
        return self.probs_diff

    def probability_batch_and_jacobian(
        self,
        theta,
        X,
        *,
        use_shots: bool,
    ) -> tuple[onp.ndarray, onp.ndarray]:
        X = onp.asarray(X)
        qnode = self.probability_qnode(use_shots=use_shots)
        probs: list[onp.ndarray] = []
        jacs: list[onp.ndarray] = []
        for x in X:
            jac_fn = qml.jacobian(lambda th: qnode(th, x), argnums=0)
            jac = v1.tonp(jac_fn(theta)).reshape(2**self.v2cfg.n_qubits, self.p)
            forward = getattr(jac_fn, "forward", None)
            if forward is None:
                p = v1.tonp(qnode(theta, x)).reshape(-1)
            else:
                p = v1.tonp(forward).reshape(-1)
            probs.append(p)
            jacs.append(jac)
        return onp.vstack(probs), onp.stack(jacs, axis=0)

    def outcome_aqng_metric(
        self,
        theta,
        X,
        outcome_features: onp.ndarray,
        *,
        use_shots: bool,
    ) -> tuple[onp.ndarray, dict]:
        p, jp = self.probability_batch_and_jacobian(theta, X, use_shots=use_shots)
        return accessible_metric_from_probability_jacobians(
            p,
            jp,
            outcome_features,
            rcond=self.v2cfg.rcond,
            cov_lam=self.v2cfg.cov_lam,
        )

    def block_qng_metric(self, theta, X) -> onp.ndarray:
        G = onp.zeros((self.p, self.p), dtype=float)
        for x in onp.asarray(X):
            G += 4.0 * v1.tonp(self.block_metric_one(theta, x)).reshape(self.p, self.p)
        G /= float(len(X))
        return 0.5 * (G + G.T)


def _tracker_value(tracker, key: str) -> int:
    if tracker is None:
        return 0
    value = tracker.totals.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _alignment_ids(n_train: int, cfg: Config) -> onp.ndarray:
    rng = onp.random.default_rng(41003 + int(cfg.seed))
    return onp.sort(
        rng.choice(n_train, size=min(cfg.alignment_batch, n_train), replace=False)
    )


def build_rank_matched_readouts(
    bundle: V2CircuitBundle,
    theta0,
    X_train: onp.ndarray,
    cfg: Config,
) -> tuple[dict, dict]:
    """Fit/read out fixed same-rank physical/random/aligned outcome functions."""

    ids = _alignment_ids(len(X_train), cfg)
    use_shots = bool(cfg.finite_shot_calibration)
    tracker = (
        qml.Tracker(bundle.prob_shot_device)
        if use_shots and bundle.prob_shot_device is not None
        else None
    )
    context = tracker if tracker is not None else nullcontext()
    with context:
        probs, jacs = bundle.probability_batch_and_jacobian(
            theta0, X_train[ids], use_shots=use_shots
        )

    p_ref = onp.mean(probs, axis=0)
    rng_fit = onp.random.default_rng(51003 + int(cfg.seed))
    rng_eval = onp.random.default_rng(61003 + int(cfg.seed))
    fit_dirs = _unit_directions(rng_fit, cfg.alignment_tangents, bundle.p)
    eval_dirs = _unit_directions(rng_eval, cfg.alignment_eval_tangents, bundle.p)

    fit_rows, p_ref, support = normalized_reference_score_rows(
        probs,
        jacs,
        fit_dirs,
        reference_probabilities=p_ref,
    )
    designs = dict(
        fit_rank_matched_readouts(
            p_ref,
            fit_rows,
            n_qubits=cfg.n_qubits,
            readout_order=cfg.readout_order,
            seed=71003 + int(cfg.seed),
        )
    )
    eval_rows, _, _ = normalized_reference_score_rows(
        probs,
        jacs,
        eval_dirs,
        reference_probabilities=p_ref,
    )

    rank = designs["physical"].rank
    centered_dim = designs["physical"].centered_dimension
    rank_baseline = rank / centered_dim
    diag = {
        "alignment_ids": ids.tolist(),
        "alignment_tangents": int(cfg.alignment_tangents),
        "alignment_eval_tangents": int(cfg.alignment_eval_tangents),
        "physical_rank": int(rank),
        "centered_score_dimension": int(centered_dim),
        "rank_baseline": float(rank_baseline),
        "reference_support_size": int(len(support)),
        "calibration_finite_shot": bool(use_shots),
        "calibration_executions": _tracker_value(tracker, "executions"),
        "calibration_shots": _tracker_value(tracker, "shots"),
    }
    for name, design in designs.items():
        retention = readout_retention(eval_rows, design)
        diag[f"{name}_heldout_retention"] = float(retention)
        diag[f"{name}_heldout_rho"] = float(retention / rank_baseline)
    return designs, diag


def _schedule(n_train: int, cfg: Config) -> list[onp.ndarray]:
    rng = onp.random.default_rng(20003 + int(cfg.seed))
    return [
        rng.choice(n_train, size=min(cfg.loss_batch, n_train), replace=False)
        for _ in range(cfg.steps)
    ]


def _spectral_summary(G: onp.ndarray, rcond: float) -> dict:
    eig = onp.maximum(onp.linalg.eigvalsh(0.5 * (G + G.T)), 0.0)
    mx = float(eig[-1]) if eig.size else 0.0
    scale = max(mx, 1.0)
    pos = eig[eig > rcond * scale]
    return {
        "metric_rank": int(len(pos)),
        "metric_trace": float(onp.trace(G)),
        "metric_max_eig": mx,
        "metric_condition": (
            float(pos[-1] / pos[0]) if len(pos) > 1 else (1.0 if len(pos) == 1 else onp.inf)
        ),
    }


def _metric_for_method(
    bundle: V2CircuitBundle,
    method: str,
    theta,
    X_metric: onp.ndarray,
    designs: dict,
    z0_features: onp.ndarray,
    cfg: Config,
) -> tuple[onp.ndarray, dict, str]:
    if method in AQNG_METHODS:
        key = AQNG_METHODS[method]
        features = z0_features if key == "Z0" else designs[key].outcome_features
        G, diag = bundle.outcome_aqng_metric(
            theta,
            X_metric,
            features,
            use_shots=cfg.shots is not None,
        )
        estimator = "finite_shot_parameter_shift" if cfg.shots is not None else "analytic"
        return G, diag, estimator
    if method == "Full-QNG":
        return bundle.full_qng_metric(theta, X_metric), {}, (
            "analytic_oracle" if cfg.shots is not None else "analytic"
        )
    if method == "Block-QNG":
        return bundle.block_qng_metric(theta, X_metric), {}, (
            "analytic_oracle" if cfg.shots is not None else "analytic"
        )
    raise ValueError(f"method {method!r} does not use a metric")


def _make_cost(pred, Xb, yb):
    Xb = onp.asarray(Xb)
    yb = onp.asarray(yb)

    def cost(theta):
        return qml.math.mean((pred(theta, Xb) - yb) ** 2)

    return cost


def _adam_direction(
    grad: onp.ndarray,
    state: dict,
    *,
    beta1: float,
    beta2: float,
    eps: float,
) -> onp.ndarray:
    state["t"] += 1
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
    mhat = state["m"] / (1.0 - beta1 ** state["t"])
    vhat = state["v"] / (1.0 - beta2 ** state["t"])
    return mhat / (onp.sqrt(vhat) + eps)


def train_method(
    bundle: V2CircuitBundle,
    method: str,
    cfg: Config,
    data: dict,
    batches: list[onp.ndarray],
    designs: dict,
) -> dict:
    theta = v1.init_theta(bundle.p, cfg.seed)
    X = data["X_train"]
    ypm = data["y_train_pm"]
    z0 = _z0_features(cfg.n_qubits)
    lr = float(cfg.method_lr(method))
    nominal_lam = float(cfg.method_lam(method))

    G = None
    metric_builds = metric_examples = loss_examples = 0
    metric_seconds = gradient_seconds = solve_seconds = 0.0
    first_grad = first_direction = first_metric = None
    curves: list[dict] = []
    metric_diags: list[dict] = []
    adam = {
        "m": onp.zeros(bundle.p, dtype=float),
        "v": onp.zeros(bundle.p, dtype=float),
        "t": 0,
    }

    loss_tracker = (
        qml.Tracker(bundle.loss_shot_device)
        if cfg.shots is not None and bundle.loss_shot_device is not None
        else None
    )
    metric_tracker = (
        qml.Tracker(bundle.prob_shot_device)
        if cfg.shots is not None
        and method in AQNG_METHODS
        and bundle.prob_shot_device is not None
        else None
    )
    loss_context = loss_tracker if loss_tracker is not None else nullcontext()
    metric_context = metric_tracker if metric_tracker is not None else nullcontext()

    t_total = time.perf_counter()
    with loss_context, metric_context:
        for step, ids in enumerate(batches, 1):
            mids = ids[: min(cfg.metric_batch, len(ids))]
            refresh = method in METRIC_METHODS and (
                G is None or ((step - 1) % cfg.metric_every == 0)
            )

            if refresh:
                t = time.perf_counter()
                G, build_diag, estimator = _metric_for_method(
                    bundle, method, theta, X[mids], designs, z0, cfg
                )
                metric_seconds += time.perf_counter() - t
                metric_builds += 1
                metric_examples += len(mids)
                sd = _spectral_summary(G, cfg.rcond)
                sd.update(
                    step=step,
                    method=method,
                    metric_estimator=estimator,
                    **build_diag,
                )
                metric_diags.append(sd)
                if first_metric is None:
                    first_metric = onp.array(G, copy=True)

            cost = _make_cost(bundle.train_pred, X[ids], ypm[ids])
            grad_fn = qml.grad(cost)
            t = time.perf_counter()
            grad = v1.tonp(grad_fn(theta)).reshape(-1)
            gradient_seconds += time.perf_counter() - t
            loss_examples += len(ids)
            forward = getattr(grad_fn, "forward", None)
            loss_before = float(forward) if forward is not None else float(cost(theta))

            t = time.perf_counter()
            if method in METRIC_METHODS:
                direction, controls = solve_controlled_direction(
                    G,
                    grad,
                    lam=nominal_lam,
                    stepsize=lr,
                    rcond=cfg.rcond,
                    metric_normalization=cfg.metric_normalization,
                    damping_mode=cfg.damping_mode,
                    max_direction_norm=cfg.max_direction_norm,
                    max_metric_step=cfg.max_metric_step,
                )
            elif method == "SGD":
                direction = grad
                controls = {
                    "metric_scale_factor": onp.nan,
                    "effective_damping": onp.nan,
                    "raw_direction_norm": float(onp.linalg.norm(direction)),
                    "direction_norm": float(onp.linalg.norm(direction)),
                    "raw_metric_step_norm": onp.nan,
                    "metric_step_norm": onp.nan,
                    "trust_region_clipped": False,
                    "clip_scale": 1.0,
                }
            elif method == "Adam":
                direction = _adam_direction(
                    grad,
                    adam,
                    beta1=cfg.adam_beta1,
                    beta2=cfg.adam_beta2,
                    eps=cfg.adam_eps,
                )
                controls = {
                    "metric_scale_factor": onp.nan,
                    "effective_damping": onp.nan,
                    "raw_direction_norm": float(onp.linalg.norm(direction)),
                    "direction_norm": float(onp.linalg.norm(direction)),
                    "raw_metric_step_norm": onp.nan,
                    "metric_step_norm": onp.nan,
                    "trust_region_clipped": False,
                    "clip_scale": 1.0,
                }
            else:  # pragma: no cover
                raise RuntimeError(method)
            solve_seconds += time.perf_counter() - t

            if first_grad is None:
                first_grad = onp.array(grad, copy=True)
                first_direction = onp.array(direction, copy=True)

            theta = np.array(v1.tonp(theta) - lr * direction, requires_grad=True)
            curves.append(
                {
                    "step": step,
                    "loss_before": loss_before,
                    "gradient_norm": float(onp.linalg.norm(grad)),
                    "direction_norm": float(onp.linalg.norm(direction)),
                    "parameter_step_norm": float(lr * onp.linalg.norm(direction)),
                    "metric_refreshed": int(refresh),
                    **controls,
                }
            )

    total_seconds = time.perf_counter() - t_total
    train_eval = v1.evaluate(
        bundle.pred,
        theta,
        data["X_train"],
        data["y_train"],
        data["y_train_pm"],
    )
    test_eval = v1.evaluate(
        bundle.pred,
        theta,
        data["X_test"],
        data["y_test"],
        data["y_test_pm"],
    )

    metric_df = pd.DataFrame(metric_diags)
    result = {
        "dataset": cfg.dataset,
        "family": bundle.family,
        "method": method,
        "seed": cfg.seed,
        "n_qubits": cfg.n_qubits,
        "n_layers": cfg.n_layers,
        "n_params": bundle.p,
        "n_samples": cfg.n_samples,
        "steps": cfg.steps,
        "lr": lr,
        "nominal_lam": nominal_lam,
        "metric_normalization": cfg.metric_normalization,
        "damping_mode": cfg.damping_mode,
        "max_direction_norm": cfg.max_direction_norm,
        "max_metric_step": cfg.max_metric_step,
        "shots": cfg.shots,
        "loss_batch": cfg.loss_batch,
        "metric_batch": cfg.metric_batch,
        "metric_every": cfg.metric_every,
        "train_loss": train_eval["loss"],
        "train_acc": train_eval["acc"],
        "test_loss": test_eval["loss"],
        "test_acc": test_eval["acc"],
        "total_seconds": total_seconds,
        "metric_seconds": metric_seconds,
        "gradient_seconds": gradient_seconds,
        "solve_seconds": solve_seconds,
        "metric_builds": metric_builds,
        "metric_examples_total": metric_examples,
        "loss_examples_total": loss_examples,
        "loss_executions": _tracker_value(loss_tracker, "executions"),
        "metric_executions": _tracker_value(metric_tracker, "executions"),
        "loss_shots_total": _tracker_value(loss_tracker, "shots"),
        "metric_shots_total": _tracker_value(metric_tracker, "shots"),
        "metric_oracle_in_shot_mode": bool(
            cfg.shots is not None and method in {"Full-QNG", "Block-QNG"}
        ),
    }
    if len(metric_df):
        result.update(
            metric_rank_first=int(metric_df.iloc[0].metric_rank),
            metric_trace_first=float(metric_df.iloc[0].metric_trace),
            metric_rank_last=int(metric_df.iloc[-1].metric_rank),
            metric_trace_last=float(metric_df.iloc[-1].metric_trace),
        )
    else:
        result.update(
            metric_rank_first=onp.nan,
            metric_trace_first=onp.nan,
            metric_rank_last=onp.nan,
            metric_trace_last=onp.nan,
        )

    return {
        "result": result,
        "curves": curves,
        "metric_diags": metric_diags,
        "first_metric": first_metric,
        "first_grad": first_grad,
        "first_direction": first_direction,
    }


def _cosine(a, b) -> float:
    a = onp.asarray(a, dtype=float).reshape(-1)
    b = onp.asarray(b, dtype=float).reshape(-1)
    na = float(onp.linalg.norm(a))
    nb = float(onp.linalg.norm(b))
    return onp.nan if na == 0.0 or nb == 0.0 else float(onp.dot(a, b) / (na * nb))


def shared_initial_orientation_diagnostic(
    bundle: V2CircuitBundle,
    theta0,
    X_metric: onp.ndarray,
    designs: dict,
    cfg: Config,
) -> dict:
    """Same probability/Jacobian record, same rank; only orientation changes."""

    p, jp = bundle.probability_batch_and_jacobian(
        theta0,
        X_metric,
        use_shots=cfg.shots is not None,
    )
    row = {
        "rank": int(designs["physical"].rank),
        "centered_score_dimension": int(designs["physical"].centered_dimension),
        "same_record_finite_shot": bool(cfg.shots is not None),
    }
    metrics = {}
    for name in ("physical", "random_rank", "aligned_crossfit"):
        G, d = accessible_metric_from_probability_jacobians(
            p,
            jp,
            designs[name].outcome_features,
            rcond=cfg.rcond,
            cov_lam=cfg.cov_lam,
        )
        metrics[name] = G
        row[f"{name}_metric_trace"] = float(onp.trace(G))
        row[f"{name}_metric_rank"] = int(d["metric_rank"])
    row["physical_vs_random_fro"] = float(
        onp.linalg.norm(metrics["physical"] - metrics["random_rank"])
    )
    row["physical_vs_aligned_fro"] = float(
        onp.linalg.norm(metrics["physical"] - metrics["aligned_crossfit"])
    )
    return row


def run_job(cfg: Config, output_dir: Path) -> pd.DataFrame:
    _validate_config(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = v1.make_data(cfg.dataset, cfg.n_samples, cfg.n_qubits)
    batches = _schedule(len(data["X_train"]), cfg)

    rows: list[dict] = []
    curves: list[dict] = []
    mdiags: list[dict] = []
    orientation_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for family in tqdm(v1.FAMILIES, desc=f"AQNG-v2 {cfg.dataset} seed={cfg.seed}"):
        theta0 = v1.init_theta(v1.parameter_count(family, cfg.n_qubits, cfg.n_layers), cfg.seed)
        calibration_bundle = V2CircuitBundle(family, cfg, stream_tag="calibration")
        designs, calibration = build_rank_matched_readouts(
            calibration_bundle, theta0, data["X_train"], cfg
        )
        calibration_rows.append(
            {"dataset": cfg.dataset, "family": family, "seed": cfg.seed, **calibration}
        )

        first_ids = batches[0][: min(cfg.metric_batch, len(batches[0]))]
        shared_bundle = V2CircuitBundle(family, cfg, stream_tag="shared-initial")
        orientation_rows.append(
            {
                "dataset": cfg.dataset,
                "family": family,
                "seed": cfg.seed,
                **shared_initial_orientation_diagnostic(
                    shared_bundle,
                    theta0,
                    data["X_train"][first_ids],
                    designs,
                    cfg,
                ),
            }
        )

        outputs = {}
        for method in cfg.methods():
            bundle = V2CircuitBundle(family, cfg, stream_tag=method)
            out = train_method(bundle, method, cfg, data, batches, designs)
            outputs[method] = out
            rows.append(out["result"])
            curves += [
                {
                    "dataset": cfg.dataset,
                    "family": family,
                    "method": method,
                    "seed": cfg.seed,
                    **r,
                }
                for r in out["curves"]
            ]
            mdiags += [
                {
                    "dataset": cfg.dataset,
                    "family": family,
                    "method": method,
                    "seed": cfg.seed,
                    **r,
                }
                for r in out["metric_diags"]
            ]

        # Analytic mode must have exactly paired initial gradients.  In finite-shot
        # mode every method starts from the same device seed/stream, so this is
        # still expected up to stochastic-transform ordering; record rather than
        # fail if a backend changes that ordering.
        ref = outputs[cfg.methods()[0]]["first_grad"]
        for method, out in outputs.items():
            diff = float(onp.max(onp.abs(ref - out["first_grad"])))
            out["result"]["initial_gradient_max_abs_diff_to_first_method"] = diff
            if cfg.shots is None and diff > 1e-9:
                raise RuntimeError(
                    f"analytic paired initial gradients differ for {family}/{method}: {diff}"
                )

        if "Full-QNG" in outputs:
            qdir = outputs["Full-QNG"]["first_direction"]
            for method, out in outputs.items():
                out["result"]["initial_direction_cosine_to_full_qng"] = _cosine(
                    out["first_direction"], qdir
                )

    stem = f"{cfg.dataset}_seed{cfg.seed:03d}"
    pd.DataFrame(rows).to_csv(output_dir / f"results_{stem}.csv", index=False)
    pd.DataFrame(curves).to_csv(output_dir / f"curves_{stem}.csv", index=False)
    pd.DataFrame(mdiags).to_csv(output_dir / f"metric_diags_{stem}.csv", index=False)
    pd.DataFrame(orientation_rows).to_csv(
        output_dir / f"orientation_diags_{stem}.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / f"calibration_diags_{stem}.csv", index=False
    )

    meta = {
        "config": cfg.__dict__,
        "methods": list(cfg.methods()),
        "subset_seed": v1.SUBSET_SEED,
        "split_seed": v1.SPLIT_SEED,
        "orientation_control": (
            "physical/random/aligned readouts have the same covariance rank and are "
            "constructed in one reference-mixture computational-basis score space"
        ),
        "aligned_crossfit": (
            "fitted from label-free calibration inputs and independent tangent directions; "
            "held-out tangent directions are used for retention diagnostics"
        ),
        "finite_shot_semantics": (
            "with --shots, AQNG probability metrics and loss gradients are finite-shot "
            "parameter-shift estimates; Full-QNG and Block-QNG are analytic oracle metrics; "
            "terminal evaluation is analytic"
        ),
        "source_geometry_repository": (
            "AHDMarwan/Spectral-Geometry-of-Accessible-Quantum-Tangents-Beyond-"
            "Isotropic-Readout-Rank-Laws"
        ),
        "pennylane": qml.__version__,
        "numpy": onp.__version__,
        "python": platform.python_version(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    (output_dir / f"config_{stem}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=v1.DATASETS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--suite", choices=("orientation", "full"), default="full")
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=80)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--loss-batch", type=int, default=8)
    p.add_argument("--metric-batch", type=int, default=4)
    p.add_argument("--metric-every", type=int, default=2)
    p.add_argument("--rcond", type=float, default=1e-8)
    p.add_argument("--cov-lam", type=float, default=0.0)
    p.add_argument("--readout-order", type=int, default=2)
    p.add_argument("--alignment-batch", type=int, default=4)
    p.add_argument("--alignment-tangents", type=int, default=64)
    p.add_argument("--alignment-eval-tangents", type=int, default=64)
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
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--finite-shot-calibration", action="store_true")
    p.add_argument(
        "--lr-override",
        action="append",
        default=[],
        metavar="METHOD=VALUE",
        help="per-method learning-rate override; may be repeated",
    )
    p.add_argument(
        "--lam-override",
        action="append",
        default=[],
        metavar="METHOD=VALUE",
        help="per-method damping override; may be repeated",
    )
    a = p.parse_args()

    cfg = Config(
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
        rcond=a.rcond,
        cov_lam=a.cov_lam,
        suite=a.suite,
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
        lr_overrides=_parse_overrides(a.lr_override),
        lam_overrides=_parse_overrides(a.lam_override),
    )
    df = run_job(cfg, a.output_dir)
    cols = [
        "dataset",
        "family",
        "method",
        "seed",
        "test_loss",
        "test_acc",
        "total_seconds",
        "loss_shots_total",
        "metric_shots_total",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
