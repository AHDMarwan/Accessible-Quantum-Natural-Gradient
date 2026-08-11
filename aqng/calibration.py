"""Automatic calibration helpers for the public AQNG optimizer API."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pennylane as qml

from aqng_readouts import normalized_reference_score_rows

ArrayFn = Callable[..., object]


def calibration_score_rows(
    probability_fn: ArrayFn,
    params,
    *args,
    directions: Optional[np.ndarray] = None,
    n_directions: int = 64,
    seed: int = 0,
    reference_probabilities: Optional[np.ndarray] = None,
    probability_floor: float = 1e-13,
    tangent_floor: float = 1e-13,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized calibration score rows directly from a probability callable.

    ``probability_fn`` must return either one probability vector with shape ``(D,)``
    or a batch with shape ``(B, D)``.  The first function argument is assumed to
    be the trainable parameter array.  The probability Jacobian is differentiated
    automatically with PennyLane/Autograd and flattened over parameter axes.

    When ``directions`` is omitted, isotropic unit directions are drawn in the
    flattened parameter space.  Labels are never consumed by this routine.
    """
    probs = np.asarray(qml.math.toarray(probability_fn(params, *args, **kwargs)), dtype=float)
    if probs.ndim not in (1, 2):
        raise ValueError("probability_fn must return shape (D,) or (B,D)")

    jac_fn = qml.jacobian(probability_fn, argnum=0)
    jac_raw = np.asarray(qml.math.toarray(jac_fn(params, *args, **kwargs)), dtype=float)

    pshape = tuple(qml.math.shape(params))
    p = int(np.prod(pshape)) if pshape else 1
    if probs.ndim == 1:
        if jac_raw.shape[:1] != probs.shape:
            raise ValueError("probability Jacobian output shape is incompatible with probabilities")
        jacs = jac_raw.reshape(1, probs.shape[0], p)
        probs_batch = probs[None, :]
    else:
        if jac_raw.shape[:2] != probs.shape:
            raise ValueError("probability Jacobian output shape is incompatible with probabilities")
        jacs = jac_raw.reshape(probs.shape[0], probs.shape[1], p)
        probs_batch = probs

    if directions is None:
        if int(n_directions) < 1:
            raise ValueError("n_directions must be >= 1")
        rng = np.random.default_rng(int(seed))
        directions = rng.normal(size=(int(n_directions), p))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        bad = norms[:, 0] <= 0.0
        while np.any(bad):
            directions[bad] = rng.normal(size=(int(np.sum(bad)), p))
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            bad = norms[:, 0] <= 0.0
        directions = directions / norms
    else:
        directions = np.asarray(directions, dtype=float)
        if directions.ndim == 1:
            directions = directions[None, :]
        if directions.ndim != 2 or directions.shape[1] != p:
            raise ValueError(f"directions must have shape (M,{p})")

    return normalized_reference_score_rows(
        probs_batch,
        jacs,
        directions,
        reference_probabilities=reference_probabilities,
        probability_floor=probability_floor,
        tangent_floor=tangent_floor,
    )
