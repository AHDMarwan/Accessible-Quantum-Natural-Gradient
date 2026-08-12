"""Finite-shot probability stabilization for the public AQNG API."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ArrayFn = Callable[..., object]


def validate_sampling_configuration(
    *,
    shots: Optional[int],
    pseudocount: float,
    support_policy: str,
    support_indices: Optional[Sequence[int]],
) -> None:
    if shots is not None and int(shots) < 1:
        raise ValueError("shots must be a positive integer or None")
    if float(pseudocount) < 0:
        raise ValueError("pseudocount must be nonnegative")
    if support_policy not in {"full", "custom"}:
        raise ValueError("support_policy must be 'full' or 'custom'")
    if support_policy == "custom":
        if support_indices is None or len(tuple(support_indices)) < 2:
            raise ValueError("custom support requires at least two support indices")
        indices = tuple(int(i) for i in support_indices)
        if min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError("support_indices must be unique nonnegative integers")


def _support_mask(dim: int, support_policy: str, support_indices):
    if support_policy == "full":
        return pnp.ones((dim,), requires_grad=False), dim
    indices = tuple(int(i) for i in support_indices)
    if max(indices) >= dim:
        raise ValueError(
            f"support index {max(indices)} is outside probability dimension {dim}"
        )
    mask = np.zeros(dim, dtype=float)
    mask[list(indices)] = 1.0
    return pnp.array(mask, requires_grad=False), len(indices)


def stabilize_probabilities(
    probabilities,
    *,
    shots: Optional[int] = None,
    pseudocount: float = 0.5,
    support_policy: str = "full",
    support_indices: Optional[Sequence[int]] = None,
):
    """Return differentiably stabilized probabilities on a fixed support.

    With finite ``shots``, ``pseudocount`` is a symmetric Dirichlet pseudocount
    in count units. For a support of size ``K`` the regularized probabilities are

    ``(shots * p_i + pseudocount) / (shots + pseudocount * K)``.

    A custom support masks excluded outcomes and renormalizes the retained mass
    before regularization. The operation uses PennyLane math primitives so a
    differentiable probability QNode remains differentiable through this wrapper.
    """
    validate_sampling_configuration(
        shots=shots,
        pseudocount=pseudocount,
        support_policy=support_policy,
        support_indices=support_indices,
    )
    probs = qml.math.asarray(probabilities)
    ndim = qml.math.ndim(probs)
    if ndim not in (1, 2):
        raise ValueError("probability_fn must return shape (D,) or (B,D)")
    dim = int(qml.math.shape(probs)[-1])
    if dim < 2:
        raise ValueError("probability dimension must be at least two")

    mask, support_size = _support_mask(dim, support_policy, support_indices)
    masked = probs * mask
    total = qml.math.sum(masked, axis=-1, keepdims=True)
    normalized = masked / total

    if shots is None:
        return normalized

    alpha = float(pseudocount)
    denominator = float(int(shots)) + alpha * support_size
    return (float(int(shots)) * normalized + alpha * mask) / denominator


def stabilized_probability_fn(
    probability_fn: ArrayFn,
    *,
    shots: Optional[int] = None,
    pseudocount: float = 0.5,
    support_policy: str = "full",
    support_indices: Optional[Sequence[int]] = None,
) -> ArrayFn:
    """Wrap a probability callable with fixed-support finite-shot stabilization."""

    def wrapped(params, *args, **kwargs):
        return stabilize_probabilities(
            probability_fn(params, *args, **kwargs),
            shots=shots,
            pseudocount=pseudocount,
            support_policy=support_policy,
            support_indices=support_indices,
        )

    return wrapped
