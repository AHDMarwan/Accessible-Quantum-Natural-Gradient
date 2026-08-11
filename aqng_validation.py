"""Validation-gate utilities for stable finite-shot AQNG readout calibration.

The production AQNG-v2 benchmark infers score-space support from nonzero empirical
probabilities.  That is fine analytically, but finite-shot zero counts can change
support dimension and physical Walsh rank across shot budgets.  This module fixes
that nuisance for the pre-large-scale validation gate:

* generic families use the full computational-basis support;
* ``u1_rzxy`` uses the known half-filled Hamming-weight sector;
* a Dirichlet-style pseudocount makes every allowed support probability positive;
* tangent-score rows are built only on the fixed support.

The readout itself is still fitted by ``aqng_readouts.fit_rank_matched_readouts``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from aqng_readouts import fit_rank_matched_readouts


def fixed_support_indices(n_qubits: int, family: str) -> np.ndarray:
    """Return a deterministic score-space support for a benchmark family."""
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dim = 2**n_qubits
    if family == "u1_rzxy":
        if n_qubits % 2:
            raise ValueError("u1_rzxy validation support requires even n_qubits")
        target = n_qubits // 2
        return np.asarray(
            [idx for idx in range(dim) if int(idx).bit_count() == target], dtype=int
        )
    return np.arange(dim, dtype=int)


def stabilize_reference_probabilities(
    probabilities: np.ndarray,
    support_indices: np.ndarray,
    *,
    shots: Optional[int] = None,
    pseudocount: float = 0.5,
    analytic_floor: float = 1e-15,
) -> np.ndarray:
    """Project onto fixed support and make every allowed outcome strictly positive.

    For finite shots, ``pseudocount`` is interpreted as a symmetric Dirichlet
    pseudocount in count units: ``p_i -> (shots*p_i + alpha)/(shots + alpha*K)``.
    For analytic probabilities only a tiny numerical floor is used.
    """
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    support = np.asarray(support_indices, dtype=int).reshape(-1)
    if p.size < 2 or support.size < 2:
        raise ValueError("probability/support dimensions are too small")
    if np.any(support < 0) or np.any(support >= p.size):
        raise ValueError("support index out of range")
    if len(np.unique(support)) != len(support):
        raise ValueError("support indices must be unique")
    if pseudocount < 0:
        raise ValueError("pseudocount must be nonnegative")
    if np.any(~np.isfinite(p)) or np.min(p) < -1e-10:
        raise ValueError("probabilities must be finite and nonnegative")

    ps = np.maximum(p[support], 0.0)
    total = float(np.sum(ps))
    if total <= 0.0:
        raise ValueError("fixed support has zero total probability")
    ps = ps / total

    if shots is not None:
        shots = int(shots)
        if shots < 1:
            raise ValueError("shots must be positive")
        alpha = float(pseudocount)
        ps = (shots * ps + alpha) / (shots + alpha * len(ps))
    else:
        floor = float(analytic_floor)
        if floor <= 0:
            raise ValueError("analytic_floor must be positive")
        ps = np.maximum(ps, floor)
        ps /= np.sum(ps)

    out = np.zeros_like(p, dtype=float)
    out[support] = ps
    return out


def normalized_reference_score_rows_fixed_support(
    probabilities: np.ndarray,
    probability_jacobians: np.ndarray,
    directions: np.ndarray,
    *,
    reference_probabilities: np.ndarray,
    support_indices: np.ndarray,
    tangent_floor: float = 1e-13,
) -> np.ndarray:
    """Build normalized tangent-score rows in one fixed reference support."""
    probs = np.asarray(probabilities, dtype=float)
    jacs = np.asarray(probability_jacobians, dtype=float)
    dirs = np.asarray(directions, dtype=float)
    p_ref = np.asarray(reference_probabilities, dtype=float).reshape(-1)
    support = np.asarray(support_indices, dtype=int).reshape(-1)

    if probs.ndim == 1:
        probs = probs[None, :]
    if jacs.ndim == 2:
        jacs = jacs[None, :, :]
    if dirs.ndim == 1:
        dirs = dirs[None, :]
    if probs.ndim != 2 or jacs.ndim != 3 or dirs.ndim != 2:
        raise ValueError("invalid probability/Jacobian/direction dimensions")
    if jacs.shape[:2] != probs.shape:
        raise ValueError("probability_jacobians must have shape (B,D,p)")
    if dirs.shape[1] != jacs.shape[2]:
        raise ValueError("direction dimension mismatch")
    if p_ref.size != probs.shape[1]:
        raise ValueError("reference probability dimension mismatch")
    if np.any(p_ref[support] <= 0.0):
        raise ValueError("reference probabilities must be positive on fixed support")

    sqrt_p = np.sqrt(p_ref[support])
    w = sqrt_p / np.linalg.norm(sqrt_p)
    rows: list[np.ndarray] = []
    for b in range(probs.shape[0]):
        dp = jacs[b] @ dirs.T
        for m in range(dp.shape[1]):
            u = dp[support, m] / sqrt_p
            u = u - w * float(w @ u)
            norm = float(np.linalg.norm(u))
            if np.isfinite(norm) and norm > float(tangent_floor):
                rows.append(u / norm)
    if not rows:
        raise ValueError("no regular calibration score directions")
    return np.vstack(rows)


def fit_stable_rank_matched_readouts(
    reference_probabilities: np.ndarray,
    score_rows: np.ndarray,
    *,
    n_qubits: int,
    readout_order: int,
    seed: int,
) -> dict:
    """Fit physical/random/aligned readouts on an already stabilized support."""
    return dict(
        fit_rank_matched_readouts(
            reference_probabilities,
            score_rows,
            n_qubits=int(n_qubits),
            readout_order=int(readout_order),
            seed=int(seed),
            probability_floor=1e-18,
        )
    )
