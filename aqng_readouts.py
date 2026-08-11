"""Readout-design and control utilities for Accessible QNG experiments.

This module carries the score-space construction used by the spectral-accessibility
paper into the optimizer repository.  It deliberately keeps the readout design
separate from the optimizer core:

* ``physical`` is the covariance-whitened low-weight diagonal Walsh/Pauli-Z span;
* ``aligned_crossfit`` is a same-rank leading score subspace fitted from an
  independent tangent/calibration sample;
* ``random_rank`` is a Haar-random same-rank centered score subspace.

All three are represented as fixed outcome functions.  Given probabilities ``p``
and their parameter Jacobian ``dp/dtheta``, the accessible metric is

    G_A = mean_b J_b.T @ Sigma_b^+ @ J_b,

where the columns of ``J_b`` are derivatives of the retained outcome-feature
expectations.  The helper therefore supports both analytic probabilities and
finite-shot probability estimates produced by PennyLane.

The centered-basis construction is adapted from the reproducibility code for
"Measurement-Accessible Quantum Tangent Geometry: Rank Baselines and Spectral
Orientation".  In the supervised setting a single reference distribution is
formed from the calibration inputs so that one fixed readout can be used across
all examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ReadoutDesign:
    """Fixed rank-r readout represented by outcome functions.

    ``basis`` contains orthonormal columns in the reference square-root score
    coordinates.  ``outcome_features`` contains the corresponding classical
    functions on the full computational-basis outcome space.  At the reference
    distribution these functions are centered and have identity covariance.
    """

    name: str
    rank: int
    centered_dimension: int
    support_indices: np.ndarray
    reference_probabilities: np.ndarray
    basis: np.ndarray
    outcome_features: np.ndarray


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def _validate_probability_vector(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("probability vector must contain at least two outcomes")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    if np.min(p) < -1e-12:
        raise ValueError("probabilities must be nonnegative")
    p = np.maximum(p, 0.0)
    total = float(np.sum(p))
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    return p / total


def centered_orthonormalize(
    q: np.ndarray,
    sqrt_p: np.ndarray,
    rank: int,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Project columns away from the constant score mode and orthonormalize."""

    q = np.asarray(q, dtype=float)
    sqrt_p = np.asarray(sqrt_p, dtype=float).reshape(-1)
    rank = int(rank)
    if q.ndim != 2 or q.shape[0] != sqrt_p.size:
        raise ValueError("q must have shape (support_size, n_columns)")
    if rank < 1:
        raise ValueError("rank must be positive")
    if rank > sqrt_p.size - 1:
        raise ValueError("rank exceeds centered score-space dimension")
    if q.shape[1] < rank:
        raise ValueError("not enough candidate columns for requested rank")

    wnorm = float(np.linalg.norm(sqrt_p))
    if wnorm <= 0.0:
        raise ValueError("sqrt_p must have nonzero norm")
    w = sqrt_p / wnorm
    centered = q - w[:, None] * (w @ q)[None, :]

    # SVD is slightly more robust than QR when a candidate column is nearly the
    # removed constant mode.
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(s[0]) if s.size else 0.0, 1.0)
    available = int(np.sum(s > tolerance * scale))
    if available < rank:
        raise ValueError(
            f"only {available} independent centered directions for requested rank {rank}"
        )
    out = u[:, :rank]

    if np.linalg.norm(out.T @ out - np.eye(rank)) > 1e-8:
        raise RuntimeError("readout basis lost orthonormality")
    if np.linalg.norm(w @ out) > 1e-8:
        raise RuntimeError("readout basis is not centered")
    return out


def walsh_readout_basis(
    p_support: np.ndarray,
    support_indices: np.ndarray,
    n_qubits: int,
    max_weight: int,
    *,
    svd_tolerance: float = 1e-10,
) -> tuple[np.ndarray, int, list[tuple[int, ...]]]:
    """Covariance-whiten the diagonal Pauli-Z/Walsh span through ``max_weight``.

    The returned columns live in square-root probability score coordinates, as
    in the original spectral-geometry implementation.
    """

    p_support = np.asarray(p_support, dtype=float).reshape(-1)
    support_indices = np.asarray(support_indices, dtype=int).reshape(-1)
    if p_support.size != support_indices.size:
        raise ValueError("support probabilities and indices must have equal length")
    if max_weight < 1:
        raise ValueError("max_weight must be >= 1")

    bits = (support_indices[:, None] >> (n_qubits - 1 - np.arange(n_qubits))) & 1
    z = 1 - 2 * bits
    cols: list[np.ndarray] = []
    labels: list[tuple[int, ...]] = []
    for order in range(1, max_weight + 1):
        for subset in combinations(range(n_qubits), order):
            cols.append(np.prod(z[:, subset], axis=1).astype(float))
            labels.append(subset)

    if not cols:
        return np.zeros((len(p_support), 0), dtype=float), 0, labels

    f = np.column_stack(cols)
    means = p_support @ f
    weighted = np.sqrt(p_support)[:, None] * (f - means[None, :])
    u, s, _ = np.linalg.svd(weighted, full_matrices=False)
    scale = max(float(s[0]) if s.size else 0.0, 1.0)
    rank = int(np.sum(s > svd_tolerance * scale))
    return u[:, :rank], rank, labels


def basis_to_outcome_features(
    basis: np.ndarray,
    reference_probabilities: np.ndarray,
    support_indices: np.ndarray,
) -> np.ndarray:
    """Convert score-coordinate basis columns to fixed classical outcome functions."""

    p = _validate_probability_vector(reference_probabilities)
    support_indices = np.asarray(support_indices, dtype=int).reshape(-1)
    basis = np.asarray(basis, dtype=float)
    if basis.shape[0] != support_indices.size:
        raise ValueError("basis/support shape mismatch")
    ps = p[support_indices]
    if np.any(ps <= 0.0):
        raise ValueError("reference support must have strictly positive probabilities")

    features = np.zeros((p.size, basis.shape[1]), dtype=float)
    features[support_indices] = basis / np.sqrt(ps)[:, None]
    return features


def normalized_reference_score_rows(
    probabilities: np.ndarray,
    probability_jacobians: np.ndarray,
    directions: np.ndarray,
    *,
    reference_probabilities: Optional[np.ndarray] = None,
    probability_floor: float = 1e-13,
    tangent_floor: float = 1e-13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized tangent-score rows in one common reference score space.

    Parameters
    ----------
    probabilities
        Shape ``(B, D)`` or ``(D,)``.
    probability_jacobians
        Shape ``(B, D, p)`` or ``(D, p)``.
    directions
        Shape ``(M, p)`` or ``(p,)``.  These are independent calibration
        directions; labels are never used.
    reference_probabilities
        Optional fixed distribution defining the common square-root score
        coordinates.  By default the mean calibration distribution is used.

    Returns
    -------
    rows, p_ref, support_indices
        ``rows`` has shape ``(n_regular, support_size)`` and unit Euclidean norm.
        It is centered against the constant mode associated with ``p_ref``.

    Notes
    -----
    A supervised dataset has one probability distribution per input, whereas the
    original paper fixed a single distribution.  A reference-mixture score space
    is used here so that the learned readout is one fixed outcome function rather
    than a different projector for every input.
    """

    probs = np.asarray(probabilities, dtype=float)
    jacs = np.asarray(probability_jacobians, dtype=float)
    dirs = np.asarray(directions, dtype=float)

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
        raise ValueError("direction dimension does not match parameter dimension")

    probs = np.vstack([_validate_probability_vector(row) for row in probs])
    p_ref = (
        _validate_probability_vector(reference_probabilities)
        if reference_probabilities is not None
        else _validate_probability_vector(np.mean(probs, axis=0))
    )
    support = np.flatnonzero(p_ref > float(probability_floor))
    if support.size < 2:
        raise ValueError("reference support has fewer than two outcomes")

    sqrt_p = np.sqrt(p_ref[support])
    w = sqrt_p / np.linalg.norm(sqrt_p)
    rows: list[np.ndarray] = []
    for b in range(probs.shape[0]):
        dp = jacs[b] @ dirs.T  # (D, M)
        for m in range(dp.shape[1]):
            u = dp[support, m] / sqrt_p
            # Sum(dp)=0 analytically.  Re-center to remove finite-precision and
            # finite-shot leakage into the constant mode.
            u = u - w * float(w @ u)
            norm = float(np.linalg.norm(u))
            if np.isfinite(norm) and norm > tangent_floor:
                rows.append(u / norm)

    if not rows:
        raise ValueError("no regular calibration score directions")
    return np.vstack(rows), p_ref, support


def fit_rank_matched_readouts(
    reference_probabilities: np.ndarray,
    score_rows: np.ndarray,
    *,
    n_qubits: int,
    readout_order: int = 2,
    seed: int = 0,
    probability_floor: float = 1e-13,
    svd_tolerance: float = 1e-10,
) -> Mapping[str, ReadoutDesign]:
    """Fit physical, cross-fitted aligned, and random same-rank readouts."""

    p_ref = _validate_probability_vector(reference_probabilities)
    support = np.flatnonzero(p_ref > probability_floor)
    ps = p_ref[support]
    if support.size < 2:
        raise ValueError("reference support has fewer than two outcomes")

    q_phys, rank, _ = walsh_readout_basis(
        ps,
        support,
        n_qubits,
        readout_order,
        svd_tolerance=svd_tolerance,
    )
    if rank < 1:
        raise ValueError("physical readout has zero covariance rank")
    q_phys = centered_orthonormalize(
        q_phys, np.sqrt(ps), rank, tolerance=svd_tolerance
    )

    rows = np.asarray(score_rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError("score_rows must be a matrix")
    if rows.shape[1] == p_ref.size:
        rows = rows[:, support]
    elif rows.shape[1] != support.size:
        raise ValueError("score_rows dimension does not match full or support space")
    if rows.shape[0] < rank:
        raise ValueError(
            f"alignment sample has {rows.shape[0]} rows but physical rank is {rank}"
        )

    # Remove any residual constant component before spectral fitting.
    w = np.sqrt(ps)
    w = w / np.linalg.norm(w)
    rows = rows - (rows @ w)[:, None] * w[None, :]
    _, _, vh = np.linalg.svd(rows, full_matrices=False)
    if vh.shape[0] < rank:
        raise ValueError("alignment SVD does not supply the requested rank")
    q_align = centered_orthonormalize(
        vh[:rank].T, np.sqrt(ps), rank, tolerance=svd_tolerance
    )

    rng = np.random.default_rng(int(seed))
    q_rand = centered_orthonormalize(
        rng.normal(size=(support.size, rank)),
        np.sqrt(ps),
        rank,
        tolerance=svd_tolerance,
    )

    centered_dim = int(support.size - 1)
    designs: dict[str, ReadoutDesign] = {}
    for name, q in (
        ("physical", q_phys),
        ("aligned_crossfit", q_align),
        ("random_rank", q_rand),
    ):
        designs[name] = ReadoutDesign(
            name=name,
            rank=rank,
            centered_dimension=centered_dim,
            support_indices=support.copy(),
            reference_probabilities=p_ref.copy(),
            basis=q.copy(),
            outcome_features=basis_to_outcome_features(q, p_ref, support),
        )
    return designs


def readout_retention(score_rows: np.ndarray, design: ReadoutDesign) -> float:
    """Mean squared projection mass of normalized score rows onto a design."""

    rows = np.asarray(score_rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError("score_rows must be a matrix")
    if rows.shape[1] == design.reference_probabilities.size:
        rows = rows[:, design.support_indices]
    if rows.shape[1] != design.basis.shape[0]:
        raise ValueError("score/readout dimension mismatch")
    coeff = rows @ design.basis
    return float(np.mean(np.sum(coeff * coeff, axis=1)))


def _psd_inverse_batch(
    covariances: np.ndarray,
    *,
    rcond: float,
    cov_lam: float,
) -> np.ndarray:
    cov = _sym(np.asarray(covariances, dtype=float))
    if cov.ndim == 2:
        cov = cov[None, :, :]
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    if cov_lam < 0.0:
        raise ValueError("cov_lam must be nonnegative")
    if rcond <= 0.0:
        raise ValueError("rcond must be positive")

    if cov_lam > 0.0:
        inv = 1.0 / (vals + float(cov_lam))
    else:
        scale = np.maximum(np.max(vals, axis=1), 1.0)
        tol = float(rcond) * scale[:, None]
        inv = np.where(vals > tol, 1.0 / np.maximum(vals, tol), 0.0)
    return np.einsum("bik,bk,bjk->bij", vecs, inv, vecs, optimize=True)


def outcome_feature_moments(
    probabilities: np.ndarray,
    outcome_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature means and covariances for fixed outcome functions."""

    probs = np.asarray(probabilities, dtype=float)
    features = np.asarray(outcome_features, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    if probs.ndim != 2 or features.ndim != 2:
        raise ValueError("probabilities/features must be matrices")
    if probs.shape[1] != features.shape[0]:
        raise ValueError("outcome dimension mismatch")
    probs = np.vstack([_validate_probability_vector(row) for row in probs])

    means = probs @ features
    second = np.einsum(
        "bd,dr,ds->brs", probs, features, features, optimize=True
    )
    cov = _sym(second - np.einsum("br,bs->brs", means, means, optimize=True))
    return means, cov


def accessible_metric_from_probability_jacobians(
    probabilities: np.ndarray,
    probability_jacobians: np.ndarray,
    outcome_features: np.ndarray,
    *,
    rcond: float = 1e-10,
    cov_lam: float = 0.0,
    reduction: str = "mean",
) -> tuple[np.ndarray, dict]:
    """Construct the accessible Fisher pullback from full outcome probabilities.

    ``probability_jacobians`` has shape ``(B,D,p)`` and ``outcome_features`` has
    shape ``(D,r)``.  This form makes physical, random, and aligned readouts use
    exactly the same probability/Jacobian record; only the retained subspace is
    changed.
    """

    probs = np.asarray(probabilities, dtype=float)
    jacs = np.asarray(probability_jacobians, dtype=float)
    features = np.asarray(outcome_features, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    if jacs.ndim == 2:
        jacs = jacs[None, :, :]
    if probs.ndim != 2 or jacs.ndim != 3:
        raise ValueError("probabilities/Jacobians must have shape (B,D)/(B,D,p)")
    if jacs.shape[:2] != probs.shape:
        raise ValueError("probability/Jacobian batch shape mismatch")
    if features.ndim != 2 or features.shape[0] != probs.shape[1]:
        raise ValueError("outcome feature shape mismatch")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")

    _, cov = outcome_feature_moments(probs, features)
    cinv = _psd_inverse_batch(cov, rcond=rcond, cov_lam=cov_lam)
    # d E[f_r] / d theta_j = sum_d f[d,r] * d p[d] / d theta_j.
    jfeat = np.einsum("dr,bdp->brp", features, jacs, optimize=True)
    metric = np.einsum(
        "bri,brs,bsj->ij", jfeat, cinv, jfeat, optimize=True
    )
    if reduction == "mean":
        metric = metric / float(probs.shape[0])
    metric = _sym(metric)

    eig = np.linalg.eigvalsh(metric)
    eig = np.maximum(eig, 0.0)
    scale = max(float(np.max(eig)) if eig.size else 0.0, 1.0)
    rank = int(np.sum(eig > rcond * scale))
    diagnostics = {
        "batch_size": int(probs.shape[0]),
        "feature_dim": int(features.shape[1]),
        "parameter_dim": int(jacs.shape[2]),
        "metric_rank": rank,
        "metric_trace": float(np.trace(metric)),
        "metric_max_eig": float(np.max(eig)) if eig.size else 0.0,
    }
    return metric, diagnostics


def normalize_metric(
    metric: np.ndarray,
    mode: str = "none",
    *,
    target: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Remove an overall metric-scale confound while preserving orientation.

    ``mode='trace'`` sets ``Tr(G)`` to ``target``; when omitted, the target is the
    parameter dimension.  ``mode='maxeig'`` sets the largest eigenvalue to
    ``target`` (default 1).  ``mode='none'`` returns the metric unchanged.
    """

    g = _sym(np.asarray(metric, dtype=float))
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("metric must be square")
    mode = str(mode).lower()
    if mode == "none":
        return g, 1.0
    if mode == "trace":
        denom = float(np.trace(g))
        tgt = float(g.shape[0] if target is None else target)
    elif mode == "maxeig":
        denom = float(np.max(np.linalg.eigvalsh(g)))
        tgt = float(1.0 if target is None else target)
    else:
        raise ValueError("metric normalization must be 'none', 'trace', or 'maxeig'")
    if tgt <= 0.0:
        raise ValueError("normalization target must be positive")
    if not np.isfinite(denom) or denom <= 0.0:
        return g, 1.0
    factor = tgt / denom
    return g * factor, float(factor)


def effective_damping(metric: np.ndarray, lam: float, mode: str = "absolute") -> float:
    """Convert a nominal damping coefficient to an effective metric-scale value."""

    if lam < 0.0:
        raise ValueError("lam must be nonnegative")
    g = _sym(np.asarray(metric, dtype=float))
    mode = str(mode).lower()
    if mode == "absolute":
        scale = 1.0
    elif mode == "mean_eig":
        scale = float(np.trace(g)) / float(g.shape[0])
    elif mode == "maxeig":
        scale = max(float(np.max(np.linalg.eigvalsh(g))), 0.0)
    else:
        raise ValueError("damping mode must be 'absolute', 'mean_eig', or 'maxeig'")
    return float(lam) * float(scale)


def solve_controlled_direction(
    metric: np.ndarray,
    gradient: np.ndarray,
    *,
    lam: float,
    stepsize: float,
    rcond: float = 1e-10,
    metric_normalization: str = "none",
    normalization_target: Optional[float] = None,
    damping_mode: str = "absolute",
    max_direction_norm: Optional[float] = None,
    max_metric_step: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """Solve a damped natural-gradient system with scale and trust controls."""

    if stepsize <= 0.0:
        raise ValueError("stepsize must be positive")
    if max_direction_norm is not None and max_direction_norm <= 0.0:
        raise ValueError("max_direction_norm must be positive or None")
    if max_metric_step is not None and max_metric_step <= 0.0:
        raise ValueError("max_metric_step must be positive or None")

    gnormed, metric_scale = normalize_metric(
        metric, metric_normalization, target=normalization_target
    )
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    if gnormed.shape[0] != grad.size:
        raise ValueError("gradient dimension does not match metric")
    lam_eff = effective_damping(gnormed, lam, damping_mode)
    system = gnormed + lam_eff * np.eye(grad.size)
    try:
        direction = np.linalg.solve(system, grad)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(system, rcond=rcond) @ grad

    raw_direction_norm = float(np.linalg.norm(direction))
    quad = float(direction @ gnormed @ direction)
    raw_metric_step_norm = float(stepsize * np.sqrt(max(quad, 0.0)))
    clip_scale = 1.0
    if (
        max_direction_norm is not None
        and raw_direction_norm > max_direction_norm
        and raw_direction_norm > 0.0
    ):
        clip_scale = min(clip_scale, max_direction_norm / raw_direction_norm)
    if (
        max_metric_step is not None
        and raw_metric_step_norm > max_metric_step
        and raw_metric_step_norm > 0.0
    ):
        clip_scale = min(clip_scale, max_metric_step / raw_metric_step_norm)

    accepted = direction * clip_scale
    accepted_quad = float(accepted @ gnormed @ accepted)
    diagnostics = {
        "metric_scale_factor": float(metric_scale),
        "effective_damping": float(lam_eff),
        "raw_direction_norm": raw_direction_norm,
        "direction_norm": float(np.linalg.norm(accepted)),
        "raw_metric_step_norm": raw_metric_step_norm,
        "metric_step_norm": float(stepsize * np.sqrt(max(accepted_quad, 0.0))),
        "trust_region_clipped": bool(clip_scale < 1.0),
        "clip_scale": float(clip_scale),
    }
    return accepted, diagnostics


def samples_to_probabilities(samples: np.ndarray, n_qubits: Optional[int] = None) -> np.ndarray:
    """Convert computational-basis bitstrings to empirical probability vectors."""

    bits = np.asarray(samples)
    if bits.ndim not in (2, 3):
        raise ValueError("samples must have shape (shots,n) or (batch,shots,n)")
    nq = int(bits.shape[-1] if n_qubits is None else n_qubits)
    if bits.shape[-1] != nq:
        raise ValueError("sample width does not match n_qubits")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("samples must contain only 0/1 bit values")

    single = bits.ndim == 2
    if single:
        bits = bits[None, :, :]
    powers = (2 ** np.arange(nq - 1, -1, -1)).astype(np.int64)
    out = np.zeros((bits.shape[0], 2**nq), dtype=float)
    for b in range(bits.shape[0]):
        idx = np.asarray(bits[b], dtype=np.int64) @ powers
        out[b] = np.bincount(idx, minlength=2**nq) / float(bits.shape[1])
    return out[0] if single else out


__all__ = [
    "ReadoutDesign",
    "accessible_metric_from_probability_jacobians",
    "basis_to_outcome_features",
    "centered_orthonormalize",
    "effective_damping",
    "fit_rank_matched_readouts",
    "normalize_metric",
    "normalized_reference_score_rows",
    "outcome_feature_moments",
    "readout_retention",
    "samples_to_probabilities",
    "solve_controlled_direction",
    "walsh_readout_basis",
]
