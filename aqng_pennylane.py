"""Accessible Quantum Natural Gradient (AQNG) for PennyLane.

Target: PennyLane 0.45.x, with a single trainable parameter array.

The optimizer uses the accessible Fisher metric

    G_acc(theta) = J(theta)^T Sigma(theta)^+ J(theta),

where J is the Jacobian of the measured/readout feature vector and Sigma is
its covariance matrix.  For mini-batches, per-example metrics are averaged.

The update plumbing is delegated to PennyLane's QNGOptimizer via its
``metric_tensor_fn`` hook.  Therefore QNG and AQNG can be compared with the
same PennyLane natural-gradient implementation; only the metric changes.

Typical usage
-------------

    aqng = AQNGOptimizer(stepsize=0.05, lam=1e-3, cov_lam=1e-4)

    # cost(theta) -> scalar
    # features(theta) -> shape (r,) or (batch, r)
    # covariance(theta) -> shape (r,r) or (batch,r,r)
    theta = aqng.step(
        cost,
        theta,
        feature_fn=features,
        covariance_fn=covariance,
    )

For a strict PennyLane-QNG comparison:

    qng = qml.QNGOptimizer(stepsize=0.05, approx="block-diag", lam=1e-3)

Notes
-----
* ``feature_fn`` must be differentiable with respect to the first argument.
  On hardware, define its QNodes with ``diff_method="parameter-shift"``.
* ``covariance_fn`` is NOT differentiated. It may therefore be estimated from
  finite-shot bitstrings.
* The theoretical pseudoinverse Sigma^+ is used when ``cov_lam=0``.  For
  finite-shot data, a small positive ``cov_lam`` is usually much more stable.
* Use one trainable parameter array if you want all parameter correlations,
  matching PennyLane's own recommendation for QNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as onp

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aqng_pennylane.py requires PennyLane. Install it with `pip install pennylane`."
    ) from exc


ArrayFn = Callable[..., object]


def _stack_if_sequence(x):
    """Convert a tuple/list of tensor outputs to one array-valued output."""
    if isinstance(x, (tuple, list)):
        return qml.math.stack(x, axis=-1)
    return x


def _to_numpy(x, dtype=float):
    """Detach a PennyLane/interface tensor for classical metric linear algebra."""
    return onp.asarray(qml.math.toarray(x), dtype=dtype)


def covariance_from_moments(means, second_moments):
    """Return Sigma = E[ff^T] - E[f]E[f]^T.

    Supports a single example with shapes ``(r,)`` and ``(r,r)`` or batched
    arrays with shapes ``(...,r)`` and ``(...,r,r)``.
    """
    means = _stack_if_sequence(means)
    return second_moments - means[..., :, None] * means[..., None, :]


def z_features_from_bitstrings(samples, z_terms: Sequence[Sequence[int]]):
    """Evaluate diagonal Z-string features on computational-basis samples.

    Parameters
    ----------
    samples:
        Bitstrings with shape ``(shots, n_wires)`` or
        ``(batch, shots, n_wires)``. Bits must be 0/1.
    z_terms:
        Iterable of wire-index tuples. ``(0,)`` represents Z_0,
        ``(0,2)`` represents Z_0 Z_2, etc.

    Returns
    -------
    ndarray
        Feature samples with shape ``(shots, r)`` or ``(batch, shots, r)``.
    """
    bits = onp.asarray(samples)
    if bits.ndim not in (2, 3):
        raise ValueError("samples must have shape (shots,n) or (batch,shots,n)")
    z = 1.0 - 2.0 * bits
    vals = []
    for term in z_terms:
        term = tuple(term)
        if len(term) == 0:
            vals.append(onp.ones(z.shape[:-1], dtype=float))
        else:
            vals.append(onp.prod(z[..., list(term)], axis=-1))
    return onp.stack(vals, axis=-1)


def z_covariance_from_bitstrings(
    samples,
    z_terms: Sequence[Sequence[int]],
    *,
    ddof: int = 0,
):
    """Estimate the covariance matrix of Z-string readout features from shots.

    The same computational-basis shots provide all diagonal Z-string
    covariances, so no r^2 collection of separate moment circuits is needed.
    """
    f = z_features_from_bitstrings(samples, z_terms)
    if f.shape[-2] <= ddof:
        raise ValueError("number of shots must be larger than ddof")
    centered = f - onp.mean(f, axis=-2, keepdims=True)
    denom = f.shape[-2] - ddof
    return onp.einsum("...si,...sj->...ij", centered, centered) / float(denom)


@dataclass
class MetricDiagnostics:
    batch_size: int
    feature_dim: int
    parameter_dim: int
    metric_rank: int
    metric_trace: float
    metric_condition: float


class AccessibleFisherMetric:
    """Callable PennyLane metric-tensor function for AQNG.

    ``feature_fn(params, *args, **kwargs)`` must return either ``(r,)`` or
    ``(batch,r)``. ``covariance_fn`` must return the corresponding ``(r,r)``
    or ``(batch,r,r)`` covariance array.

    The returned object has shape ``(p,p)``, where p is the total number of
    entries in the first (trainable) parameter array. PennyLane QNG flattens
    the gradient to the same dimension before applying the metric inverse.
    """

    def __init__(
        self,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        *,
        cov_lam: float = 0.0,
        rcond: float = 1e-10,
        project_cov_psd: bool = True,
        reduction: str = "mean",
    ):
        if cov_lam < 0:
            raise ValueError("cov_lam must be nonnegative")
        if rcond <= 0:
            raise ValueError("rcond must be positive")
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.feature_fn = feature_fn
        self.covariance_fn = covariance_fn
        self.cov_lam = float(cov_lam)
        self.rcond = float(rcond)
        self.project_cov_psd = bool(project_cov_psd)
        self.reduction = reduction
        self.last_diagnostics: Optional[MetricDiagnostics] = None
        self.last_metric = None

    def _feature_array(self, params, *args, **kwargs):
        return _stack_if_sequence(self.feature_fn(params, *args, **kwargs))

    def _inverse_covariance(self, sigma: onp.ndarray) -> onp.ndarray:
        sigma = 0.5 * (sigma + sigma.T)
        evals, evecs = onp.linalg.eigh(sigma)

        if self.project_cov_psd:
            evals = onp.maximum(evals, 0.0)

        scale = max(float(onp.max(onp.abs(evals))) if evals.size else 0.0, 1.0)
        tol = self.rcond * scale

        if self.cov_lam > 0.0:
            # Ridge-stabilized inverse. Particularly useful with finite shots.
            inv_evals = 1.0 / (evals + self.cov_lam)
        else:
            # Moore-Penrose pseudoinverse used by the ideal AQNG definition.
            inv_evals = onp.where(evals > tol, 1.0 / evals, 0.0)

        return (evecs * inv_evals) @ evecs.T

    def __call__(self, params, *args, **kwargs):
        # PennyLane 0.45 exposes an interface-dispatching Jacobian helper.
        jac_fn = qml.math.jacobian(
            lambda p: self._feature_array(p, *args, **kwargs), argnums=0
        )
        feature_values = self._feature_array(params, *args, **kwargs)
        jac = jac_fn(params)
        sigma = self.covariance_fn(params, *args, **kwargs)

        f_shape = tuple(qml.math.shape(feature_values))
        p_shape = tuple(qml.math.shape(params))
        if len(f_shape) < 1:
            raise ValueError("feature_fn must return at least one feature")

        r = int(f_shape[-1])
        p = int(onp.prod(p_shape)) if p_shape else 1
        batch = int(onp.prod(f_shape[:-1])) if len(f_shape) > 1 else 1

        jac_np = _to_numpy(jac).reshape(batch, r, p)
        sigma_np = _to_numpy(sigma)
        if sigma_np.ndim == 2:
            sigma_np = sigma_np.reshape(1, r, r)
        else:
            sigma_np = sigma_np.reshape(batch, r, r)

        if sigma_np.shape[0] != batch:
            raise ValueError(
                "Batch mismatch: feature_fn returned batch size "
                f"{batch}, covariance_fn returned {sigma_np.shape[0]}"
            )
        if sigma_np.shape[1:] != (r, r):
            raise ValueError(
                f"covariance_fn must end in shape ({r},{r}), got {sigma_np.shape}"
            )

        metric = onp.zeros((p, p), dtype=float)
        for b in range(batch):
            inv_sigma = self._inverse_covariance(sigma_np[b])
            jb = jac_np[b]
            metric += jb.T @ inv_sigma @ jb

        if self.reduction == "mean":
            metric /= float(batch)

        metric = 0.5 * (metric + metric.T)
        self.last_metric = metric.copy()

        eigvals = onp.linalg.eigvalsh(metric)
        maxeig = max(float(onp.max(onp.abs(eigvals))) if eigvals.size else 0.0, 1.0)
        rank = int(onp.sum(eigvals > self.rcond * maxeig))
        pos = eigvals[eigvals > self.rcond * maxeig]
        condition = float(onp.max(pos) / onp.min(pos)) if pos.size > 1 else 1.0
        self.last_diagnostics = MetricDiagnostics(
            batch_size=batch,
            feature_dim=r,
            parameter_dim=p,
            metric_rank=rank,
            metric_trace=float(onp.trace(metric)),
            metric_condition=condition,
        )

        # PennyLane QNG accepts a user-supplied (p,p) metric and reshapes it
        # internally to match its flattened gradient representation.
        return pnp.array(metric, requires_grad=False)


class AQNGOptimizer:
    """PennyLane-compatible Accessible Quantum Natural Gradient optimizer.

    This is deliberately a thin wrapper around ``qml.QNGOptimizer``.  The
    update code is therefore PennyLane's QNG update; AQNG differs only by
    replacing the Fubini--Study metric with the accessible Fisher metric.

    Parameters
    ----------
    stepsize:
        Natural-gradient step size.
    lam:
        PennyLane metric damping added to ``G_acc`` before pseudoinversion.
    cov_lam:
        Ridge added inside each readout covariance inversion. Use 0 for the
        ideal Moore-Penrose definition; use a small positive value with shots.
    rcond:
        Relative eigenvalue cutoff for covariance pseudoinverses/diagnostics.
    """

    def __init__(
        self,
        stepsize: float = 0.01,
        *,
        lam: float = 1e-3,
        cov_lam: float = 0.0,
        rcond: float = 1e-10,
        project_cov_psd: bool = True,
        reduction: str = "mean",
    ):
        self.stepsize = float(stepsize)
        self.lam = float(lam)
        self.cov_lam = float(cov_lam)
        self.rcond = float(rcond)
        self.project_cov_psd = bool(project_cov_psd)
        self.reduction = reduction
        # approx is irrelevant when metric_tensor_fn is supplied, but None
        # makes the intended full supplied metric explicit.
        self._optimizer = qml.QNGOptimizer(stepsize=self.stepsize, approx=None, lam=self.lam)
        self.metric_fn: Optional[AccessibleFisherMetric] = None

    @property
    def metric_tensor(self):
        return self._optimizer.metric_tensor

    @property
    def diagnostics(self) -> Optional[MetricDiagnostics]:
        return None if self.metric_fn is None else self.metric_fn.last_diagnostics

    def _make_metric(self, feature_fn: ArrayFn, covariance_fn: ArrayFn):
        self.metric_fn = AccessibleFisherMetric(
            feature_fn,
            covariance_fn,
            cov_lam=self.cov_lam,
            rcond=self.rcond,
            project_cov_psd=self.project_cov_psd,
            reduction=self.reduction,
        )
        return self.metric_fn

    def step(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        grad_fn: Optional[ArrayFn] = None,
        recompute_tensor: bool = True,
        **kwargs,
    ):
        """Take one AQNG step.

        ``feature_fn`` and ``covariance_fn`` must accept the same positional and
        keyword arguments as ``objective_fn`` (unused arguments may be ignored).
        For data training, a convenient pattern is to close over the current
        mini-batch and call ``step`` with only ``params`` as a positional arg.
        """
        metric_fn = self._make_metric(feature_fn, covariance_fn)
        return self._optimizer.step(
            objective_fn,
            params,
            *args,
            grad_fn=grad_fn,
            recompute_tensor=recompute_tensor,
            metric_tensor_fn=metric_fn,
            **kwargs,
        )

    def step_and_cost(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        grad_fn: Optional[ArrayFn] = None,
        recompute_tensor: bool = True,
        **kwargs,
    ):
        """Take one AQNG step and return ``(new_params, old_cost)``."""
        metric_fn = self._make_metric(feature_fn, covariance_fn)
        return self._optimizer.step_and_cost(
            objective_fn,
            params,
            *args,
            grad_fn=grad_fn,
            recompute_tensor=recompute_tensor,
            metric_tensor_fn=metric_fn,
            **kwargs,
        )


def make_aqng_metric_tensor_fn(
    feature_fn: ArrayFn,
    covariance_fn: ArrayFn,
    *,
    cov_lam: float = 0.0,
    rcond: float = 1e-10,
    project_cov_psd: bool = True,
    reduction: str = "mean",
):
    """Return a metric function usable directly with ``qml.QNGOptimizer``.

    Example
    -------
    ````python
    aqng_metric = make_aqng_metric_tensor_fn(features, covariance, cov_lam=1e-4)
    opt = qml.QNGOptimizer(stepsize=0.05, approx=None, lam=1e-3)
    theta = opt.step(cost, theta, metric_tensor_fn=aqng_metric)
    ````
    """
    return AccessibleFisherMetric(
        feature_fn,
        covariance_fn,
        cov_lam=cov_lam,
        rcond=rcond,
        project_cov_psd=project_cov_psd,
        reduction=reduction,
    )
