"""Accessible Quantum Natural Gradient (AQNG) for PennyLane.

AQNG preconditions the objective gradient with the Fisher information that is
accessible through a chosen readout feature space,

    G_acc(theta) = reduction_b J_b(theta)^T Sigma_b(theta)^+ J_b(theta),

where J_b is the Jacobian of the readout features and Sigma_b is their
covariance matrix. The optimizer targets PennyLane's Autograd/NumPy workflow
(the same workflow as ``qml.QNGOptimizer``) and assumes one trainable parameter
array. Extra positional arguments may be supplied, but only argument 0 is
differentiated.

Unlike the first prototype, ``AQNGOptimizer`` does not delegate the update to
``qml.QNGOptimizer`` and does not need to materialize the p x p accessible
metric. It factors

    G_acc = A.T @ A

and solves ``(G_acc + lam I) d = grad`` using an automatic primal/dual choice:

* primal solve, dimension p, when p <= B*r;
* Woodbury dual solve, dimension B*r, when B*r < p;
* SVD pseudoinverse when lam == 0.

For a mini-batch of B independent covariance blocks, the exact stacked dual
dimension is B*r, not r.

Target: PennyLane 0.45.x.
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
    if isinstance(x, (tuple, list)):
        return qml.math.stack(x, axis=-1)
    return x


def _to_numpy(x, dtype=float):
    return onp.asarray(qml.math.toarray(x), dtype=dtype)


def covariance_from_moments(means, second_moments):
    """Return ``Sigma = E[ff^T] - E[f]E[f]^T``."""
    means = _stack_if_sequence(means)
    return second_moments - means[..., :, None] * means[..., None, :]


def z_features_from_bitstrings(samples, z_terms: Sequence[Sequence[int]]):
    """Evaluate diagonal Z-string features on computational-basis samples."""
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
    """Estimate all Z-string covariances from the same bitstring shots."""
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
    factor_rows: int
    metric_rank: int
    metric_trace: float
    metric_condition: float
    solver: str
    solve_dimension: int
    gradient_norm: float
    natural_gradient_norm: float


class AccessibleFisherFactor:
    """Build ``A`` such that ``G_acc = A.T @ A``.

    ``feature_fn(params, *args, **kwargs)`` must return ``(r,)`` or
    ``(batch,r)``. ``covariance_fn`` must return ``(r,r)`` or
    ``(batch,r,r)``. Only the first argument is differentiated.
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
        self.last_factor: Optional[onp.ndarray] = None
        self.last_batch_size: Optional[int] = None
        self.last_feature_dim: Optional[int] = None
        self.last_parameter_shape = None

    def _feature_array(self, params, *args, **kwargs):
        return _stack_if_sequence(self.feature_fn(params, *args, **kwargs))

    def _whitener_weights(self, sigma: onp.ndarray):
        sigma = 0.5 * (sigma + sigma.T)
        evals, evecs = onp.linalg.eigh(sigma)
        if self.project_cov_psd:
            evals = onp.maximum(evals, 0.0)

        scale = max(float(onp.max(onp.abs(evals))) if evals.size else 0.0, 1.0)
        tol = self.rcond * scale

        if self.cov_lam > 0.0:
            weights = 1.0 / onp.sqrt(evals + self.cov_lam)
        else:
            weights = onp.zeros_like(evals)
            keep = evals > tol
            weights[keep] = 1.0 / onp.sqrt(evals[keep])

        return weights, evecs

    def __call__(self, params, *args, **kwargs) -> onp.ndarray:
        jac_fn = qml.jacobian(
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

        if sigma_np.shape != (batch, r, r):
            raise ValueError(
                "covariance_fn shape mismatch: expected "
                f"{(batch, r, r)}, got {sigma_np.shape}"
            )

        scale = 1.0 / onp.sqrt(float(batch)) if self.reduction == "mean" else 1.0
        blocks = []
        for b in range(batch):
            weights, evecs = self._whitener_weights(sigma_np[b])
            whitened_jac = weights[:, None] * (evecs.T @ jac_np[b])
            blocks.append(scale * whitened_jac)

        factor = onp.concatenate(blocks, axis=0)
        self.last_factor = factor
        self.last_batch_size = batch
        self.last_feature_dim = r
        self.last_parameter_shape = p_shape
        return factor


class AccessibleFisherMetric:
    """Compatibility callable returning the explicit p x p AQNG metric.

    Use this only when intentionally passing AQNG as ``metric_tensor_fn`` to
    ``qml.QNGOptimizer``. It materializes ``A.T @ A`` and therefore does not
    provide the solve advantage of ``AQNGOptimizer`` below.
    """

    def __init__(self, feature_fn: ArrayFn, covariance_fn: ArrayFn, **kwargs):
        self.factor_fn = AccessibleFisherFactor(feature_fn, covariance_fn, **kwargs)
        self.last_metric = None

    def __call__(self, params, *args, **kwargs):
        a = self.factor_fn(params, *args, **kwargs)
        metric = a.T @ a
        metric = 0.5 * (metric + metric.T)
        self.last_metric = metric
        return pnp.array(metric, requires_grad=False)


class AQNGOptimizer:
    """Accessible Quantum Natural Gradient optimizer for PennyLane NumPy.

    The update is

        theta <- theta - stepsize * (G_acc + lam I)^+ grad L.

    ``solver='auto'`` uses the smaller exact linear system when ``lam > 0``:
    parameter space of size p or stacked readout space of size B*r. With
    ``lam == 0`` an SVD pseudoinverse is used.
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
        solver: str = "auto",
    ):
        if stepsize <= 0:
            raise ValueError("stepsize must be positive")
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        if solver not in ("auto", "primal", "dual", "svd"):
            raise ValueError("solver must be 'auto', 'primal', 'dual', or 'svd'")

        self.stepsize = float(stepsize)
        self.lam = float(lam)
        self.cov_lam = float(cov_lam)
        self.rcond = float(rcond)
        self.project_cov_psd = bool(project_cov_psd)
        self.reduction = reduction
        self.solver = solver

        self.factor_fn: Optional[AccessibleFisherFactor] = None
        self._cached_factor: Optional[onp.ndarray] = None
        self._cached_param_shape = None
        self.last_direction = None
        self.last_diagnostics: Optional[MetricDiagnostics] = None

    @property
    def diagnostics(self) -> Optional[MetricDiagnostics]:
        return self.last_diagnostics

    @property
    def metric_tensor(self):
        """Explicit last metric, computed lazily for diagnostics only."""
        if self._cached_factor is None:
            return None
        metric = self._cached_factor.T @ self._cached_factor
        return pnp.array(metric, requires_grad=False)

    def _make_factor(self, feature_fn: ArrayFn, covariance_fn: ArrayFn):
        self.factor_fn = AccessibleFisherFactor(
            feature_fn,
            covariance_fn,
            cov_lam=self.cov_lam,
            rcond=self.rcond,
            project_cov_psd=self.project_cov_psd,
            reduction=self.reduction,
        )
        return self.factor_fn

    def _select_solver(self, m: int, p: int) -> str:
        if self.lam == 0.0:
            return "svd"
        if self.solver == "auto":
            return "dual" if m < p else "primal"
        return self.solver

    def _solve(self, a: onp.ndarray, grad: onp.ndarray):
        """Solve ``(A.T A + lam I) direction = grad``."""
        m, p = a.shape
        solver = self._select_solver(m, p)

        if solver == "primal":
            gmat = a.T @ a
            system = gmat + self.lam * onp.eye(p)
            try:
                direction = onp.linalg.solve(system, grad)
            except onp.linalg.LinAlgError:
                direction = onp.linalg.pinv(system, rcond=self.rcond) @ grad
            solve_dim = p

        elif solver == "dual":
            if self.lam <= 0.0:
                raise ValueError("dual Woodbury solve requires lam > 0")
            system = a @ a.T + self.lam * onp.eye(m)
            rhs = a @ grad
            try:
                y = onp.linalg.solve(system, rhs)
            except onp.linalg.LinAlgError:
                y = onp.linalg.pinv(system, rcond=self.rcond) @ rhs
            direction = (grad - a.T @ y) / self.lam
            solve_dim = m

        elif solver == "svd":
            _, s, vt = onp.linalg.svd(a, full_matrices=False)
            s2 = s * s
            coeff = vt @ grad
            if self.lam > 0.0:
                direction = vt.T @ (coeff / (s2 + self.lam))
                projected = vt.T @ coeff
                direction += (grad - projected) / self.lam
            else:
                scale = max(float(onp.max(s2)) if s2.size else 0.0, 1.0)
                keep = s2 > self.rcond * scale
                inv = onp.zeros_like(s2)
                inv[keep] = 1.0 / s2[keep]
                direction = vt.T @ (inv * coeff)
            solve_dim = min(m, p)

        else:  # pragma: no cover
            raise RuntimeError(f"unknown solver {solver}")

        return direction, solver, solve_dim

    def _compute_grad_and_forward(
        self, objective_fn, params, args, kwargs, grad_fn=None, need_forward=False
    ):
        if grad_fn is None:
            grad_callable = qml.grad(objective_fn, argnums=0)
            grad = grad_callable(params, *args, **kwargs)
            forward = grad_callable.forward if need_forward else None
        else:
            grad = grad_fn(params, *args, **kwargs)
            forward = objective_fn(params, *args, **kwargs) if need_forward else None
        return grad, forward

    def _direction_and_diagnostics(self, factor, grad_flat):
        direction, solver, solve_dim = self._solve(factor, grad_flat)
        s = onp.linalg.svd(factor, compute_uv=False)
        eigs = s * s
        scale = max(float(onp.max(eigs)) if eigs.size else 0.0, 1.0)
        keep = eigs > self.rcond * scale
        rank = int(onp.sum(keep))
        pos = eigs[keep]
        condition = float(onp.max(pos) / onp.min(pos)) if pos.size > 1 else 1.0

        batch = int(self.factor_fn.last_batch_size) if self.factor_fn else 0
        r = int(self.factor_fn.last_feature_dim) if self.factor_fn else 0
        p = int(grad_flat.size)
        self.last_diagnostics = MetricDiagnostics(
            batch_size=batch,
            feature_dim=r,
            parameter_dim=p,
            factor_rows=int(factor.shape[0]),
            metric_rank=rank,
            metric_trace=float(onp.sum(eigs)),
            metric_condition=condition,
            solver=solver,
            solve_dimension=int(solve_dim),
            gradient_norm=float(onp.linalg.norm(grad_flat)),
            natural_gradient_norm=float(onp.linalg.norm(direction)),
        )
        return direction

    def _step_impl(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        grad_fn: Optional[ArrayFn] = None,
        recompute_tensor: bool = True,
        need_forward: bool = False,
        **kwargs,
    ):
        grad, forward = self._compute_grad_and_forward(
            objective_fn, params, args, kwargs, grad_fn=grad_fn, need_forward=need_forward
        )
        p_shape = tuple(qml.math.shape(params))
        grad_flat = _to_numpy(grad).reshape(-1)
        p = int(onp.prod(p_shape)) if p_shape else 1
        if grad_flat.size != p:
            raise ValueError(
                "AQNGOptimizer supports one trainable parameter array; gradient size "
                f"{grad_flat.size} does not match parameter size {p}."
            )

        if recompute_tensor or self._cached_factor is None:
            factor_fn = self._make_factor(feature_fn, covariance_fn)
            factor = factor_fn(params, *args, **kwargs)
            self._cached_factor = factor
            self._cached_param_shape = p_shape
        else:
            factor = self._cached_factor
            if self._cached_param_shape != p_shape:
                raise ValueError("parameter shape changed while reusing cached AQNG metric")

        if factor.shape[1] != p:
            raise ValueError(
                f"AQNG factor parameter dimension {factor.shape[1]} != gradient dimension {p}"
            )

        direction = self._direction_and_diagnostics(factor, grad_flat)
        self.last_direction = direction.copy()
        updated = _to_numpy(params) - self.stepsize * direction.reshape(p_shape)
        new_params = pnp.array(updated, requires_grad=getattr(params, "requires_grad", True))
        return new_params, forward

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
        """Take one AQNG step and return updated parameters."""
        new_params, _ = self._step_impl(
            objective_fn,
            params,
            *args,
            feature_fn=feature_fn,
            covariance_fn=covariance_fn,
            grad_fn=grad_fn,
            recompute_tensor=recompute_tensor,
            need_forward=False,
            **kwargs,
        )
        return new_params

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
        return self._step_impl(
            objective_fn,
            params,
            *args,
            feature_fn=feature_fn,
            covariance_fn=covariance_fn,
            grad_fn=grad_fn,
            recompute_tensor=recompute_tensor,
            need_forward=True,
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
    """Return explicit ``G_acc`` for PennyLane ``metric_tensor_fn``.

    This compatibility path materializes a p x p matrix. Use
    ``AQNGOptimizer`` when you want automatic primal/dual solving.
    """
    return AccessibleFisherMetric(
        feature_fn,
        covariance_fn,
        cov_lam=cov_lam,
        rcond=rcond,
        project_cov_psd=project_cov_psd,
        reduction=reduction,
    )


__all__ = [
    "AQNGOptimizer",
    "AccessibleFisherFactor",
    "AccessibleFisherMetric",
    "MetricDiagnostics",
    "covariance_from_moments",
    "make_aqng_metric_tensor_fn",
    "z_covariance_from_bitstrings",
    "z_features_from_bitstrings",
]
