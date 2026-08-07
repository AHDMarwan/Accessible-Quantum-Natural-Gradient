"""Efficient Accessible Quantum Natural Gradient (AQNG) for PennyLane.

This module builds on :mod:`aqng_pennylane` and targets the dominant practical
cost of AQNG: repeatedly rebuilding the accessible Fisher geometry.

Key changes relative to ``AQNGOptimizer``:
  * cache the accessible Fisher factor for ``metric_every`` optimization steps;
  * allow the metric feature/covariance closures to use a smaller mini-batch
    than the objective/gradient mini-batch;
  * retain automatic primal/dual solving;
  * compute expensive spectral diagnostics only when the metric is refreshed;
  * expose timing diagnostics to identify the real bottleneck.

For simulator benchmarks, define the differentiable feature QNode with
``diff_method="adjoint"`` (when supported). On finite-shot hardware, use
``diff_method="parameter-shift"``. The optimizer itself does not change a
QNode's differentiation method.

Target: PennyLane 0.45.x, Autograd / ``pennylane.numpy``, one trainable
parameter array.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import numpy as onp

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aqng_efficient.py requires PennyLane. Install with `pip install pennylane`."
    ) from exc

from aqng_pennylane import AccessibleFisherFactor

ArrayFn = Callable[..., object]


def _to_numpy(x, dtype=float):
    return onp.asarray(qml.math.toarray(x), dtype=dtype)


@dataclass
class EfficientDiagnostics:
    step: int
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
    metric_recomputed: bool
    metric_age: int
    metric_refreshes: int
    gradient_seconds: float
    metric_seconds: float
    solve_seconds: float
    total_step_seconds: float


class AQNGEfficientOptimizer:
    """Cached / stochastic-metric AQNG optimizer for PennyLane.

    The objective and metric may use different mini-batches. Close
    ``objective_fn`` over the loss batch and ``feature_fn`` /
    ``covariance_fn`` over a smaller metric batch.

    ``solver="auto"`` uses the smaller exact damped system: parameter space
    dimension ``p`` or stacked readout dimension ``B_metric * r``.
    """

    def __init__(
        self,
        stepsize: float = 0.01,
        *,
        lam: float = 1e-3,
        cov_lam: float = 0.0,
        metric_every: int = 4,
        solver: str = "auto",
        rcond: float = 1e-10,
        project_cov_psd: bool = True,
        reduction: str = "mean",
    ):
        if stepsize <= 0:
            raise ValueError("stepsize must be positive")
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        if cov_lam < 0:
            raise ValueError("cov_lam must be nonnegative")
        if metric_every < 1:
            raise ValueError("metric_every must be >= 1")
        if solver not in ("auto", "primal", "dual", "svd"):
            raise ValueError("solver must be 'auto', 'primal', 'dual', or 'svd'")
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if rcond <= 0:
            raise ValueError("rcond must be positive")

        self.stepsize = float(stepsize)
        self.lam = float(lam)
        self.cov_lam = float(cov_lam)
        self.metric_every = int(metric_every)
        self.solver = solver
        self.rcond = float(rcond)
        self.project_cov_psd = bool(project_cov_psd)
        self.reduction = reduction

        self._cached_factor: Optional[onp.ndarray] = None
        self._cached_param_shape = None
        self._factor_fn: Optional[AccessibleFisherFactor] = None
        self._spectral_cache = None
        self._step_index = 0
        self._last_refresh_step: Optional[int] = None
        self._metric_refreshes = 0

        self.last_direction = None
        self.last_diagnostics: Optional[EfficientDiagnostics] = None

    @property
    def diagnostics(self) -> Optional[EfficientDiagnostics]:
        return self.last_diagnostics

    @property
    def metric_tensor(self):
        """Materialize the cached p x p metric only when explicitly requested."""
        if self._cached_factor is None:
            return None
        metric = self._cached_factor.T @ self._cached_factor
        return pnp.array(metric, requires_grad=False)

    @property
    def metric_age(self) -> Optional[int]:
        if self._last_refresh_step is None:
            return None
        return self._step_index - self._last_refresh_step

    def reset_metric(self):
        """Discard cached geometry; the next step will rebuild it."""
        self._cached_factor = None
        self._cached_param_shape = None
        self._factor_fn = None
        self._spectral_cache = None
        self._last_refresh_step = None

    def _make_factor(self, feature_fn: ArrayFn, covariance_fn: ArrayFn):
        return AccessibleFisherFactor(
            feature_fn,
            covariance_fn,
            cov_lam=self.cov_lam,
            rcond=self.rcond,
            project_cov_psd=self.project_cov_psd,
            reduction=self.reduction,
        )

    def _should_refresh(self, recompute_metric: Optional[bool]) -> bool:
        if recompute_metric is not None:
            return bool(recompute_metric) or self._cached_factor is None
        if self._cached_factor is None:
            return True
        return (self._step_index % self.metric_every) == 0

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
            system = a.T @ a + self.lam * onp.eye(p)
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

        return direction, solver, int(solve_dim)

    def _spectral_summary(self, factor: onp.ndarray):
        s = onp.linalg.svd(factor, compute_uv=False)
        eigs = s * s
        scale = max(float(onp.max(eigs)) if eigs.size else 0.0, 1.0)
        keep = eigs > self.rcond * scale
        rank = int(onp.sum(keep))
        pos = eigs[keep]
        condition = float(onp.max(pos) / onp.min(pos)) if pos.size > 1 else 1.0
        return rank, float(onp.sum(eigs)), condition

    def _gradient_and_forward(
        self,
        objective_fn: ArrayFn,
        params,
        args,
        kwargs,
        grad_fn: Optional[ArrayFn],
        need_forward: bool,
    ):
        if grad_fn is None:
            grad_callable = qml.grad(objective_fn, argnums=0)
            grad = grad_callable(params, *args, **kwargs)
            forward = getattr(grad_callable, "forward", None) if need_forward else None
            if need_forward and forward is None:
                forward = objective_fn(params, *args, **kwargs)
        else:
            grad = grad_fn(params, *args, **kwargs)
            forward = objective_fn(params, *args, **kwargs) if need_forward else None
        return grad, forward

    def _step_impl(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        grad_fn: Optional[ArrayFn] = None,
        recompute_metric: Optional[bool] = None,
        recompute_tensor: Optional[bool] = None,
        need_forward: bool = False,
        **kwargs,
    ):
        if recompute_metric is not None and recompute_tensor is not None:
            raise ValueError("pass only one of recompute_metric or recompute_tensor")
        if recompute_metric is None and recompute_tensor is not None:
            recompute_metric = bool(recompute_tensor)

        t_total = perf_counter()

        t_grad = perf_counter()
        grad, forward = self._gradient_and_forward(
            objective_fn, params, args, kwargs, grad_fn, need_forward
        )
        grad_seconds = perf_counter() - t_grad

        p_shape = tuple(qml.math.shape(params))
        grad_flat = _to_numpy(grad).reshape(-1)
        p = int(onp.prod(p_shape)) if p_shape else 1
        if grad_flat.size != p:
            raise ValueError(
                "AQNGEfficientOptimizer supports one trainable parameter array; "
                f"gradient size {grad_flat.size} != parameter size {p}."
            )

        refresh = self._should_refresh(recompute_metric)
        metric_seconds = 0.0

        if refresh:
            t_metric = perf_counter()
            factor_fn = self._make_factor(feature_fn, covariance_fn)
            factor = factor_fn(params, *args, **kwargs)
            metric_seconds = perf_counter() - t_metric

            if factor.shape[1] != p:
                raise ValueError(
                    f"AQNG factor parameter dimension {factor.shape[1]} "
                    f"!= gradient dimension {p}"
                )

            self._cached_factor = factor
            self._cached_param_shape = p_shape
            self._factor_fn = factor_fn
            self._spectral_cache = self._spectral_summary(factor)
            self._last_refresh_step = self._step_index
            self._metric_refreshes += 1
        else:
            factor = self._cached_factor
            if self._cached_param_shape != p_shape:
                raise ValueError(
                    "parameter shape changed while reusing cached AQNG geometry"
                )

        t_solve = perf_counter()
        direction, solver, solve_dim = self._solve(factor, grad_flat)
        solve_seconds = perf_counter() - t_solve

        self.last_direction = direction.copy()
        updated = _to_numpy(params) - self.stepsize * direction.reshape(p_shape)
        new_params = pnp.array(
            updated, requires_grad=getattr(params, "requires_grad", True)
        )

        rank, trace, condition = self._spectral_cache
        batch = int(self._factor_fn.last_batch_size) if self._factor_fn else 0
        r = int(self._factor_fn.last_feature_dim) if self._factor_fn else 0
        age = self._step_index - int(self._last_refresh_step)

        total_seconds = perf_counter() - t_total
        self.last_diagnostics = EfficientDiagnostics(
            step=self._step_index,
            batch_size=batch,
            feature_dim=r,
            parameter_dim=p,
            factor_rows=int(factor.shape[0]),
            metric_rank=rank,
            metric_trace=trace,
            metric_condition=condition,
            solver=solver,
            solve_dimension=solve_dim,
            gradient_norm=float(onp.linalg.norm(grad_flat)),
            natural_gradient_norm=float(onp.linalg.norm(direction)),
            metric_recomputed=bool(refresh),
            metric_age=int(age),
            metric_refreshes=int(self._metric_refreshes),
            gradient_seconds=float(grad_seconds),
            metric_seconds=float(metric_seconds),
            solve_seconds=float(solve_seconds),
            total_step_seconds=float(total_seconds),
        )

        self._step_index += 1
        return new_params, forward

    def step(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        grad_fn: Optional[ArrayFn] = None,
        recompute_metric: Optional[bool] = None,
        recompute_tensor: Optional[bool] = None,
        **kwargs,
    ):
        """Take one efficient AQNG step."""
        new_params, _ = self._step_impl(
            objective_fn,
            params,
            *args,
            feature_fn=feature_fn,
            covariance_fn=covariance_fn,
            grad_fn=grad_fn,
            recompute_metric=recompute_metric,
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
        recompute_metric: Optional[bool] = None,
        recompute_tensor: Optional[bool] = None,
        **kwargs,
    ):
        """Take one efficient AQNG step and return ``(new_params, old_cost)``."""
        return self._step_impl(
            objective_fn,
            params,
            *args,
            feature_fn=feature_fn,
            covariance_fn=covariance_fn,
            grad_fn=grad_fn,
            recompute_metric=recompute_metric,
            recompute_tensor=recompute_tensor,
            need_forward=True,
            **kwargs,
        )


AQNGEfficient = AQNGEfficientOptimizer

__all__ = [
    "AQNGEfficient",
    "AQNGEfficientOptimizer",
    "EfficientDiagnostics",
]
