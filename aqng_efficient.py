"""Efficient Accessible Quantum Natural Gradient (AQNG) for PennyLane.

This module builds on :mod:`aqng_pennylane` and reduces the dominant practical
cost of AQNG: repeatedly rebuilding the accessible Fisher geometry.

In addition to stochastic metric mini-batches and metric caching, this version
adds two safeguards for stale cached geometry:

* adaptive refresh: when a cached metric proposes an anomalously large natural
  gradient direction, rebuild the metric at the current parameters and solve
  again before accepting the step;
* trust-region clipping: bound the accepted direction in Euclidean norm and/or
  in the accessible metric norm

      ||Delta theta||_G = stepsize * ||A d||,

  where ``G_acc = A.T @ A`` and ``d`` is the damped natural-gradient direction.

For simulator benchmarks, define differentiable expectation-value QNodes with
``diff_method="adjoint"`` when supported. On finite-shot hardware, use
``diff_method="parameter-shift"``. The optimizer itself does not change QNode
differentiation settings.

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
    raw_direction_norm: float
    natural_gradient_norm: float
    raw_metric_step_norm: float
    metric_step_norm: float
    metric_recomputed: bool
    adaptive_refresh_triggered: bool
    refresh_reason: str
    metric_age: int
    metric_refreshes: int
    trust_region_clipped: bool
    clip_scale: float
    gradient_seconds: float
    metric_seconds: float
    solve_seconds: float
    total_step_seconds: float


class AQNGEfficientOptimizer:
    """Cached / stochastic-metric AQNG optimizer with stale-metric safeguards.

    ``solver='auto'`` chooses the smaller exact damped system: parameter-space
    dimension ``p`` or stacked readout dimension ``B_metric * r``.
    The objective and metric may use different mini-batches.
    """

    def __init__(
        self,
        stepsize: float = 0.01,
        *,
        lam: float = 1e-3,
        cov_lam: float = 0.0,
        metric_every: int = 2,
        adaptive_refresh: bool = True,
        refresh_direction_growth: Optional[float] = 2.5,
        max_direction_norm: Optional[float] = None,
        max_metric_step: Optional[float] = None,
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
        if refresh_direction_growth is not None and refresh_direction_growth <= 1.0:
            raise ValueError("refresh_direction_growth must be > 1 or None")
        if max_direction_norm is not None and max_direction_norm <= 0:
            raise ValueError("max_direction_norm must be positive or None")
        if max_metric_step is not None and max_metric_step <= 0:
            raise ValueError("max_metric_step must be positive or None")
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
        self.adaptive_refresh = bool(adaptive_refresh)
        self.refresh_direction_growth = (
            None if refresh_direction_growth is None else float(refresh_direction_growth)
        )
        self.max_direction_norm = (
            None if max_direction_norm is None else float(max_direction_norm)
        )
        self.max_metric_step = (
            None if max_metric_step is None else float(max_metric_step)
        )
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
        self._last_raw_direction_norm: Optional[float] = None

        self.last_direction = None
        self.last_diagnostics: Optional[EfficientDiagnostics] = None

    @property
    def diagnostics(self) -> Optional[EfficientDiagnostics]:
        return self.last_diagnostics

    @property
    def metric_tensor(self):
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
        self._cached_factor = None
        self._cached_param_shape = None
        self._factor_fn = None
        self._spectral_cache = None
        self._last_refresh_step = None

    def reset_history(self):
        self._last_raw_direction_norm = None

    def _make_factor(self, feature_fn: ArrayFn, covariance_fn: ArrayFn):
        return AccessibleFisherFactor(
            feature_fn,
            covariance_fn,
            cov_lam=self.cov_lam,
            rcond=self.rcond,
            project_cov_psd=self.project_cov_psd,
            reduction=self.reduction,
        )

    def _scheduled_refresh(self, recompute_metric: Optional[bool]) -> bool:
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

    def _refresh_factor(
        self,
        params,
        args,
        kwargs,
        feature_fn: ArrayFn,
        covariance_fn: ArrayFn,
        p_shape,
        p: int,
    ):
        factor_fn = self._make_factor(feature_fn, covariance_fn)
        factor = factor_fn(params, *args, **kwargs)
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
        return factor

    def _adaptive_refresh_reason(self, direction: onp.ndarray, using_stale_metric: bool):
        if not self.adaptive_refresh or not using_stale_metric:
            return ""
        norm = float(onp.linalg.norm(direction))
        if not onp.isfinite(norm) or not onp.all(onp.isfinite(direction)):
            return "nonfinite_direction"
        if self.max_direction_norm is not None and norm > self.max_direction_norm:
            return "max_direction_norm"
        if (
            self.refresh_direction_growth is not None
            and self._last_raw_direction_norm is not None
            and self._last_raw_direction_norm > 0.0
            and norm > self.refresh_direction_growth * self._last_raw_direction_norm
        ):
            return "direction_growth"
        return ""

    def _apply_trust_region(self, factor: onp.ndarray, direction: onp.ndarray):
        raw_direction_norm = float(onp.linalg.norm(direction))
        raw_metric_step_norm = float(
            self.stepsize * onp.linalg.norm(factor @ direction)
        )
        scale = 1.0
        if (
            self.max_direction_norm is not None
            and raw_direction_norm > self.max_direction_norm
            and raw_direction_norm > 0.0
        ):
            scale = min(scale, self.max_direction_norm / raw_direction_norm)
        if (
            self.max_metric_step is not None
            and raw_metric_step_norm > self.max_metric_step
            and raw_metric_step_norm > 0.0
        ):
            scale = min(scale, self.max_metric_step / raw_metric_step_norm)
        clipped = bool(scale < 1.0)
        accepted = direction * scale
        direction_norm = float(onp.linalg.norm(accepted))
        metric_step_norm = float(
            self.stepsize * onp.linalg.norm(factor @ accepted)
        )
        return (
            accepted,
            raw_direction_norm,
            direction_norm,
            raw_metric_step_norm,
            metric_step_norm,
            clipped,
            float(scale),
        )

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

        scheduled_refresh = self._scheduled_refresh(recompute_metric)
        metric_recomputed = bool(scheduled_refresh)
        adaptive_refresh_triggered = False
        refresh_reason = "scheduled" if scheduled_refresh else ""
        metric_seconds = 0.0
        solve_seconds = 0.0

        if scheduled_refresh:
            t_metric = perf_counter()
            factor = self._refresh_factor(
                params, args, kwargs, feature_fn, covariance_fn, p_shape, p
            )
            metric_seconds += perf_counter() - t_metric
        else:
            factor = self._cached_factor
            if self._cached_param_shape != p_shape:
                raise ValueError(
                    "parameter shape changed while reusing cached AQNG geometry"
                )

        t_solve = perf_counter()
        direction, solver, solve_dim = self._solve(factor, grad_flat)
        solve_seconds += perf_counter() - t_solve

        reason = self._adaptive_refresh_reason(
            direction, using_stale_metric=not scheduled_refresh
        )
        if reason:
            adaptive_refresh_triggered = True
            refresh_reason = reason
            metric_recomputed = True
            t_metric = perf_counter()
            factor = self._refresh_factor(
                params, args, kwargs, feature_fn, covariance_fn, p_shape, p
            )
            metric_seconds += perf_counter() - t_metric
            t_solve = perf_counter()
            direction, solver, solve_dim = self._solve(factor, grad_flat)
            solve_seconds += perf_counter() - t_solve

        if not onp.all(onp.isfinite(direction)):
            raise FloatingPointError(
                "AQNG produced a non-finite natural-gradient direction even "
                "after the available metric refresh."
            )

        (
            accepted_direction,
            raw_direction_norm,
            direction_norm,
            raw_metric_step_norm,
            metric_step_norm,
            clipped,
            clip_scale,
        ) = self._apply_trust_region(factor, direction)

        self._last_raw_direction_norm = raw_direction_norm
        self.last_direction = accepted_direction.copy()
        updated = (
            _to_numpy(params)
            - self.stepsize * accepted_direction.reshape(p_shape)
        )
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
            raw_direction_norm=raw_direction_norm,
            natural_gradient_norm=direction_norm,
            raw_metric_step_norm=raw_metric_step_norm,
            metric_step_norm=metric_step_norm,
            metric_recomputed=metric_recomputed,
            adaptive_refresh_triggered=adaptive_refresh_triggered,
            refresh_reason=refresh_reason,
            metric_age=int(age),
            metric_refreshes=int(self._metric_refreshes),
            trust_region_clipped=clipped,
            clip_scale=clip_scale,
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
