"""Public AQNG numerical core extensions.

This module layers metric normalization and scale-aware damping on top of the
validated cached AQNG core without changing the experiment implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from aqng_efficient import AQNGEfficientOptimizer


class ControlledAQNGCore(AQNGEfficientOptimizer):
    """AQNG core with explicit metric normalization and damping conventions."""

    def __init__(
        self,
        *args,
        metric_normalization: str = "none",
        normalization_target: Optional[float] = None,
        damping_mode: str = "absolute",
        **kwargs,
    ):
        mode = str(metric_normalization).lower()
        if mode not in {"none", "trace", "maxeig"}:
            raise ValueError("metric_normalization must be none, trace, or maxeig")
        dmode = str(damping_mode).lower()
        if dmode not in {"absolute", "mean_eig", "maxeig"}:
            raise ValueError("damping_mode must be absolute, mean_eig, or maxeig")
        if normalization_target is not None and float(normalization_target) <= 0:
            raise ValueError("normalization_target must be positive or None")
        self.metric_normalization = mode
        self.normalization_target = (
            None if normalization_target is None else float(normalization_target)
        )
        self.damping_mode = dmode
        self.last_metric_scale = 1.0
        self.last_effective_damping = None
        super().__init__(*args, **kwargs)

    def _normalize_factor(self, factor: np.ndarray) -> np.ndarray:
        factor = np.asarray(factor, dtype=float)
        if self.metric_normalization == "none":
            self.last_metric_scale = 1.0
            return factor

        p = int(factor.shape[1])
        if self.metric_normalization == "trace":
            denom = float(np.sum(factor * factor))
            target = float(p if self.normalization_target is None else self.normalization_target)
        else:
            s = np.linalg.svd(factor, compute_uv=False)
            denom = float(s[0] * s[0]) if s.size else 0.0
            target = float(1.0 if self.normalization_target is None else self.normalization_target)

        if not np.isfinite(denom) or denom <= 0.0:
            self.last_metric_scale = 1.0
            return factor
        scale = target / denom
        self.last_metric_scale = float(scale)
        return factor * np.sqrt(scale)

    def _effective_damping(self, factor: np.ndarray) -> float:
        if self.damping_mode == "absolute":
            value = float(self.lam)
        elif self.damping_mode == "mean_eig":
            p = max(int(factor.shape[1]), 1)
            value = float(self.lam) * float(np.sum(factor * factor)) / float(p)
        else:
            s = np.linalg.svd(factor, compute_uv=False)
            maxeig = float(s[0] * s[0]) if s.size else 0.0
            value = float(self.lam) * maxeig
        self.last_effective_damping = float(value)
        return float(value)

    def _refresh_factor(self, params, args, kwargs, feature_fn, covariance_fn, p_shape, p):
        factor = super()._refresh_factor(
            params, args, kwargs, feature_fn, covariance_fn, p_shape, p
        )
        factor = self._normalize_factor(factor)
        self._cached_factor = factor
        self._spectral_cache = self._spectral_summary(factor)
        return factor

    def _solve(self, a: np.ndarray, grad: np.ndarray):
        m, p = a.shape
        lam_eff = self._effective_damping(a)
        solver = "svd" if lam_eff == 0.0 else self._select_solver(m, p)

        if solver == "primal":
            system = a.T @ a + lam_eff * np.eye(p)
            try:
                direction = np.linalg.solve(system, grad)
            except np.linalg.LinAlgError:
                direction = np.linalg.pinv(system, rcond=self.rcond) @ grad
            solve_dim = p
        elif solver == "dual":
            if lam_eff <= 0.0:
                raise ValueError("dual Woodbury solve requires positive effective damping")
            system = a @ a.T + lam_eff * np.eye(m)
            rhs = a @ grad
            try:
                y = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                y = np.linalg.pinv(system, rcond=self.rcond) @ rhs
            direction = (grad - a.T @ y) / lam_eff
            solve_dim = m
        elif solver == "svd":
            _, s, vt = np.linalg.svd(a, full_matrices=False)
            s2 = s * s
            coeff = vt @ grad
            if lam_eff > 0.0:
                direction = vt.T @ (coeff / (s2 + lam_eff))
                projected = vt.T @ coeff
                direction += (grad - projected) / lam_eff
            else:
                scale = max(float(np.max(s2)) if s2.size else 0.0, 1.0)
                keep = s2 > self.rcond * scale
                inv = np.zeros_like(s2)
                inv[keep] = 1.0 / s2[keep]
                direction = vt.T @ (inv * coeff)
            solve_dim = min(m, p)
        else:  # pragma: no cover
            raise RuntimeError(f"unknown solver {solver}")

        return direction, solver, int(solve_dim)
