"""Standalone public AQNG optimizer API."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .calibration import calibration_score_rows
from .config import AQNGConfig
from .optimizer import AQNGOptimizer as _BaseAQNGOptimizer, ReadoutMode
from .state import load_optimizer_state, save_optimizer_state


class AQNGOptimizer(_BaseAQNGOptimizer):
    """High-level AQNG optimizer with automatic readout calibration."""

    @classmethod
    def from_config(cls, config: AQNGConfig, *, probability_fn=None) -> "AQNGOptimizer":
        values = config.to_dict()
        return cls(probability_fn=probability_fn, **values)

    @property
    def configuration(self) -> AQNGConfig:
        core = self._core
        return AQNGConfig(
            stepsize=core.stepsize,
            readout=self.readout_name,
            lam=core.lam,
            cov_lam=core.cov_lam,
            metric_every=core.metric_every,
            adaptive_refresh=core.adaptive_refresh,
            refresh_direction_growth=core.refresh_direction_growth,
            max_direction_norm=core.max_direction_norm,
            max_metric_step=core.max_metric_step,
            solver=core.solver,
            rcond=core.rcond,
            project_cov_psd=core.project_cov_psd,
            reduction=core.reduction,
            metric_normalization=core.metric_normalization,
            normalization_target=core.normalization_target,
            damping_mode=core.damping_mode,
            seed=self.seed,
            readout_order=self.readout_order,
        )

    def calibrate(
        self,
        params,
        *calibration_args,
        n_qubits: int,
        directions: Optional[np.ndarray] = None,
        n_directions: int = 64,
        reference_probabilities: Optional[np.ndarray] = None,
        readout_order: Optional[int] = None,
        seed: Optional[int] = None,
        probability_floor: float = 1e-13,
        tangent_floor: float = 1e-13,
        svd_tolerance: float = 1e-10,
        **calibration_kwargs,
    ):
        if self.probability_fn is None:
            raise RuntimeError(
                "calibrate() requires probability_fn. Pass it to AQNGOptimizer(...) "
                "or call bind_probability_fn(...) first."
            )
        rng_seed = self.seed if seed is None else int(seed)
        rows, p_ref, _support = calibration_score_rows(
            self.probability_fn,
            params,
            *calibration_args,
            directions=directions,
            n_directions=n_directions,
            seed=rng_seed,
            reference_probabilities=reference_probabilities,
            probability_floor=probability_floor,
            tangent_floor=tangent_floor,
            **calibration_kwargs,
        )
        return self.fit_readout(
            p_ref,
            rows,
            n_qubits=int(n_qubits),
            readout_order=readout_order,
            seed=rng_seed,
            probability_floor=probability_floor,
            svd_tolerance=svd_tolerance,
        )

    fit = calibrate

    def save(self, path):
        return save_optimizer_state(
            path,
            config=self.configuration.to_dict(),
            readout=self.readout_name,
            designs=self._designs,
        )

    @classmethod
    def load(cls, path, *, probability_fn=None) -> "AQNGOptimizer":
        metadata, designs = load_optimizer_state(path)
        config = AQNGConfig.from_dict(metadata["config"])
        optimizer = cls.from_config(config, probability_fn=probability_fn)
        if designs:
            optimizer.bind_readout_designs(designs)
        return optimizer

    def _metric_functions_for_step(
        self,
        metric_args: Optional[Sequence[object]],
        metric_kwargs: Optional[Mapping[str, object]],
    ):
        if self._feature_fn is None or self._covariance_fn is None:
            raise RuntimeError(
                "AQNG metric functions are not bound. Provide probability_fn and "
                "call calibrate()/fit_readout(), or bind custom metric functions."
            )
        if metric_args is None and metric_kwargs is None:
            return self._feature_fn, self._covariance_fn

        m_args = tuple(() if metric_args is None else metric_args)
        m_kwargs = dict({} if metric_kwargs is None else metric_kwargs)
        feature_base = self._feature_fn
        covariance_base = self._covariance_fn

        def feature_fn(params, *_objective_args, **_objective_kwargs):
            return feature_base(params, *m_args, **m_kwargs)

        def covariance_fn(params, *_objective_args, **_objective_kwargs):
            return covariance_base(params, *m_args, **m_kwargs)

        return feature_fn, covariance_fn

    def step(
        self,
        objective_fn,
        params,
        *objective_args,
        metric_args: Optional[Sequence[object]] = None,
        metric_kwargs: Optional[Mapping[str, object]] = None,
        grad_fn=None,
        recompute_metric: Optional[bool] = None,
        **objective_kwargs,
    ):
        feature_fn, covariance_fn = self._metric_functions_for_step(
            metric_args, metric_kwargs
        )
        return self._core.step(
            objective_fn,
            params,
            *objective_args,
            feature_fn=feature_fn,
            covariance_fn=covariance_fn,
            grad_fn=grad_fn,
            recompute_metric=recompute_metric,
            **objective_kwargs,
        )

    def step_and_cost(
        self,
        objective_fn,
        params,
        *objective_args,
        metric_args: Optional[Sequence[object]] = None,
        metric_kwargs: Optional[Mapping[str, object]] = None,
        grad_fn=None,
        recompute_metric: Optional[bool] = None,
        **objective_kwargs,
    ):
        feature_fn, covariance_fn = self._metric_functions_for_step(
            metric_args, metric_kwargs
        )
        return self._core.step_and_cost(
            objective_fn,
            params,
            *objective_args,
            feature_fn=feature_fn,
            covariance_fn=covariance_fn,
            grad_fn=grad_fn,
            recompute_metric=recompute_metric,
            **objective_kwargs,
        )


__all__ = ["AQNGOptimizer", "AQNGConfig", "ReadoutMode"]
