"""Standalone public AQNG optimizer API."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .calibration import calibration_score_rows
from .optimizer import AQNGOptimizer as _BaseAQNGOptimizer, ReadoutMode


class AQNGOptimizer(_BaseAQNGOptimizer):
    """High-level AQNG optimizer with automatic readout calibration.

    The optimizer can be used without manually constructing ``feature_fn`` and
    ``covariance_fn``. Bind a differentiable computational-basis probability
    callable, calibrate the readout from unlabeled calibration inputs, then call
    :meth:`step` or :meth:`step_and_cost`.

    The supervised objective and the AQNG metric may consume different data.
    Pass ``metric_args`` / ``metric_kwargs`` to a step when the metric minibatch
    differs from the objective minibatch.
    """

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
        """Calibrate and bind physical/random/aligned readouts from ``probability_fn``.

        ``calibration_args`` and ``calibration_kwargs`` are forwarded only to the
        probability callable. Calibration is label-free. For ``readout='aligned'``
        the calibration inputs should be independent of the supervised minibatch
        used for optimization/evaluation.
        """
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
        """Perform one AQNG update.

        By default the probability/readout metric receives the same arguments as
        ``objective_fn``. Set ``metric_args`` and/or ``metric_kwargs`` to use a
        distinct metric minibatch while leaving the supervised objective call
        unchanged.
        """
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
        """Perform one update and return ``(new_params, objective_before_step)``."""
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


__all__ = ["AQNGOptimizer", "ReadoutMode"]
