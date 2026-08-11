"""Standalone public AQNG optimizer API."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .calibration import calibration_score_rows
from .optimizer import AQNGOptimizer as _BaseAQNGOptimizer, ReadoutMode


class AQNGOptimizer(_BaseAQNGOptimizer):
    """High-level AQNG optimizer with automatic readout calibration.

    The optimizer can be used without manually constructing ``feature_fn`` and
    ``covariance_fn``.  Bind a differentiable computational-basis probability
    callable, calibrate the readout from unlabeled calibration inputs, then call
    :meth:`step` or :meth:`step_and_cost`.
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

        Parameters
        ----------
        params
            Parameter point at which calibration probabilities and Jacobians are
            evaluated.
        calibration_args, calibration_kwargs
            Unlabeled inputs forwarded to ``probability_fn``.  The callable may
            return one probability vector ``(D,)`` or a batch ``(B,D)``.
        n_qubits
            Number of measured qubits, used to construct the physical low-weight
            diagonal readout family.
        directions
            Optional tangent directions in flattened parameter coordinates.  If
            omitted, ``n_directions`` isotropic unit directions are generated.
        n_directions
            Number of random calibration directions when ``directions`` is omitted.
        reference_probabilities
            Optional fixed reference distribution for the common score space.
            By default the mean calibration distribution is used.

        Notes
        -----
        Calibration is label-free.  For ``readout='aligned'`` the caller should
        provide calibration inputs independent of the supervised minibatch used
        for optimization/evaluation.
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


__all__ = ["AQNGOptimizer", "ReadoutMode"]
