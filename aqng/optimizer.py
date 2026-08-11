"""High-level reusable optimizer facade for AQNG.

The experimental code historically exposed the optimizer core and readout-design
utilities separately.  :class:`AQNGOptimizer` provides one public object that
selects the readout strategy, stores all optimizer controls, and delegates the
validated numerical step to :class:`aqng_efficient.AQNGEfficientOptimizer`.

The public readout names are ``physical``, ``random``, and ``aligned``.  The
latter two map to the validated internal design names ``random_rank`` and
``aligned_crossfit``.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Mapping, Optional

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("AQNGOptimizer requires PennyLane.") from exc

from aqng_efficient import AQNGEfficientOptimizer
from aqng_readouts import ReadoutDesign, fit_rank_matched_readouts

ArrayFn = Callable[..., object]


class ReadoutMode(str, Enum):
    """Supported rank-matched readout strategies."""

    PHYSICAL = "physical"
    RANDOM = "random"
    ALIGNED = "aligned"


_READOUT_KEYS = {
    ReadoutMode.PHYSICAL.value: "physical",
    ReadoutMode.RANDOM.value: "random_rank",
    ReadoutMode.ALIGNED.value: "aligned_crossfit",
    "random_rank": "random_rank",
    "aligned_crossfit": "aligned_crossfit",
}


def _normalize_readout(value: str | ReadoutMode) -> str:
    key = value.value if isinstance(value, ReadoutMode) else str(value).lower()
    if key not in _READOUT_KEYS:
        allowed = "physical, random, aligned"
        raise ValueError(f"readout must be one of: {allowed}")
    return key


class AQNGOptimizer:
    """Reusable Accessible Quantum Natural Gradient optimizer.

    Parameters
    ----------
    stepsize
        Parameter-update learning rate.
    readout
        Readout strategy: ``'physical'``, ``'random'``, or ``'aligned'``.
    probability_fn
        Optional callable returning computational-basis probabilities.  When a
        :class:`~aqng_readouts.ReadoutDesign` is fitted/bound, this is used to
        construct differentiable feature and covariance functions automatically.
    lam, cov_lam, metric_every, adaptive_refresh, refresh_direction_growth,
    max_direction_norm, max_metric_step, solver, rcond, project_cov_psd,
    reduction
        Numerical controls delegated to the validated efficient AQNG core.

    Notes
    -----
    ``aligned`` is a cross-fitted readout and therefore requires an independent
    calibration sample.  The optimizer never silently fits alignment on the
    supervised minibatch used for the update.
    """

    def __init__(
        self,
        stepsize: float = 0.01,
        *,
        readout: str | ReadoutMode = ReadoutMode.PHYSICAL,
        probability_fn: Optional[ArrayFn] = None,
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
        seed: int = 0,
        readout_order: int = 1,
    ):
        self.readout = _normalize_readout(readout)
        self.probability_fn = probability_fn
        self.seed = int(seed)
        self.readout_order = int(readout_order)
        if self.readout_order < 1:
            raise ValueError("readout_order must be >= 1")

        self._designs: Optional[Mapping[str, ReadoutDesign]] = None
        self._design: Optional[ReadoutDesign] = None
        self._feature_fn: Optional[ArrayFn] = None
        self._covariance_fn: Optional[ArrayFn] = None

        self._core = AQNGEfficientOptimizer(
            stepsize=stepsize,
            lam=lam,
            cov_lam=cov_lam,
            metric_every=metric_every,
            adaptive_refresh=adaptive_refresh,
            refresh_direction_growth=refresh_direction_growth,
            max_direction_norm=max_direction_norm,
            max_metric_step=max_metric_step,
            solver=solver,
            rcond=rcond,
            project_cov_psd=project_cov_psd,
            reduction=reduction,
        )

    @property
    def core(self) -> AQNGEfficientOptimizer:
        """Validated low-level optimizer used for the numerical step."""
        return self._core

    @property
    def diagnostics(self):
        return self._core.diagnostics

    @property
    def metric_tensor(self):
        return self._core.metric_tensor

    @property
    def readout_design(self) -> Optional[ReadoutDesign]:
        return self._design

    @property
    def readout_name(self) -> str:
        """Canonical public readout name."""
        if self.readout in ("random_rank", "random"):
            return "random"
        if self.readout in ("aligned_crossfit", "aligned"):
            return "aligned"
        return "physical"

    def set_readout(self, readout: str | ReadoutMode) -> "AQNGOptimizer":
        """Switch to another already-fitted rank-matched readout design."""
        self.readout = _normalize_readout(readout)
        if self._designs is not None:
            self._bind_selected_design()
        self._core.reset_metric()
        return self

    def fit_readout(
        self,
        reference_probabilities: np.ndarray,
        score_rows: np.ndarray,
        *,
        n_qubits: int,
        readout_order: Optional[int] = None,
        seed: Optional[int] = None,
        probability_floor: float = 1e-13,
        svd_tolerance: float = 1e-10,
    ) -> ReadoutDesign:
        """Fit the three same-rank designs and bind the selected strategy.

        ``score_rows`` must come from an independent calibration sample when the
        selected strategy is ``aligned``.
        """
        order = self.readout_order if readout_order is None else int(readout_order)
        rng_seed = self.seed if seed is None else int(seed)
        self._designs = fit_rank_matched_readouts(
            reference_probabilities,
            score_rows,
            n_qubits=int(n_qubits),
            readout_order=order,
            seed=rng_seed,
            probability_floor=probability_floor,
            svd_tolerance=svd_tolerance,
        )
        self._bind_selected_design()
        self._core.reset_metric()
        return self._design

    def bind_readout_designs(
        self, designs: Mapping[str, ReadoutDesign]
    ) -> ReadoutDesign:
        """Bind precomputed physical/random/aligned designs."""
        self._designs = designs
        self._bind_selected_design()
        self._core.reset_metric()
        return self._design

    def _bind_selected_design(self) -> None:
        if self._designs is None:
            raise RuntimeError("no readout designs are available")
        internal = _READOUT_KEYS[self.readout]
        if internal not in self._designs:
            raise KeyError(f"readout design {internal!r} is not present")
        self._design = self._designs[internal]
        if self.probability_fn is not None:
            self._feature_fn, self._covariance_fn = self._functions_from_design(
                self.probability_fn, self._design
            )

    @staticmethod
    def _functions_from_design(
        probability_fn: ArrayFn, design: ReadoutDesign
    ) -> tuple[ArrayFn, ArrayFn]:
        features = pnp.array(design.outcome_features, requires_grad=False)

        def feature_fn(params, *args, **kwargs):
            probs = probability_fn(params, *args, **kwargs)
            probs = qml.math.asarray(probs)
            if qml.math.ndim(probs) == 1:
                return qml.math.dot(probs, features)
            return qml.math.dot(probs, features)

        def covariance_fn(params, *args, **kwargs):
            probs = probability_fn(params, *args, **kwargs)
            probs = qml.math.asarray(probs)
            if qml.math.ndim(probs) == 1:
                probs = qml.math.reshape(probs, (1, -1))
                squeeze = True
            else:
                squeeze = False
            means = qml.math.dot(probs, features)
            weighted_features = probs[:, :, None] * features[None, :, :]
            second = qml.math.einsum(
                "bdr,ds->brs", weighted_features, features
            )
            cov = second - qml.math.einsum("br,bs->brs", means, means)
            return cov[0] if squeeze else cov

        return feature_fn, covariance_fn

    def bind_probability_fn(self, probability_fn: ArrayFn) -> "AQNGOptimizer":
        """Bind/change the probability callable used by the selected readout."""
        self.probability_fn = probability_fn
        if self._design is not None:
            self._feature_fn, self._covariance_fn = self._functions_from_design(
                probability_fn, self._design
            )
        self._core.reset_metric()
        return self

    def bind_metric_functions(
        self, feature_fn: ArrayFn, covariance_fn: ArrayFn
    ) -> "AQNGOptimizer":
        """Bind custom differentiable feature/covariance functions."""
        self._feature_fn = feature_fn
        self._covariance_fn = covariance_fn
        self._core.reset_metric()
        return self

    def reset_metric(self) -> None:
        self._core.reset_metric()

    def step(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: Optional[ArrayFn] = None,
        covariance_fn: Optional[ArrayFn] = None,
        grad_fn: Optional[ArrayFn] = None,
        recompute_metric: Optional[bool] = None,
        **kwargs,
    ):
        """Perform one AQNG update.

        Custom ``feature_fn``/``covariance_fn`` may be supplied per call.  If
        omitted, the functions bound by ``probability_fn`` and the selected
        readout design are used.
        """
        f_fn = feature_fn or self._feature_fn
        c_fn = covariance_fn or self._covariance_fn
        if f_fn is None or c_fn is None:
            raise RuntimeError(
                "AQNG metric functions are not bound. Provide probability_fn and "
                "fit/bind a readout design, or pass feature_fn and covariance_fn."
            )
        return self._core.step(
            objective_fn,
            params,
            *args,
            feature_fn=f_fn,
            covariance_fn=c_fn,
            grad_fn=grad_fn,
            recompute_metric=recompute_metric,
            **kwargs,
        )

    def step_and_cost(
        self,
        objective_fn: ArrayFn,
        params,
        *args,
        feature_fn: Optional[ArrayFn] = None,
        covariance_fn: Optional[ArrayFn] = None,
        grad_fn: Optional[ArrayFn] = None,
        recompute_metric: Optional[bool] = None,
        **kwargs,
    ):
        """Perform one update and return ``(new_params, objective_before_step)``."""
        f_fn = feature_fn or self._feature_fn
        c_fn = covariance_fn or self._covariance_fn
        if f_fn is None or c_fn is None:
            raise RuntimeError(
                "AQNG metric functions are not bound. Provide probability_fn and "
                "fit/bind a readout design, or pass feature_fn and covariance_fn."
            )
        return self._core.step_and_cost(
            objective_fn,
            params,
            *args,
            feature_fn=f_fn,
            covariance_fn=c_fn,
            grad_fn=grad_fn,
            recompute_metric=recompute_metric,
            **kwargs,
        )
