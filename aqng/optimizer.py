"""High-level reusable optimizer facade for AQNG."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("AQNGOptimizer requires PennyLane.") from exc

from aqng_readouts import ReadoutDesign, fit_rank_matched_readouts
from .core import ControlledAQNGCore
from .sampling import stabilized_probability_fn, validate_sampling_configuration

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

    ``shots`` describes the sampling budget used by the bound probability
    callable. The optimizer does not mutate a QNode's device shots; instead it
    applies a differentiable fixed-support Dirichlet regularization to the
    returned probabilities before constructing AQNG readout geometry.
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
        metric_normalization: str = "none",
        normalization_target: Optional[float] = None,
        damping_mode: str = "absolute",
        shots: Optional[int] = None,
        pseudocount: float = 0.5,
        support_policy: str = "full",
        support_indices: Optional[Sequence[int]] = None,
        seed: int = 0,
        readout_order: int = 1,
    ):
        self.readout = _normalize_readout(readout)
        self.seed = int(seed)
        self.readout_order = int(readout_order)
        if self.readout_order < 1:
            raise ValueError("readout_order must be >= 1")

        validate_sampling_configuration(
            shots=shots,
            pseudocount=pseudocount,
            support_policy=support_policy,
            support_indices=support_indices,
        )
        self.shots = None if shots is None else int(shots)
        self.pseudocount = float(pseudocount)
        self.support_policy = str(support_policy)
        self.support_indices = (
            None if support_indices is None else tuple(int(i) for i in support_indices)
        )
        self.probability_fn: Optional[ArrayFn] = None
        self._metric_probability_fn: Optional[ArrayFn] = None

        self._designs: Optional[Mapping[str, ReadoutDesign]] = None
        self._design: Optional[ReadoutDesign] = None
        self._feature_fn: Optional[ArrayFn] = None
        self._covariance_fn: Optional[ArrayFn] = None

        self._core = ControlledAQNGCore(
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
            metric_normalization=metric_normalization,
            normalization_target=normalization_target,
            damping_mode=damping_mode,
        )
        if probability_fn is not None:
            self.bind_probability_fn(probability_fn)

    @property
    def core(self) -> ControlledAQNGCore:
        return self._core

    @property
    def diagnostics(self):
        return self._core.diagnostics

    @property
    def metric_tensor(self):
        return self._core.metric_tensor

    @property
    def effective_damping(self) -> Optional[float]:
        return self._core.last_effective_damping

    @property
    def metric_scale(self) -> float:
        return float(self._core.last_metric_scale)

    @property
    def finite_shot(self) -> bool:
        """Whether the optimizer is configured for finite-shot probabilities."""
        return self.shots is not None

    @property
    def metric_probability_fn(self) -> Optional[ArrayFn]:
        """Probability callable actually used to construct/calibrate AQNG geometry."""
        return self._metric_probability_fn

    @property
    def readout_design(self) -> Optional[ReadoutDesign]:
        return self._design

    @property
    def readout_name(self) -> str:
        if self.readout in ("random_rank", "random"):
            return "random"
        if self.readout in ("aligned_crossfit", "aligned"):
            return "aligned"
        return "physical"

    def set_readout(self, readout: str | ReadoutMode) -> "AQNGOptimizer":
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
        if self._metric_probability_fn is not None:
            self._feature_fn, self._covariance_fn = self._functions_from_design(
                self._metric_probability_fn, self._design
            )

    @staticmethod
    def _functions_from_design(
        probability_fn: ArrayFn, design: ReadoutDesign
    ) -> tuple[ArrayFn, ArrayFn]:
        features = pnp.array(design.outcome_features, requires_grad=False)

        def feature_fn(params, *args, **kwargs):
            probs = qml.math.asarray(probability_fn(params, *args, **kwargs))
            return qml.math.dot(probs, features)

        def covariance_fn(params, *args, **kwargs):
            probs = qml.math.asarray(probability_fn(params, *args, **kwargs))
            if qml.math.ndim(probs) == 1:
                probs = qml.math.reshape(probs, (1, -1))
                squeeze = True
            else:
                squeeze = False
            means = qml.math.dot(probs, features)
            weighted_features = probs[:, :, None] * features[None, :, :]
            second = qml.math.einsum("bdr,ds->brs", weighted_features, features)
            cov = second - qml.math.einsum("br,bs->brs", means, means)
            return cov[0] if squeeze else cov

        return feature_fn, covariance_fn

    def bind_probability_fn(self, probability_fn: ArrayFn) -> "AQNGOptimizer":
        """Bind a probability callable and apply the configured sampling policy."""
        self.probability_fn = probability_fn
        self._metric_probability_fn = stabilized_probability_fn(
            probability_fn,
            shots=self.shots,
            pseudocount=self.pseudocount,
            support_policy=self.support_policy,
            support_indices=self.support_indices,
        )
        if self._design is not None:
            self._feature_fn, self._covariance_fn = self._functions_from_design(
                self._metric_probability_fn, self._design
            )
        self._core.reset_metric()
        return self

    def bind_metric_functions(
        self, feature_fn: ArrayFn, covariance_fn: ArrayFn
    ) -> "AQNGOptimizer":
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
