"""Typed configuration for the public AQNG optimizer API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AQNGConfig:
    """Serializable configuration for :class:`aqng.AQNGOptimizer`."""

    stepsize: float = 0.01
    readout: str = "physical"
    lam: float = 1e-3
    cov_lam: float = 0.0
    metric_every: int = 2
    adaptive_refresh: bool = True
    refresh_direction_growth: Optional[float] = 2.5
    max_direction_norm: Optional[float] = None
    max_metric_step: Optional[float] = None
    solver: str = "auto"
    rcond: float = 1e-10
    project_cov_psd: bool = True
    reduction: str = "mean"
    metric_normalization: str = "none"
    normalization_target: Optional[float] = None
    damping_mode: str = "absolute"
    shots: Optional[int] = None
    pseudocount: float = 0.5
    support_policy: str = "full"
    support_indices: Optional[Tuple[int, ...]] = None
    seed: int = 0
    readout_order: int = 1

    def __post_init__(self):
        if self.stepsize <= 0:
            raise ValueError("stepsize must be positive")
        if self.lam < 0 or self.cov_lam < 0:
            raise ValueError("lam and cov_lam must be nonnegative")
        if self.metric_every < 1:
            raise ValueError("metric_every must be >= 1")
        if self.refresh_direction_growth is not None and self.refresh_direction_growth <= 1:
            raise ValueError("refresh_direction_growth must be > 1 or None")
        if self.max_direction_norm is not None and self.max_direction_norm <= 0:
            raise ValueError("max_direction_norm must be positive or None")
        if self.max_metric_step is not None and self.max_metric_step <= 0:
            raise ValueError("max_metric_step must be positive or None")
        if self.solver not in {"auto", "primal", "dual", "svd"}:
            raise ValueError("solver must be auto, primal, dual, or svd")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be mean or sum")
        if self.rcond <= 0:
            raise ValueError("rcond must be positive")
        if self.metric_normalization not in {"none", "trace", "maxeig"}:
            raise ValueError("metric_normalization must be none, trace, or maxeig")
        if self.normalization_target is not None and self.normalization_target <= 0:
            raise ValueError("normalization_target must be positive or None")
        if self.damping_mode not in {"absolute", "mean_eig", "maxeig"}:
            raise ValueError("damping_mode must be absolute, mean_eig, or maxeig")
        if self.shots is not None and int(self.shots) < 1:
            raise ValueError("shots must be a positive integer or None")
        if self.pseudocount < 0:
            raise ValueError("pseudocount must be nonnegative")
        if self.support_policy not in {"full", "custom"}:
            raise ValueError("support_policy must be full or custom")
        if self.support_policy == "custom":
            if self.support_indices is None or len(tuple(self.support_indices)) < 2:
                raise ValueError("custom support requires at least two support indices")
            indices = tuple(int(i) for i in self.support_indices)
            if min(indices) < 0 or len(set(indices)) != len(indices):
                raise ValueError("support_indices must be unique nonnegative integers")
            object.__setattr__(self, "support_indices", indices)
        elif self.support_indices is not None:
            object.__setattr__(
                self, "support_indices", tuple(int(i) for i in self.support_indices)
            )
        if self.readout_order < 1:
            raise ValueError("readout_order must be >= 1")
        if self.readout not in {"physical", "random", "aligned"}:
            raise ValueError("readout must be physical, random, or aligned")

    def to_dict(self) -> dict:
        values = asdict(self)
        if values["support_indices"] is not None:
            values["support_indices"] = list(values["support_indices"])
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "AQNGConfig":
        data = dict(values)
        if data.get("support_indices") is not None:
            data["support_indices"] = tuple(int(i) for i in data["support_indices"])
        return cls(**data)
