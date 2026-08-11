"""Typed configuration for the public AQNG optimizer API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class AQNGConfig:
    """Serializable configuration for :class:`aqng.AQNGOptimizer`.

    This object contains only numerical/readout policy. Callables such as the
    probability function and objective are deliberately excluded.
    """

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
        if self.readout_order < 1:
            raise ValueError("readout_order must be >= 1")
        if self.readout not in {"physical", "random", "aligned"}:
            raise ValueError("readout must be physical, random, or aligned")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "AQNGConfig":
        return cls(**dict(values))
