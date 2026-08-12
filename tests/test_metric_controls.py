import numpy as np

from aqng import AQNGConfig, AQNGOptimizer
from aqng.core import ControlledAQNGCore


def _core(**kwargs):
    return ControlledAQNGCore(
        stepsize=0.1,
        lam=0.03,
        metric_every=1,
        adaptive_refresh=False,
        **kwargs,
    )


def test_trace_normalization_targets_parameter_dimension():
    a = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]])
    core = _core(metric_normalization="trace")
    an = core._normalize_factor(a)
    assert np.isclose(np.sum(an * an), 2.0)
    assert core.last_metric_scale > 0.0


def test_maxeig_normalization_targets_one():
    a = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]])
    core = _core(metric_normalization="maxeig")
    an = core._normalize_factor(a)
    s = np.linalg.svd(an, compute_uv=False)
    assert np.isclose(s[0] ** 2, 1.0)


def test_trace_normalization_removes_global_factor_scale():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 3))
    grad = rng.normal(size=3)

    c1 = _core(metric_normalization="trace", damping_mode="absolute", solver="primal")
    c2 = _core(metric_normalization="trace", damping_mode="absolute", solver="primal")
    d1, _, _ = c1._solve(c1._normalize_factor(a), grad)
    d2, _, _ = c2._solve(c2._normalize_factor(7.0 * a), grad)
    assert np.allclose(d1, d2, rtol=1e-10, atol=1e-10)


def test_primal_dual_svd_agree_with_controlled_damping():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(3, 5))
    grad = rng.normal(size=5)

    directions = []
    for solver in ("primal", "dual", "svd"):
        core = _core(
            metric_normalization="trace",
            damping_mode="mean_eig",
            solver=solver,
        )
        an = core._normalize_factor(a)
        direction, _, _ = core._solve(an, grad)
        directions.append(direction)

    assert np.allclose(directions[0], directions[1], rtol=1e-9, atol=1e-9)
    assert np.allclose(directions[0], directions[2], rtol=1e-9, atol=1e-9)


def test_config_roundtrip_includes_metric_controls():
    config = AQNGConfig(
        metric_normalization="trace",
        normalization_target=4.0,
        damping_mode="maxeig",
    )
    restored = AQNGConfig.from_dict(config.to_dict())
    assert restored == config

    opt = AQNGOptimizer.from_config(config)
    assert opt.configuration.metric_normalization == "trace"
    assert opt.configuration.normalization_target == 4.0
    assert opt.configuration.damping_mode == "maxeig"
