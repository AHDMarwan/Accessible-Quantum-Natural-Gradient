import numpy as np

from aqng_efficient import AQNGEfficientOptimizer


def test_dual_matches_dense_damped_solve():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 20))
    grad = rng.normal(size=20)
    lam = 0.03

    opt = AQNGEfficientOptimizer(lam=lam, solver="dual")
    direction, solver, solve_dim = opt._solve(a, grad)

    expected = np.linalg.solve(a.T @ a + lam * np.eye(20), grad)
    assert solver == "dual"
    assert solve_dim == 8
    assert np.allclose(direction, expected, rtol=1e-9, atol=1e-9)


def test_adaptive_growth_trigger_only_for_stale_geometry():
    opt = AQNGEfficientOptimizer(
        adaptive_refresh=True,
        refresh_direction_growth=2.0,
        max_direction_norm=None,
    )
    previous = np.array([1.0, 0.0])
    opt._last_raw_direction_norm = np.linalg.norm(previous)

    assert opt._adaptive_refresh_reason(1.2 * previous, True) == ""
    assert opt._adaptive_refresh_reason(2.1 * previous, True) == "direction_growth"
    assert opt._adaptive_refresh_reason(3.0 * previous, False) == ""


def test_trust_region_obeys_direction_and_metric_bounds():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(4, 6))
    direction = rng.normal(size=6) * 20.0

    opt = AQNGEfficientOptimizer(
        stepsize=0.03,
        max_direction_norm=3.0,
        max_metric_step=0.05,
    )

    accepted, _, direction_norm, _, metric_step_norm, clipped, scale = (
        opt._apply_trust_region(a, direction)
    )

    assert clipped
    assert scale < 1.0
    assert direction_norm <= 3.0 + 1e-12
    assert metric_step_norm <= 0.05 + 1e-12
    assert np.all(np.isfinite(accepted))
