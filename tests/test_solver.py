import numpy as np

from aqng_pennylane import AQNGOptimizer, AccessibleFisherFactor


def test_primal_dual_svd_match_dense_solution():
    rng = np.random.default_rng(1)
    lam = 0.13
    for m, p in [(5, 20), (20, 5), (10, 10)]:
        a = rng.normal(size=(m, p))
        grad = rng.normal(size=p)
        ref = np.linalg.solve(a.T @ a + lam * np.eye(p), grad)

        for solver in ("primal", "dual", "svd", "auto"):
            opt = AQNGOptimizer(lam=lam, solver=solver)
            direction, _, _ = opt._solve(a, grad)
            np.testing.assert_allclose(direction, ref, rtol=1e-10, atol=1e-10)


def test_zero_damping_matches_moore_penrose():
    rng = np.random.default_rng(2)
    for m, p in [(5, 20), (20, 5)]:
        a = rng.normal(size=(m, p))
        grad = rng.normal(size=p)
        ref = np.linalg.pinv(a.T @ a, rcond=1e-10) @ grad

        opt = AQNGOptimizer(lam=0.0, solver="auto")
        direction, used, _ = opt._solve(a, grad)
        assert used == "svd"
        np.testing.assert_allclose(direction, ref, rtol=1e-9, atol=1e-9)


def test_whitening_factor_matches_accessible_metric():
    rng = np.random.default_rng(3)
    r, p = 7, 11
    x = rng.normal(size=(r, r))
    sigma = x @ x.T
    sigma[-1, :] = 0.0
    sigma[:, -1] = 0.0
    jac = rng.normal(size=(r, p))

    factor_fn = AccessibleFisherFactor(lambda x: x, lambda x: x, cov_lam=0.0)
    weights, evecs = factor_fn._whitener_weights(sigma)
    a = weights[:, None] * (evecs.T @ jac)

    ref = jac.T @ np.linalg.pinv(sigma, rcond=1e-10) @ jac
    np.testing.assert_allclose(a.T @ a, ref, rtol=1e-9, atol=1e-9)
