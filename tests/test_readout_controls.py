import numpy as np

from aqng_readouts import (
    accessible_metric_from_probability_jacobians,
    fit_rank_matched_readouts,
    normalize_metric,
    readout_retention,
    samples_to_probabilities,
    solve_controlled_direction,
)


def _uniform_two_body_score_rows(n=3, count=64, seed=7):
    d = 2**n
    states = np.arange(d)
    bits = ((states[:, None] >> (n - 1 - np.arange(n))) & 1)
    z = 1.0 - 2.0 * bits
    # Under the uniform distribution these two-body Walsh functions are exactly
    # orthogonal to the physical weight-1 Walsh span.
    target = np.column_stack(
        [z[:, 0] * z[:, 1], z[:, 0] * z[:, 2], z[:, 1] * z[:, 2]]
    ) / np.sqrt(float(d))
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=(count, target.shape[1]))
    rows = coeff @ target.T
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows


def test_aligned_same_rank_recovers_heldout_target_subspace():
    n = 3
    d = 2**n
    p = np.full(d, 1.0 / d)
    fit_rows = _uniform_two_body_score_rows(n=n, count=80, seed=1)
    eval_rows = _uniform_two_body_score_rows(n=n, count=80, seed=2)

    designs = fit_rank_matched_readouts(
        p,
        fit_rows,
        n_qubits=n,
        readout_order=1,
        seed=13,
    )

    ranks = {design.rank for design in designs.values()}
    assert ranks == {n}
    assert designs["physical"].centered_dimension == d - 1

    r_phys = readout_retention(eval_rows, designs["physical"])
    r_align = readout_retention(eval_rows, designs["aligned_crossfit"])
    r_rand = readout_retention(eval_rows, designs["random_rank"])

    assert r_phys < 1e-10
    assert r_align > 1.0 - 1e-10
    assert 0.0 <= r_rand <= 1.0 + 1e-12


def test_reference_features_are_centered_and_whitened():
    rng = np.random.default_rng(3)
    d = 8
    raw = rng.uniform(0.2, 1.0, size=d)
    p = raw / raw.sum()
    rows = rng.normal(size=(40, d))
    sqrt_p = np.sqrt(p)
    rows -= (rows @ sqrt_p)[:, None] * sqrt_p[None, :]
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)

    designs = fit_rank_matched_readouts(
        p,
        rows,
        n_qubits=3,
        readout_order=1,
        seed=4,
    )
    for design in designs.values():
        f = design.outcome_features
        mu = p @ f
        cov = f.T @ (p[:, None] * f) - np.outer(mu, mu)
        np.testing.assert_allclose(mu, 0.0, atol=1e-10)
        np.testing.assert_allclose(cov, np.eye(design.rank), atol=1e-9)


def test_accessible_metric_is_invariant_under_invertible_feature_coordinates():
    rng = np.random.default_rng(5)
    b, d, p_dim, r = 4, 9, 6, 3
    probs = rng.dirichlet(np.ones(d), size=b)
    jacs = rng.normal(size=(b, d, p_dim))
    # Probability derivatives must sum to zero over outcomes.
    jacs -= jacs.mean(axis=1, keepdims=True)
    features = rng.normal(size=(d, r))
    transform = rng.normal(size=(r, r))
    while abs(np.linalg.det(transform)) < 0.2:
        transform = rng.normal(size=(r, r))

    g1, _ = accessible_metric_from_probability_jacobians(
        probs, jacs, features, rcond=1e-12
    )
    g2, _ = accessible_metric_from_probability_jacobians(
        probs, jacs, features @ transform, rcond=1e-12
    )
    np.testing.assert_allclose(g1, g2, rtol=1e-9, atol=1e-9)


def test_trace_normalization_and_metric_trust_radius():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(5, 7))
    metric = a.T @ a
    grad = rng.normal(size=7)

    normalized, scale = normalize_metric(metric, "trace")
    assert scale > 0.0
    np.testing.assert_allclose(np.trace(normalized), 7.0, rtol=1e-12, atol=1e-12)

    direction, diag = solve_controlled_direction(
        metric,
        grad,
        lam=1e-3,
        stepsize=0.1,
        metric_normalization="trace",
        max_direction_norm=0.4,
        max_metric_step=0.02,
    )
    assert np.all(np.isfinite(direction))
    assert diag["direction_norm"] <= 0.4 + 1e-12
    assert diag["metric_step_norm"] <= 0.02 + 1e-12
    assert diag["clip_scale"] <= 1.0


def test_samples_to_probabilities_uses_pennylane_big_endian_order():
    samples = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=int,
    )
    p = samples_to_probabilities(samples)
    expected = np.zeros(8)
    expected[0] = 0.25
    expected[1] = 0.25
    expected[4] = 0.50
    np.testing.assert_allclose(p, expected)
