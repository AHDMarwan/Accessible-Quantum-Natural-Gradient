import numpy as np

from aqng_validation import (
    fixed_support_indices,
    stabilize_reference_probabilities,
    normalized_reference_score_rows_fixed_support,
)


def test_u1_half_filled_support_is_shot_independent():
    s = fixed_support_indices(6, "u1_rzxy")
    assert len(s) == 20
    assert all(int(i).bit_count() == 3 for i in s)


def test_generic_support_is_full_basis():
    s = fixed_support_indices(6, "su2_haar")
    assert np.array_equal(s, np.arange(64))


def test_pseudocount_keeps_zero_count_allowed_outcomes():
    support = fixed_support_indices(4, "u1_rzxy")
    p = np.zeros(16)
    p[support[:2]] = [0.7, 0.3]
    p1 = stabilize_reference_probabilities(p, support, shots=1000, pseudocount=0.5)
    p2 = stabilize_reference_probabilities(p, support, shots=10000, pseudocount=0.5)
    assert np.all(p1[support] > 0)
    assert np.all(p2[support] > 0)
    assert np.count_nonzero(p1) == len(support)
    assert np.count_nonzero(p2) == len(support)
    assert np.isclose(p1.sum(), 1.0)
    assert np.isclose(p2.sum(), 1.0)


def test_fixed_score_rows_keep_requested_dimension():
    D, p = 8, 3
    support = np.arange(D)
    probs = np.full((2, D), 1 / D)
    rng = np.random.default_rng(4)
    jacs = rng.normal(size=(2, D, p))
    jacs -= jacs.mean(axis=1, keepdims=True)
    dirs = rng.normal(size=(5, p))
    pref = stabilize_reference_probabilities(probs.mean(0), support)
    rows = normalized_reference_score_rows_fixed_support(
        probs, jacs, dirs, reference_probabilities=pref, support_indices=support
    )
    assert rows.shape[1] == D
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)
