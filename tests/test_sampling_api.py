import numpy as np
from pennylane import numpy as pnp

from aqng import AQNGConfig, AQNGOptimizer
from aqng.sampling import stabilize_probabilities


def _probability_fn(params, x=1.0):
    logits = pnp.stack(
        [x * params[0], x * params[1], -x * params[0], -x * params[1]]
    )
    weights = pnp.exp(logits)
    return weights / pnp.sum(weights)


def test_full_support_dirichlet_regularization():
    p = pnp.array([0.7, 0.3, 0.0, 0.0], requires_grad=True)
    out = stabilize_probabilities(p, shots=10, pseudocount=0.5)
    expected = (10.0 * np.asarray(p) + 0.5) / 12.0
    assert np.allclose(np.asarray(out), expected)
    assert np.isclose(float(pnp.sum(out)), 1.0)
    assert np.all(np.asarray(out) > 0.0)


def test_custom_support_masks_and_renormalizes():
    p = pnp.array([0.5, 0.2, 0.2, 0.1], requires_grad=True)
    out = stabilize_probabilities(
        p,
        shots=20,
        pseudocount=0.5,
        support_policy="custom",
        support_indices=(0, 3),
    )
    arr = np.asarray(out)
    assert np.isclose(arr.sum(), 1.0)
    assert arr[1] == 0.0 and arr[2] == 0.0
    assert arr[0] > arr[3] > 0.0


def test_sampling_configuration_roundtrip():
    config = AQNGConfig(
        readout="aligned",
        shots=1000,
        pseudocount=0.25,
        support_policy="custom",
        support_indices=(0, 1, 3),
    )
    restored = AQNGConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.support_indices == (0, 1, 3)


def test_finite_shot_calibration_and_step():
    params = pnp.array([0.17, -0.23], requires_grad=True)
    opt = AQNGOptimizer(
        stepsize=0.02,
        readout="physical",
        probability_fn=_probability_fn,
        lam=1e-2,
        metric_every=1,
        metric_normalization="trace",
        shots=1000,
        pseudocount=0.5,
        support_policy="full",
        readout_order=1,
    )
    assert opt.finite_shot
    opt.calibrate(params, 0.7, n_qubits=2, n_directions=12)

    def objective_fn(theta, x, target):
        probs = _probability_fn(theta, x)
        prediction = probs[0] - probs[2]
        return (prediction - target) ** 2

    new_params = opt.step(
        objective_fn,
        params,
        1.1,
        0.2,
        metric_args=(0.5,),
        recompute_metric=True,
    )
    assert new_params.shape == params.shape
    assert np.all(np.isfinite(np.asarray(new_params)))
    assert opt.metric_tensor is not None


def test_sampling_state_roundtrip(tmp_path):
    params = pnp.array([0.11, -0.19], requires_grad=True)
    opt = AQNGOptimizer(
        readout="random",
        probability_fn=_probability_fn,
        shots=500,
        pseudocount=0.75,
        support_policy="full",
        seed=4,
    )
    opt.calibrate(params, 0.9, n_qubits=2, n_directions=10)
    path = tmp_path / "finite_shot.aqng"
    opt.save(path)

    restored = AQNGOptimizer.load(path, probability_fn=_probability_fn)
    assert restored.shots == 500
    assert restored.pseudocount == 0.75
    assert restored.support_policy == "full"
    assert restored.readout_name == "random"
    assert restored.readout_design is not None
