import numpy as np
import pytest
from pennylane import numpy as pnp

from aqng import AQNGOptimizer, ReadoutMode


def test_public_readout_aliases():
    assert AQNGOptimizer(readout="physical").readout_name == "physical"
    assert AQNGOptimizer(readout="random").readout_name == "random"
    assert AQNGOptimizer(readout="aligned").readout_name == "aligned"
    assert AQNGOptimizer(readout=ReadoutMode.ALIGNED).readout_name == "aligned"


def test_invalid_readout_rejected():
    with pytest.raises(ValueError, match="physical, random, aligned"):
        AQNGOptimizer(readout="unknown")


def test_fit_and_switch_rank_matched_readouts():
    # Four-outcome reference distribution -> centered score dimension three.
    p_ref = np.array([0.25, 0.25, 0.25, 0.25])
    score_rows = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [1.0, 0.0, -1.0, 0.0],
        ]
    )
    score_rows = score_rows / np.linalg.norm(score_rows, axis=1, keepdims=True)

    opt = AQNGOptimizer(readout="physical", readout_order=1, seed=3)
    physical = opt.fit_readout(p_ref, score_rows, n_qubits=2)
    assert physical.name == "physical"
    assert physical.rank == 2

    random_design = opt.set_readout("random").readout_design
    assert random_design.name == "random_rank"
    assert random_design.rank == physical.rank

    aligned_design = opt.set_readout("aligned").readout_design
    assert aligned_design.name == "aligned_crossfit"
    assert aligned_design.rank == physical.rank


def test_standalone_calibrate_from_probability_callable():
    def probability_fn(params):
        logits = pnp.stack([params[0], params[1], -params[0], -params[1]])
        weights = pnp.exp(logits)
        return weights / pnp.sum(weights)

    params = pnp.array([0.17, -0.23], requires_grad=True)
    opt = AQNGOptimizer(
        readout="aligned",
        probability_fn=probability_fn,
        readout_order=1,
        seed=7,
    )
    design = opt.calibrate(params, n_qubits=2, n_directions=16)

    assert design.name == "aligned_crossfit"
    assert design.rank == 2
    assert opt.readout_design is design

    random_design = opt.set_readout("random").readout_design
    assert random_design.rank == design.rank


def test_standalone_calibrate_requires_probability_fn():
    opt = AQNGOptimizer(readout="physical")
    params = pnp.array([0.1, 0.2], requires_grad=True)
    with pytest.raises(RuntimeError, match="requires probability_fn"):
        opt.calibrate(params, n_qubits=2)
