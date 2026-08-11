"""Minimal reusable AQNGOptimizer example.

The probability callable supplies the measurement model.  Calibration is
label-free and can use data independent of the supervised optimization batch.
"""

import pennylane as qml
from pennylane import numpy as np

from aqng import AQNGOptimizer


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def probability_fn(params, x):
    qml.RY(x, wires=0)
    qml.RX(0.5 * x, wires=1)
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])


def objective(params, x, target):
    probs = probability_fn(params, x)
    z0 = probs[0] + probs[1] - probs[2] - probs[3]
    return (z0 - target) ** 2


params = np.array([0.1, -0.2, 0.3], requires_grad=True)

optimizer = AQNGOptimizer(
    stepsize=0.06,
    readout="aligned",  # "physical" | "random" | "aligned"
    probability_fn=probability_fn,
    lam=3e-3,
    cov_lam=1e-3,
    metric_every=2,
    max_direction_norm=8.0,
    max_metric_step=0.25,
    readout_order=1,
    seed=0,
)

# Independent, unlabeled calibration point.  A batched probability callable can
# instead calibrate on a full calibration minibatch in one call.
optimizer.calibrate(params, 0.35, n_qubits=2, n_directions=64)

for _ in range(10):
    # The supervised objective and metric can use different data.
    params = optimizer.step(
        objective,
        params,
        0.9,
        0.4,
        metric_args=(0.6,),
    )

print("params:", params)
print("metric rank:", optimizer.diagnostics.metric_rank)
print("readout:", optimizer.readout_name)
