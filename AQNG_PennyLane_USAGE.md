# AQNG with PennyLane — minimal usage

Target API: PennyLane 0.45.x.

The implementation is intentionally based on PennyLane's own `QNGOptimizer`.
PennyLane allows a custom `metric_tensor_fn`; AQNG supplies

\[
G_{\rm acc}=J^T\Sigma^+J
\]

instead of the Fubini–Study metric.

## 1. Your VQC readout

Use one trainable parameter array. For a real-data mini-batch, it is easiest to
close over `X_batch, y_batch` so only `theta` is trainable.

```python
import pennylane as qml
from pennylane import numpy as np
from aqng_pennylane import AQNGOptimizer, z_covariance_from_bitstrings

# Example diagonal readout dictionary. Extend as needed.
z_terms = [(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

# `feature_qnode(theta, x)` should return expectation values for these terms.
# `sample_qnode(theta, x)` should return computational-basis samples.
```

For PennyLane QNodes, make the feature QNode array-valued outside the QNode:

```python
def feature_vec(theta, x):
    # feature_qnode may return a tuple of expvals
    return qml.math.stack(feature_qnode(theta, x), axis=-1)

def covariance_one(theta, x):
    shots = sample_qnode(theta, x)       # shape (shots, n_wires)
    return z_covariance_from_bitstrings(shots, z_terms)
```

With finite-shot hardware, use `diff_method="parameter-shift"` for the feature
QNode. The covariance is not differentiated, so it can come directly from
bitstrings.

## 2. Mini-batch closures

```python
def make_batch_functions(X_batch, y_batch):
    def features(theta):
        return qml.math.stack([feature_vec(theta, x) for x in X_batch])

    def covariance(theta):
        return np.stack([covariance_one(theta, x) for x in X_batch])

    def cost(theta):
        pred = features(theta)
        # Replace with your actual differentiable loss.
        # Example: one selected feature as a regression/logit output.
        return np.mean((pred[:, 0] - y_batch) ** 2)

    return cost, features, covariance
```

## 3. AQNG training step

```python
aqng = AQNGOptimizer(
    stepsize=0.05,
    lam=1e-3,       # damping of G_acc before natural-gradient inversion
    cov_lam=1e-3,   # recommended for finite-shot covariance estimates
    rcond=1e-8,
)

for X_batch, y_batch in loader:
    cost, features, covariance = make_batch_functions(X_batch, y_batch)
    theta, old_cost = aqng.step_and_cost(
        cost,
        theta,
        feature_fn=features,
        covariance_fn=covariance,
    )
    print(float(old_cost), aqng.diagnostics)
```

## 4. PennyLane QNG baseline

Use your normal PennyLane QNG setup on a copy of the same initial parameters:

```python
qng = qml.QNGOptimizer(stepsize=0.05, approx="block-diag", lam=1e-3)
```

For a strict comparison, keep identical:

- circuit architecture and initialization;
- mini-batches and data order;
- loss function;
- number of optimization steps;
- shot budget, where possible;
- learning-rate tuning protocol.

AQNG and PennyLane QNG differ in the geometry: AQNG uses the Fisher information
available in the selected readout; PennyLane QNG uses the circuit-state
Fubini–Study metric (or its selected approximation).

## 5. Direct custom-metric mode

If you prefer not to use the wrapper class:

```python
from aqng_pennylane import make_aqng_metric_tensor_fn

metric = make_aqng_metric_tensor_fn(
    features,
    covariance,
    cov_lam=1e-3,
)

pl_opt = qml.QNGOptimizer(stepsize=0.05, approx=None, lam=1e-3)
theta = pl_opt.step(cost, theta, metric_tensor_fn=metric)
```

That is the cleanest apples-to-apples implementation: PennyLane performs the
same QNG update, while the supplied metric is `G_acc` rather than its native
Fubini–Study metric.
