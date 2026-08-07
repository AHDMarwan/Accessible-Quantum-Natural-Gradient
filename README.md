# AQNG for PennyLane

Accessible Quantum Natural Gradient (AQNG) uses the Fisher geometry visible through a selected measurement/readout space,

\[
G_{\rm acc}=\frac1B\sum_{b=1}^B J_b^T\Sigma_b^{+}J_b.
\]

Target API: PennyLane `0.45.x`, using the `pennylane.numpy` / Autograd workflow.

## Install in Colab

```python
!pip -q install "pennylane>=0.45,<0.46"
!pip -q install --upgrade "git+https://github.com/AHDMarwan/aqng.git"
```

Then import either implementation:

```python
from aqng_pennylane import AQNGOptimizer
from aqng_efficient import AQNGEfficientOptimizer
```

Package version `0.3.0` installs both `aqng_pennylane.py` and `aqng_efficient.py`.

## Standard corrected AQNG

`AQNGOptimizer` factors

\[
G_{\rm acc}=A^T A
\]

and solves

\[
(G_{\rm acc}+\lambda I)d=\nabla L
\]

directly. With `solver="auto"` and `lam > 0` it chooses the smaller exact system:

- **primal:** size `p` when `p <= B*r`;
- **dual (Woodbury):** size `B*r` when `B*r < p`.

The mini-batch dual dimension is `B*r`, not generally `r`.

```python
from aqng_pennylane import AQNGOptimizer

aqng = AQNGOptimizer(
    stepsize=0.03,
    lam=1e-3,
    cov_lam=1e-3,
    rcond=1e-8,
    solver="auto",
)
```

## Efficient AQNG

`AQNGEfficientOptimizer` targets the main practical cost: rebuilding the accessible Jacobian/metric every optimization step.

It adds:

1. **metric caching** with `metric_every`;
2. **smaller metric mini-batches**: the objective/gradient batch and metric batch can be different;
3. the same automatic primal/dual solve;
4. spectral diagnostics computed only on metric refresh;
5. timing diagnostics for gradient, metric construction, solve, and total step time.

Recommended first benchmark:

```python
from aqng_efficient import AQNGEfficientOptimizer

aqng = AQNGEfficientOptimizer(
    stepsize=0.03,
    lam=1e-3,
    cov_lam=1e-3,
    metric_every=4,
    solver="auto",
    rcond=1e-8,
)
```

Use a normal loss batch, e.g. 10 examples, but close `feature_fn` and `covariance_fn` over a smaller metric batch, e.g. 2 examples:

```python
loss_ids = batch_ids[t]          # e.g. 10 examples
metric_ids = loss_ids[:2]        # cheap stochastic geometry

cost = make_cost(X_train[loss_ids], y_train[loss_ids])
features, covariance = make_metric_fns(X_train[metric_ids])

theta, old_loss = aqng.step_and_cost(
    cost,
    theta,
    feature_fn=features,
    covariance_fn=covariance,
)

print(aqng.diagnostics)
```

With `metric_every=4`, the metric is rebuilt on steps 0, 4, 8, ... and reused in between. You can override this on a step with `recompute_metric=True` or `False`.

### Simulator differentiation

For simulator benchmarks on `default.qubit`, define expectation-value feature QNodes with an efficient simulator derivative when supported:

```python
@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def feature_qnode(theta, x):
    ...
```

PennyLane's adjoint method supports expectation-value Jacobians on compatible simulators. For finite-shot hardware, use `diff_method="parameter-shift"` instead.

### Diagnostics

`aqng.diagnostics` includes:

- `metric_recomputed` and `metric_age`;
- `batch_size`, `feature_dim`, `parameter_dim`;
- `solver` and `solve_dimension`;
- metric rank/trace/condition;
- `gradient_seconds`;
- `metric_seconds`;
- `solve_seconds`;
- `total_step_seconds`.

These fields are intended to identify whether the bottleneck is the objective gradient, accessible Jacobian, or linear solve.

## Finite-shot Z readouts

If the accessible features are diagonal Z strings, all feature covariances can be estimated from the same computational-basis bitstrings:

```python
from aqng_pennylane import z_covariance_from_bitstrings

Sigma = z_covariance_from_bitstrings(
    samples,
    z_terms=[(0,), (1,), (0,1), (0,2)],
)
```

Use `diff_method="parameter-shift"` for hardware-compatible feature Jacobians. `covariance_fn` itself is not differentiated.

## PennyLane QNG baseline

```python
import pennylane as qml
qng = qml.QNGOptimizer(stepsize=0.03, approx="block-diag", lam=1e-3)
```

For a fair comparison keep the circuit, initialization, loss batches, loss function, tuning protocol, and evaluation budget fixed. For AQNG, report both the loss batch and the smaller metric batch. Compare convergence versus:

1. optimization step;
2. wall-clock time;
3. circuit executions;
4. total shots.

## Colab example

Use:

`examples/AQNG_Efficient_vs_PennyLane_QNG_Iris_Colab.ipynb`

It compares efficient AQNG against PennyLane QNG on the Iris dataset and logs the timing diagnostics above.

## Explicit custom-metric compatibility

`make_aqng_metric_tensor_fn(...)` in `aqng_pennylane.py` is retained for experiments that intentionally pass `G_acc` into PennyLane's `QNGOptimizer`. That compatibility path materializes a `p x p` matrix and does not provide metric caching or the efficient solver path.
