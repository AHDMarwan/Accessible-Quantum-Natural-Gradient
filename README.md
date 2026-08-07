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

Package version `0.4.0` installs both modules.

## Standard corrected AQNG

`AQNGOptimizer` factors

\[
G_{\rm acc}=A^T A
\]

and solves

\[
(G_{\rm acc}+\lambda I)d=\nabla L.
\]

With `solver="auto"` and `lam > 0`, it chooses the smaller exact system:

- primal: size `p` when `p <= B*r`;
- dual (Woodbury): size `B*r` when `B*r < p`.

The mini-batch dual dimension is `B*r`, not generally `r`.

## Efficient AQNG

`AQNGEfficientOptimizer` targets the dominant practical cost: rebuilding the accessible Jacobian/metric.

It supports:

1. a smaller metric mini-batch than the objective/gradient batch;
2. scheduled metric caching via `metric_every`;
3. adaptive early refresh when stale geometry proposes an anomalously large direction;
4. Euclidean direction clipping through `max_direction_norm`;
5. an accessible-metric trust radius
   \[
   \|\Delta\theta\|_{G_{\rm acc}}=\eta\|A d\|\le \Delta_G;
   \]
6. automatic primal/dual solving;
7. timing and safety diagnostics.

Recommended benchmark configuration after the Iris efficiency tests:

```python
from aqng_efficient import AQNGEfficientOptimizer

aqng = AQNGEfficientOptimizer(
    stepsize=0.03,
    lam=1e-3,
    cov_lam=1e-3,
    metric_every=2,
    adaptive_refresh=True,
    refresh_direction_growth=2.0,
    max_direction_norm=8.0,
    max_metric_step=0.25,
    solver="auto",
    rcond=1e-8,
)
```

Use a normal loss batch and a smaller metric batch:

```python
loss_ids = batch_ids[t]      # e.g. 10 examples
metric_ids = loss_ids[:2]    # stochastic accessible geometry

cost = make_cost(X_train[loss_ids], y_train[loss_ids])
features, covariance = make_metric_fns(X_train[metric_ids])

theta, old_loss = aqng.step_and_cost(
    cost,
    theta,
    feature_fn=features,
    covariance_fn=covariance,
)
```

### Adaptive refresh

The metric is refreshed on the normal `metric_every` schedule. On a stale step, AQNG first solves with the cached factor. Before accepting the direction it can force an immediate refresh when:

- the direction becomes non-finite;
- `||d|| > max_direction_norm`;
- the raw direction norm grows by more than `refresh_direction_growth` relative to the previous step.

After an adaptive refresh the direction is solved again using the current metric.

### Trust region

After the final solve, the direction can be clipped by two independent safeguards:

```python
max_direction_norm=...
max_metric_step=...
```

`max_metric_step` bounds the actual parameter displacement in accessible Fisher length,

\[
\eta\|A d\|\le {\tt max\_metric\_step}.
\]

Set either value to `None` to disable that bound.

### Diagnostics

`aqng.diagnostics` includes:

- `metric_recomputed`, `adaptive_refresh_triggered`, `refresh_reason`, `metric_age`;
- `raw_direction_norm`, `natural_gradient_norm`;
- `raw_metric_step_norm`, `metric_step_norm`;
- `trust_region_clipped`, `clip_scale`;
- `batch_size`, `feature_dim`, `parameter_dim`;
- `solver`, `solve_dimension`;
- metric rank/trace/condition;
- gradient, metric, solve and total-step timings.

## Simulator differentiation

For analytic simulator benchmarks on compatible devices such as `default.qubit`, expectation-value QNodes can use:

```python
@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def feature_qnode(theta, x):
    ...
```

For finite-shot hardware use `diff_method="parameter-shift"`.

## Finite-shot Z readouts

For diagonal Z strings, all feature covariances can be estimated from the same computational-basis bitstrings:

```python
from aqng_pennylane import z_covariance_from_bitstrings

Sigma = z_covariance_from_bitstrings(
    samples,
    z_terms=[(0,), (1,), (0,1), (0,2)],
)
```

`covariance_fn` itself is not differentiated.

## PennyLane QNG baseline

```python
import pennylane as qml
qng = qml.QNGOptimizer(stepsize=0.03, approx="block-diag", lam=1e-3)
```

For a fair comparison keep the circuit, initialization, loss batches, loss function, tuning protocol and evaluation budget fixed. Report both AQNG's loss batch and metric batch. Compare convergence versus optimization step, wall-clock time, circuit executions and total shots.

## Colab example

Use:

`examples/AQNG_Efficient_vs_PennyLane_QNG_Iris_Colab.ipynb`

The current notebook uses `metric_batch=2`, `metric_every=2`, adaptive refresh and trust-region safeguards, and logs all related diagnostics.
