# AQNGOptimizer public API

The supported public imports are:

```python
from aqng import AQNGConfig, AQNGOptimizer, ReadoutMode
```

## Constructor

```python
AQNGOptimizer(
    stepsize=0.01,
    *,
    readout="physical",
    probability_fn=None,
    lam=1e-3,
    cov_lam=0.0,
    metric_every=2,
    adaptive_refresh=True,
    refresh_direction_growth=2.5,
    max_direction_norm=None,
    max_metric_step=None,
    solver="auto",
    rcond=1e-10,
    project_cov_psd=True,
    reduction="mean",
    seed=0,
    readout_order=1,
)
```

### Parameters

- `stepsize`: parameter update learning rate.
- `readout`: one of `"physical"`, `"random"`, or `"aligned"`.
- `probability_fn`: differentiable callable returning computational-basis probabilities with shape `(D,)` or `(B, D)`.
- `lam`: nonnegative Tikhonov damping applied to the accessible natural-gradient solve.
- `cov_lam`: nonnegative regularization used when whitening the retained feature covariance.
- `metric_every`: number of optimizer steps between scheduled metric refreshes.
- `adaptive_refresh`: rebuild stale geometry when the cached metric produces an anomalous direction.
- `refresh_direction_growth`: stale-direction growth factor that triggers adaptive refresh; `None` disables this trigger.
- `max_direction_norm`: optional Euclidean cap on the accepted natural-gradient direction.
- `max_metric_step`: optional trust-radius cap on `stepsize * sqrt(d^T G_acc d)`.
- `solver`: `"auto"`, `"primal"`, `"dual"`, or `"svd"`.
- `rcond`: relative numerical tolerance for pseudoinverses and rank decisions.
- `project_cov_psd`: project numerical feature covariances onto the PSD cone before whitening.
- `reduction`: `"mean"` or `"sum"` across the metric minibatch.
- `seed`: seed used for random rank-matched readouts and default calibration directions.
- `readout_order`: maximum diagonal Pauli/Walsh weight used to define the physical readout family.

## Typed configuration

```python
config = AQNGConfig(
    stepsize=0.06,
    readout="aligned",
    lam=3e-3,
    metric_every=2,
    max_direction_norm=8.0,
    max_metric_step=0.25,
    readout_order=1,
    seed=0,
)

optimizer = AQNGOptimizer.from_config(
    config,
    probability_fn=probability_fn,
)
```

`AQNGConfig` contains serializable numerical/readout policy only. User callables and datasets are deliberately excluded.

## Calibration

```python
optimizer.calibrate(
    params,
    calibration_inputs,
    n_qubits=n_qubits,
    n_directions=64,
)
```

Calibration differentiates `probability_fn` with respect to the trainable parameter array, builds normalized tangent-score rows, and fits the equal-rank physical, random, and aligned designs. The aligned design must be calibrated on inputs independent of the supervised minibatch used for evaluation.

Custom tangent directions can be supplied explicitly with `directions=...`.

## Optimization step

```python
params = optimizer.step(
    objective_fn,
    params,
    x_batch,
    y_batch,
    metric_args=(metric_batch,),
)
```

If `metric_args` and `metric_kwargs` are omitted, the metric callable receives the same arguments as the supervised objective. Supplying them separates the supervised and metric minibatches.

`step_and_cost(...)` returns `(new_params, objective_before_step)`.

## Readout switching

All rank-matched designs are fitted together, so switching does not require recalibration:

```python
optimizer.set_readout("physical")
optimizer.set_readout("random")
optimizer.set_readout("aligned")
```

Changing the active readout invalidates the cached metric.

## Persistence

```python
optimizer.save("aqng_state.aqng")

restored = AQNGOptimizer.load(
    "aqng_state.aqng",
    probability_fn=probability_fn,
)
```

The state archive stores hyperparameters and calibrated readout designs. It does **not** pickle or serialize Python callables, objective functions, datasets, or cached metric tensors. The archive is a ZIP containing JSON metadata and NumPy arrays loaded with `allow_pickle=False`.

## Diagnostics

After a successful step:

```python
diag = optimizer.diagnostics
metric = optimizer.metric_tensor
```

Diagnostics include metric rank/trace/condition estimate, solver and solve dimension, gradient/direction norms, trust-region clipping, refresh state, and timing information.

## Advanced interface

Advanced users may bypass automatic probability-based binding and provide custom differentiable feature and covariance functions with `bind_metric_functions(feature_fn, covariance_fn)`. The package-level `AQNGOptimizer` remains the supported high-level entry point; lower-level modules are retained mainly for benchmark reproducibility and numerical development.
