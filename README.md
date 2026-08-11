# AQNG for PennyLane

Accessible Quantum Natural Gradient (AQNG) uses the Fisher geometry visible through a selected measurement/readout space,

\[
G_{\rm acc}=\frac1B\sum_{b=1}^B J_b^T\Sigma_b^{+}J_b.
\]

Target API: PennyLane `0.45.x`, using the `pennylane.numpy` / Autograd workflow.

The rank-matched readout controls in version `0.5.0` are based on the score-space construction used in the companion spectral-geometry project:

`AHDMarwan/Spectral-Geometry-of-Accessible-Quantum-Tangents-Beyond-Isotropic-Readout-Rank-Laws`.

## Install in Colab

```python
!pip -q install "pennylane>=0.45,<0.46"
!pip -q install --upgrade "git+https://github.com/AHDMarwan/Accessible-Quantum-Natural-Gradient.git"
```

Then import the optimizer implementations and, when needed, the readout-control utilities:

```python
from aqng_pennylane import AQNGOptimizer
from aqng_efficient import AQNGEfficientOptimizer
from aqng_readouts import fit_rank_matched_readouts, solve_controlled_direction
```

Package version `0.5.0` installs all three modules.

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
loss_ids = batch_ids[t]
metric_ids = loss_ids[:2]

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

## AQNG v2: rank-matched orientation controls

`aqng_readouts.py` ports the relevant score-space machinery from the spectral-geometry reproducibility code. For one calibration reference distribution it constructs three readouts with exactly the same covariance rank:

- `physical`: the low-weight diagonal Walsh/Pauli-Z span;
- `random_rank`: a Haar-random centered score subspace of the same rank;
- `aligned_crossfit`: a leading score subspace fitted on independent calibration tangent directions.

The basis vectors are converted to fixed classical outcome functions. Physical, random and aligned metrics can therefore be formed from the same computational-basis probability/Jacobian record; only the retained orientation changes.

The helper

```python
solve_controlled_direction(...)
```

adds controls needed for fair optimizer comparisons:

- `metric_normalization="trace"` or `"maxeig"` to remove a pure metric-scale confound;
- `damping_mode="absolute"`, `"mean_eig"`, or `"maxeig"`;
- Euclidean direction clipping;
- an accessible-metric trust radius.

## AQNG v2 benchmark suite

The published v1 experiment remains unchanged in `experiments/paper_classification.py`. The follow-up controls live in:

`experiments/aqng_v2_benchmark.py`

Install experiment extras:

```bash
pip install -e ".[experiments,test]"
```

### 1. Same-rank physical / random / aligned AQNG

```bash
python experiments/aqng_v2_benchmark.py \
  --dataset iris01 \
  --seed 0 \
  --suite orientation \
  --metric-normalization trace \
  --max-metric-step 0.25 \
  --output-dir results/aqng_v2/iris_seed0
```

The script fits the aligned readout from label-free calibration inputs and one tangent set, freezes it, and evaluates held-out tangent retention with a separate tangent set. `orientation_diags_*.csv` also computes the three AQNG metrics from one shared initial probability/Jacobian record.

### 2. Metric scale, damping, learning-rate, and trust-radius controls

Per-method tuning can be specified without changing the global protocol:

```bash
python experiments/aqng_v2_benchmark.py \
  --dataset wine01 --seed 0 --suite full \
  --metric-normalization trace \
  --damping-mode absolute \
  --lr-override AQNG-aligned=0.02 \
  --lam-override Full-QNG=0.003 \
  --max-direction-norm 8 \
  --max-metric-step 0.25 \
  --output-dir results/aqng_v2/wine_seed0
```

For a strict ablation, repeat with `--metric-normalization none` and/or a disabled trust radius using a sufficiently large bound.

### 3. Optimizer baselines

`--suite full` runs:

- `AQNG-physical`;
- `AQNG-random`;
- `AQNG-aligned`;
- `AQNG-Z0` (only the actual scalar task head as accessible geometry);
- `Full-QNG`;
- `Block-QNG`;
- `SGD`;
- `Adam`.

All metric methods share the same metric minibatch and refresh schedule. Full QNG and block QNG use the same QFIM convention (`4 *` the Fubini-Study metric).

### 4. End-to-end finite-shot AQNG

```bash
python experiments/aqng_v2_benchmark.py \
  --dataset digits01 \
  --seed 0 \
  --suite full \
  --shots 10000 \
  --cov-lam 1e-3 \
  --finite-shot-calibration \
  --metric-normalization trace \
  --max-metric-step 0.25 \
  --output-dir results/aqng_v2/digits_shots10k_seed0
```

With `--shots`, AQNG uses finite-shot computational-basis probabilities and a PennyLane parameter-shift probability Jacobian; the supervised Z0 loss gradient is also finite-shot and parameter-shift differentiated. Terminal train/test metrics are evaluated analytically. `loss_shots_total`, `metric_shots_total`, and execution counters are recorded through PennyLane trackers when exposed by the device.

In finite-shot mode `Full-QNG` and `Block-QNG` are deliberately labeled analytic oracle metric baselines; they are not presented as shot-matched hardware costs. Use SGD/Adam as genuinely finite-shot non-metric baselines.

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

## Tests

The new pure-NumPy readout/control tests check:

- identical rank for physical/random/aligned readouts;
- recovery of a held-out target score subspace by cross-fitted alignment;
- centered/whitened reference features;
- invariance of `J^T Sigma^+ J` under invertible coordinates inside one feature span;
- trace normalization and trust-radius enforcement;
- computational-basis sample indexing.

Run:

```bash
pytest -q
```

## Colab example

Use:

`examples/AQNG_Efficient_vs_PennyLane_QNG_Iris_Colab.ipynb`

The current notebook uses `metric_batch=2`, `metric_every=2`, adaptive refresh and trust-region safeguards, and logs all related diagnostics.
