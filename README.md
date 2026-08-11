# Accessible Quantum Natural Gradient (AQNG)

Accessible Quantum Natural Gradient (AQNG) is a measurement-aware natural-gradient method for variational quantum models. Instead of preconditioning with the full quantum Fisher information matrix, AQNG uses the Fisher geometry that is actually visible through a chosen commuting measurement/readout interface.

For retained readout features with expectation Jacobian `J` and covariance `Sigma`, the accessible metric is

\[
G_{\mathrm{acc}} = J^\mathsf{T}\Sigma^{+}J.
\]

For jointly measurable retained features this admits the exact score-space identity

\[
G_{\mathrm{acc}} = S^\mathsf{T} P S,
\]

where `S` is the probability-weighted measurement-score matrix and `P` is the orthogonal projector onto the retained centered readout span. AQNG is therefore a metric induced by a specified measurement interface, not merely a numerical low-rank approximation to the QFIM.

## Installation

Target environment: Python 3.10+ and PennyLane 0.45.x.

```bash
pip install -e ".[test]"
```

## Reusable optimizer API

The public interface is `AQNGOptimizer`:

```python
from aqng import AQNGOptimizer

optimizer = AQNGOptimizer(
    stepsize=0.06,
    readout="aligned",          # "physical" | "random" | "aligned"
    probability_fn=probability_fn,
    lam=3e-3,
    cov_lam=1e-3,
    metric_every=2,
    max_direction_norm=8.0,
    max_metric_step=0.25,
    solver="auto",
    readout_order=1,
    seed=0,
)

# Label-free calibration of the measurement readout.
optimizer.calibrate(
    params,
    calibration_inputs,
    n_qubits=n_qubits,
    n_directions=64,
)

for x_batch, y_batch, metric_batch in training_batches:
    params = optimizer.step(
        objective,
        params,
        x_batch,
        y_batch,
        metric_args=(metric_batch,),
    )
```

`probability_fn(params, ...)` must return computational-basis probabilities with shape `(D,)` or `(B, D)`. `calibrate(...)` differentiates that callable with respect to the trainable parameter array, builds normalized score tangents, fits equal-rank physical/random/aligned readouts, and binds the selected readout automatically. For `readout="aligned"`, calibration data should be independent of the supervised minibatch used for optimization and evaluation.

The optimizer exposes the main numerical controls directly in its constructor: learning rate, readout strategy, damping, covariance regularization, metric refresh cadence, adaptive refresh, direction-growth trigger, Euclidean direction cap, accessible-metric trust radius, solver selection, pseudoinverse tolerance, covariance PSD projection, reduction mode, seed, and readout order.

The supervised objective and metric may use different minibatches through `metric_args` and `metric_kwargs`. Advanced users can still bind custom differentiable feature/covariance functions directly.

A runnable example is provided in `examples/aqng_optimizer_quickstart.py`.

## Current research repository

This repository also contains the complete paper-scale result package and manuscript source.

- **Current manuscript:** `paper/manuscript/`
- **Complete paper-scale results:** `results/paper/paper_scale_v2/`
- **Primary benchmark runner:** `experiments/aqng_v2_benchmark.py`
- **Validation / support model:** `aqng_validation.py`
- **Validated numerical core:** `aqng_pennylane.py`, `aqng_efficient.py`, `aqng_readouts.py`

## Main experimental results

The study contains **470 experiment cells and 3160 terminal optimizer rows**, followed by a separately identified **15-cell / 30-row SGD–Adam large/deep follow-up**.

The primary matrix combines four binary datasets, four circuit families, 20 paired seeds, and eight optimizers. Auxiliary blocks study qubit scaling, depth scaling, readout order, and finite-shot behavior.

Across the 240 generic primary dataset–architecture cases (excluding the structured `U(1)` negative control), pooled descriptive means are:

| Method | Mean test loss | Mean test accuracy |
|---|---:|---:|
| AQNG-aligned | **0.594494** | **0.7947** |
| AQNG-random | 0.595566 | 0.7914 |
| SGD | 0.608363 | 0.7858 |
| Adam | 0.620272 | 0.7731 |
| AQNG-physical | 0.622907 | 0.7878 |
| AQNG-Z0 | 0.635241 | 0.7828 |
| Full-QNG | 0.656169 | 0.7689 |
| Block-QNG | 0.678055 | 0.7606 |

These pooled values are descriptive rather than a claim that one optimizer is universally superior. Architecture dependence remains substantial.

In the matched large/deep SU(2)-Haar follow-up, aligned AQNG beats SGD in all 15 paired cases and remains competitive with Adam. The Adam difference is not statistically resolved with five seeds per setting. The follow-up is reported explicitly as post-hoc because it was specified after inspection of the main campaign.

The geometry diagnostics are equally important:

- SU(2)-Haar scaling shows strong orientation above the random-rank baseline, while optimization gains remain nonmonotonic.
- In the number-conserving `U(1)` control, measurement accessibility stays near complete while the supervised gradient collapses by orders of magnitude. Accessibility is therefore not equivalent to trainability.
- Increasing the diagonal readout from weight one to weight two can increase retained tangent mass while worsening optimization because the accessible metric becomes severely ill-conditioned.
- Finite-shot experiments support the measurement-accessible construction, but Full-QNG uses an analytic metric oracle in those runs and is not a shot-matched hardware-cost baseline.

## Result package

`results/paper/paper_scale_v2/` contains the raw terminal results, calibration/orientation diagnostics, complete compressed training curves, metric diagnostics, corrected summaries, finite-shot/resource summaries, the SGD/Adam follow-up, checksums, and provenance metadata.

## AQNG readout controls

The three equal-rank readout strategies are:

- `physical` — low-weight diagonal Walsh / Pauli-Z readout span;
- `random` — Haar-random centered score subspace of the same rank;
- `aligned` — leading score subspace fitted from independent calibration tangents and then frozen.

Holding rank fixed isolates readout orientation from a trivial dimension advantage.

AQNG also supports metric trace or max-eigenvalue normalization in the experiment layer, several damping conventions, Euclidean direction clipping, an accessible-metric trust radius, primal/dual/SVD solves, and metric caching.

## Running the benchmark

A compact analytic run:

```bash
python experiments/aqng_v2_benchmark.py \
  --dataset iris01 \
  --seed 0 \
  --suite full \
  --metric-normalization trace \
  --max-direction-norm 8 \
  --max-metric-step 0.25 \
  --output-dir results/local/iris_seed0
```

The full suite contains `AQNG-physical`, `AQNG-random`, `AQNG-aligned`, `AQNG-Z0`, `Full-QNG`, `Block-QNG`, `SGD`, and `Adam`.

## Tests

```bash
pytest -q
```

The tests cover solver safeguards, readout-rank matching, automatic score calibration, public readout selection, separate objective/metric data, score-space whitening and invariance, trust-radius controls, sample indexing, and fixed-support validation behavior.

## Manuscript

The manuscript is in `paper/manuscript/` and is built with REVTeX.

The immediate prior works are:

1. Marwan Ait Haddou and Mohamed Bennai, *Sculpting Quantum Landscapes: Fubini–Study Metric Conditioning for Geometry Aware Learning in Parameterized Quantum Circuits*, arXiv:2506.21940.
2. Marwan Ait Haddou, *Readout-Rank Laws for Isotropic Quantum Tangents*, arXiv:2608.07628.
3. Marwan Ait Haddou, *Measurement-Accessible Quantum Tangent Geometry: Rank Baselines and Spectral Orientation*, currently cited as an unpublished manuscript until a permanent identifier is available.

## Scope and claims

AQNG should not be read as a universal replacement for Adam or as a solution to barren plateaus. The experiments support a narrower statement: when useful task-gradient signal survives, measurement-accessible geometry can provide effective and interpretable preconditioning; when the task gradient itself collapses, the accessible metric cannot create missing signal.

## Repository layout

```text
.
├── aqng/                      # public reusable package API
│   ├── __init__.py
│   ├── optimizer.py           # validated facade / readout binding
│   ├── standalone.py          # automatic calibration + public step API
│   └── calibration.py         # score-tangent calibration helper
├── aqng_pennylane.py          # reference AQNG metric implementation
├── aqng_efficient.py          # cached / safeguarded numerical core
├── aqng_readouts.py           # rank-matched readout geometry
├── aqng_validation.py         # fixed-support finite-shot utilities
├── examples/
├── experiments/
├── paper/manuscript/
├── results/paper/paper_scale_v2/
└── tests/
```

## License / citation

Please cite the manuscript and the relevant prior work when using the accessible tangent-geometry or rank-matched readout constructions.
