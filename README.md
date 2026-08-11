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

## Current status

This repository contains the current AQNG implementation, the complete frozen paper-scale result package, and the new manuscript source.

- **Current manuscript:** `paper/manuscript/`
- **Complete paper-scale results:** `results/paper/paper_scale_v2/`
- **Primary benchmark runner:** `experiments/aqng_v2_benchmark.py`
- **Validation / support model:** `aqng_validation.py`
- **PennyLane implementation:** `aqng_pennylane.py`, `aqng_efficient.py`, `aqng_readouts.py`

The previous AQNG manuscript and the earlier `paper_main_v1` result package are intentionally not part of the canonical `main` tree.

## Main experimental results

The frozen campaign contains **470 experiment cells and 3160 terminal optimizer rows**, followed by a separately identified **15-cell / 30-row SGD–Adam large/deep follow-up**.

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

The central empirical message is:

> Accessible geometry is a controllable and diagnostically interpretable optimization resource, but task utility depends jointly on readout orientation, metric conditioning, and surviving task-gradient signal—not retained information alone.

## Result package

`results/paper/paper_scale_v2/` is the durable paper-facing package. It contains:

- `raw/results_all.csv` — all 3160 terminal rows;
- `raw/calibration_all.csv` — all 470 calibration rows;
- `raw/orientation_all.csv` — orientation diagnostics;
- `raw/curves_all.csv.gz` — complete training curves;
- `raw/metric_diags_all.csv.gz` — complete metric diagnostics;
- `summaries/` — corrected primary, scaling, depth, readout-order, finite-shot, geometry, resource and paired-test summaries;
- `followup/` — the complete SGD/Adam large/deep follow-up and merged matched comparisons;
- `report.json` — provenance and completeness metadata;
- `SHA256SUMS` — checksums for the committed result package.

The source paper-scale workflow completed successfully. A reporting-tag mismatch in the original aggregation step produced empty block-summary outputs, so the committed `summaries/` files were regenerated from the unchanged complete raw tables. The raw experiment outputs themselves were not rerun or altered for that reporting fix.

## Installation

Target environment: Python 3.10+ and PennyLane 0.45.x.

```bash
pip install "pennylane>=0.45,<0.46"
pip install -e ".[experiments,test]"
```

Core imports:

```python
from aqng_pennylane import AQNGOptimizer
from aqng_efficient import AQNGEfficientOptimizer
from aqng_readouts import fit_rank_matched_readouts, solve_controlled_direction
```

## AQNG readout controls

`aqng_readouts.py` constructs equal-rank measurement controls:

- `physical` — low-weight diagonal Walsh / Pauli-Z readout span;
- `random_rank` — Haar-random centered score subspace of the same rank;
- `aligned_crossfit` — leading score subspace fitted from independent calibration tangents and evaluated on held-out tangents.

The physical, random and aligned readouts have the same retained rank. This isolates orientation from the trivial dimension advantage.

AQNG also supports metric trace or max-eigenvalue normalization, several damping conventions, Euclidean direction clipping, an accessible-metric trust radius, primal/dual/SVD solves, and metric caching.

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

The full suite contains:

`AQNG-physical`, `AQNG-random`, `AQNG-aligned`, `AQNG-Z0`, `Full-QNG`, `Block-QNG`, `SGD`, and `Adam`.

For finite-shot runs, AQNG uses finite-shot computational-basis probabilities and parameter-shift probability Jacobians. The known support is fixed from circuit structure; the number-conserving family uses its fixed-Hamming-weight sector rather than inferring support from observed counts.

## Reproducing the paper-scale analysis

The experiment and analysis scripts used for the final campaign are retained under `experiments/`, and the corresponding GitHub Actions workflows are retained under `.github/workflows/`.

The committed result package already contains the complete outputs. Re-running the 470-cell matrix is not required to inspect or regenerate the paper-facing summaries.

## Tests

```bash
pytest -q
```

The tests cover solver safeguards, readout-rank matching, score-space whitening and invariance, trust-radius controls, sample indexing, and fixed-support validation behavior.

## Manuscript

The current manuscript is written from scratch in `paper/manuscript/`. It is built with REVTeX and cites the research lineage explicitly rather than treating measurement accessibility itself as a new idea.

The immediate prior works are:

1. Marwan Ait Haddou and Mohamed Bennai, *Sculpting Quantum Landscapes: Fubini–Study Metric Conditioning for Geometry Aware Learning in Parameterized Quantum Circuits*, arXiv:2506.21940.
2. Marwan Ait Haddou, *Readout-Rank Laws for Isotropic Quantum Tangents*, arXiv:2608.07628.
3. Marwan Ait Haddou, *Measurement-Accessible Quantum Tangent Geometry: Rank Baselines and Spectral Orientation*, currently cited as an unpublished manuscript until a permanent identifier is available.

The manuscript build workflow validates the LaTeX source on pull requests and builds the canonical PDF on `main`.

## Scope and claims

AQNG should not be read as a universal replacement for Adam or as a solution to barren plateaus. The experiments support a narrower statement: when useful task-gradient signal survives, measurement-accessible geometry can provide effective and interpretable preconditioning; when the task gradient itself collapses, the accessible metric cannot create missing signal.

## Repository layout

```text
.
├── aqng_pennylane.py          # reference AQNG optimizer
├── aqng_efficient.py          # cached / safeguarded implementation
├── aqng_readouts.py           # rank-matched readout geometry
├── aqng_validation.py         # fixed-support finite-shot utilities
├── experiments/               # frozen benchmark, validation and analysis scripts
├── paper/manuscript/          # current paper source and built PDF
├── results/paper/paper_scale_v2/
│   ├── raw/
│   ├── summaries/
│   └── followup/
└── tests/
```

## License / citation

Please cite the manuscript and the relevant prior work when using the accessible tangent-geometry or rank-matched readout constructions. The manuscript bibliography contains the current citation metadata, including the temporary unpublished citation for the spectral-geometry paper.