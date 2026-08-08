# Paper classification benchmark protocol

This benchmark is the supervised extension of the tangent-accessibility experiments.

## Circuit families

The trainable ansatz families follow the papers:

1. `ryrz_cz`: RY-RZ single-qubit layers followed by a nearest-neighbour CZ line.
2. `su2_cnot`: RX-RY-RZ single-qubit SU(2) layers followed by a nearest-neighbour CNOT line.
3. `su2_haar`: the same SU(2) layers followed by fixed Haar-random two-qubit brickwork gates. Haar gates are fixed by the paired stochastic seed and are not trainable.
4. `u1_rzxy`: half-filled U(1)-conserving RZ layers plus parameterized XY brickwork.

The supervised task requires an input encoding not present in the tangent-only experiments. Generic families receive one RY input layer. The U(1) family receives a number-conserving RZ input layer after a fixed number-conserving XY mixer, so the half-filled symmetry sector is preserved exactly.

The default classification depth is deliberately smaller than the deep-circuit `d=6n` tangent experiments. Claims should therefore say that the benchmark reuses the **same circuit families/symmetry classes**, not the identical deep-circuit ensemble.

## Optimizer mathematics (frozen)

AQNG uses all diagonal Pauli-Z strings through weight 2,

`G_acc = mean_b J_b^T Sigma_b^+ J_b`.

Full-QNG uses PennyLane's exact pure-state Fubini-Study metric multiplied by 4 to obtain QFIM units.

Both methods use the same loss minibatches, metric minibatches, metric refresh schedule, damping and fixed learning rate:

`(G + lambda I) d = grad L`, then `theta <- theta - lr d`.

No implementation optimization is allowed to alter these equations.

## Fair-comparison guardrails

### 1. Fixed supervised task across stochastic seeds

The dataset subset and train/test split are frozen with fixed constants (`SUBSET_SEED=1729`, `SPLIT_SEED=314159`). The optimizer/circuit seed does **not** change the train/test split.

A paired stochastic seed changes only:
- parameter initialization,
- minibatch schedule,
- the fixed Haar brickwork instance for `su2_haar`.

Therefore the 20 paired seeds estimate stochastic training/circuit variability on one fixed supervised task per dataset. Dataset-split robustness must be reported separately and must not be mixed into the main confirmatory p-values.

Preprocessing is fitted on the training split only: StandardScaler followed, when needed, by train-fitted PCA to `n_qubits` dimensions. Datasets with fewer than `n_qubits` features are zero-padded.

### 2. Runtime claims are separated from logical workload

AQNG loss/feature QNodes run on `lightning.qubit`; exact Full-QNG remains `qml.adjoint_metric_tensor` on `default.qubit`. Therefore reported seconds are explicitly labelled **backend-specific implementation wall-clock** and are not interpreted as intrinsic algorithmic complexity.

Each run also records backend-independent logical workload fields:
- number of trainable parameters,
- state dimension,
- readout feature count,
- metric builds,
- total metric examples processed,
- total loss examples processed.

The aggregator refuses a paired AQNG/Full-QNG cell if matched logical workload counts differ.

### 3. Readout-span probes without extra training

The classification loss uses `<Z0>`, while the main AQNG metric uses all diagonal Z strings through weight 2. To make that modeling choice auditable, the first AQNG metric build also reports nested probes

- `A1 = {Z0}`,
- `local = {Zi}`,
- `le2 = {Zi} union {Zi Zj}`.

Crucially, these probes reuse the **same first AQNG Jacobian and covariance build**. They add only small classical linear-algebra contractions and do not introduce extra quantum evaluations or extra optimizer trajectories.

For every paired seed we record probe rank, trace/QFIM ratio, direction cosine to Full-QNG, and the raw numerical Loewner residuals for

`G_A1 <= G_local <= G_le2 <= G_Q`.

The theoretical hierarchy is exact. Numerically, however, these metrics contain eigendecompositions and Moore-Penrose pseudoinverses. The half-filled U(1) readout covariance has exact linear dependencies, so tiny positive residuals can appear from floating-point arithmetic even when the underlying nested-span identity is satisfied. The smoke test therefore uses a transparent scale-aware tolerance

`residual <= 1e-10 + 1e-5 * max(1, lambda_max(G_Q))`

and prints every raw residual. This tolerance affects **validation only**; it does not change `rcond`, the covariance pseudoinverse, any optimizer metric, or any update direction.

## Default confirmatory protocol

- datasets: `iris01`, `breast_cancer`, `wine01`, `digits01`
- qubits: 6
- ansatz layers: 3
- samples/dataset: 80, followed by one fixed 75/25 stratified split
- paired stochastic seeds: 20
- steps: 20
- fixed LR: 0.03
- damping: 1e-3
- loss batch: 8
- metric batch: 4
- metric refresh: every 2 steps
- main AQNG readout: all Z strings through weight 2

This is 4 datasets x 4 circuit families x 2 optimizers x 20 seeds = 640 optimizer runs.

The main claim is explicitly **under a common fixed-LR protocol**. LR robustness is a separate post-hoc robustness experiment, not a tuning step for the confirmatory run.

## Implementation-only optimizations

- `lightning.qubit` for loss and AQNG QNodes with adjoint VJPs.
- Native input broadcasting instead of Python sample loops for prediction/features.
- Exact feature covariance assembled from broadcast probabilities with vectorized contractions.
- A1/local/le2 probes reuse the first `J,Sigma` computation.
- Cached QNodes, fixed Haar gates and metric transforms within each job.
- Full-QNG remains exact `qml.adjoint_metric_tensor` on `default.qubit`.
- Identical metric caching (`metric_every`) for AQNG and Full-QNG.
- GitHub Actions parallelizes independent dataset/seed jobs; parallelism does not change any within-run calculation.

## Correctness gate

Before the 80 dataset/seed jobs start, the workflow checks all four families for:

- native broadcasting = scalar execution,
- symmetric PSD A1/local/le2/QFIM matrices,
- `A1 <= local <= le2 <= QFIM` within the documented numerical tolerance while retaining the raw residuals,
- main AQNG metric = the `le2` probe,
- finite damped directions,
- exact half-filling support preservation for the U(1) circuit.

During each actual job the code additionally asserts:
- paired AQNG/Full-QNG initial gradients are numerically identical,
- metric build counts match,
- metric-example counts match,
- loss-example counts match.

If any check fails, that job fails and the aggregate result cannot be committed.

## Statistics and immutability

The aggregate job refuses partial experiments, checks all expected seeds and duplicate keys, and reports per dataset x circuit family:

- paired AQNG minus Full-QNG test-loss difference,
- 95% paired bootstrap CI over stochastic seeds,
- AQNG win rate,
- paired accuracy difference,
- Wilcoxon signed-rank p-value,
- Holm correction across the 16 dataset-family comparisons,
- backend-specific implemented wall-clock speedup,
- A1/local/le2 rank, trace/QFIM ratio and direction-cosine summaries,
- resource-accounting table with matched logical workloads.

A completed confirmatory result bundle is written to `results/paper/<result_tag>/`. Existing result tags are never overwritten.
