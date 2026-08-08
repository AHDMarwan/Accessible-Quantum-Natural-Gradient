# Paper classification benchmark protocol

This benchmark is the supervised extension of the tangent-accessibility experiments.

## Circuit families

The trainable ansatz families follow the papers:

1. `ryrz_cz`: RY-RZ single-qubit layers followed by a nearest-neighbour CZ line.
2. `su2_cnot`: RX-RY-RZ single-qubit SU(2) layers followed by a nearest-neighbour CNOT line.
3. `su2_haar`: the same SU(2) layers followed by fixed Haar-random two-qubit brickwork gates. Haar gates are fixed by the paired seed and are not trainable.
4. `u1_rzxy`: half-filled U(1)-conserving RZ layers plus parameterized XY brickwork.

The supervised task requires an input encoding not present in the tangent-only experiments. Generic families receive one RY input layer. The U(1) family receives a number-conserving RZ input layer after a fixed number-conserving XY mixer, so the half-filled symmetry sector is preserved exactly.

The default classification depth is deliberately smaller than the deep-circuit `d=6n` tangent experiments. Claims should therefore say that the benchmark reuses the **same circuit families/symmetry classes**, not the identical deep-circuit ensemble.

## Optimizer mathematics (frozen)

AQNG uses all diagonal Pauli-Z strings through weight 2 and

`G_acc = mean_b J_b^T Sigma_b^+ J_b`.

Full-QNG uses PennyLane's exact pure-state Fubini-Study metric multiplied by 4 to obtain QFIM units.

Both methods use the same loss minibatches, metric minibatches, metric refresh schedule, damping and fixed learning rate:

`(G + lambda I) d = grad L`, then `theta <- theta - lr d`.

No implementation optimization is allowed to alter these equations.

## Default confirmatory protocol

- datasets: `iris01`, `breast_cancer`, `wine01`, `digits01`
- qubits: 6
- ansatz layers: 3
- samples/dataset: 80, followed by a 75/25 stratified split
- paired seeds: 20
- steps: 20
- fixed LR: 0.03
- damping: 1e-3
- loss batch: 8
- metric batch: 4
- metric refresh: every 2 steps
- readout: all Z strings through weight 2

This is 4 datasets x 4 circuit families x 2 optimizers x 20 seeds = 640 optimizer runs.

## Implementation-only optimizations

- `lightning.qubit` for loss and AQNG QNodes with adjoint VJPs.
- Native input broadcasting instead of Python sample loops for prediction/features.
- Exact feature covariance assembled from broadcast probabilities with vectorized contractions.
- Cached QNodes, Haar gates and metric transforms.
- Full-QNG remains `qml.adjoint_metric_tensor` on `default.qubit` because PennyLane 0.45 restricts that transform to this device.
- Identical metric caching (`metric_every`) for both AQNG and Full-QNG.
- GitHub Actions parallelizes independent dataset/seed jobs; parallelism does not change any within-run calculation.

## Correctness gate

Before the 80 dataset/seed jobs start, the workflow checks all four families for:

- native broadcasting = scalar execution,
- symmetric PSD AQNG and QFIM matrices,
- `G_acc <= G_Q` numerically on the same batch,
- exact half-filling support preservation for the U(1) circuit.

If any check fails, the full experiment does not start.

## Statistics and immutability

The aggregate job refuses partial experiments, checks all expected seeds and duplicate keys, and reports per dataset x circuit family:

- paired AQNG minus Full-QNG test-loss difference,
- 95% paired bootstrap CI,
- AQNG win rate,
- paired accuracy difference,
- Wilcoxon signed-rank p-value,
- Holm correction across the 16 dataset-family comparisons,
- Full-QNG/AQNG runtime speedup,
- initial direction cosine and accessible/full metric trace ratio.

A completed confirmatory result bundle is written to `results/paper/<result_tag>/`. Existing result tags are never overwritten.
