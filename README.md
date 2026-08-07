# AQNG for PennyLane

Accessible Quantum Natural Gradient (AQNG) uses the Fisher geometry visible through a selected measurement/readout space,

\[
G_{\rm acc}=\frac1B\sum_{b=1}^B J_b^T\Sigma_b^{+}J_b.
\]

Target API: PennyLane `0.45.x`, using the `pennylane.numpy` / Autograd workflow.

## Install

```bash
pip install --upgrade "git+https://github.com/AHDMarwan/aqng.git"
```

## Corrected solver

The first prototype built the full `p x p` matrix `G_acc` and delegated the update to `qml.QNGOptimizer`. The current `AQNGOptimizer` instead factors

\[
G_{\rm acc}=A^T A
\]

and solves

\[
(G_{\rm acc}+\lambda I)d=\nabla L
\]

directly.

With `solver="auto"` and `lam > 0` it chooses the smaller exact system:

- **primal:** size `p` when `p <= B*r`;
- **dual (Woodbury):** size `B*r` when `B*r < p`.

For `lam=0` it uses an SVD Moore-Penrose pseudoinverse. The mini-batch dual dimension is `B*r`, not generally `r`.

## Minimal use

```python
from aqng_pennylane import AQNGOptimizer

aqng = AQNGOptimizer(
    stepsize=0.03,
    lam=1e-3,
    cov_lam=1e-3,
    rcond=1e-8,
    solver="auto",
)

theta, old_loss = aqng.step_and_cost(
    cost,
    theta,
    feature_fn=features,          # (r,) or (batch, r), differentiable
    covariance_fn=covariance,    # (r,r) or (batch,r,r), not differentiated
)

print(aqng.diagnostics)
```

`aqng.diagnostics` reports the metric rank/condition number, chosen solver, and actual linear-system dimension.

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

For a fair comparison keep the circuit, initialization, mini-batches, loss, learning-rate tuning protocol, and shot budget identical. Report convergence versus iterations **and** total circuit executions/shots.

## Explicit custom-metric compatibility

`make_aqng_metric_tensor_fn(...)` is retained for experiments that intentionally use PennyLane's `QNGOptimizer` with `G_acc` as a custom metric. That compatibility path materializes a `p x p` matrix and therefore does not provide the primal/dual solve advantage of `AQNGOptimizer`.
