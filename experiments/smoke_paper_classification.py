"""Fast correctness gate for the paper classification benchmark.

The theoretical Loewner hierarchy is exact. The implemented metrics contain
floating-point eigendecompositions and Moore-Penrose pseudoinverses, and the
half-filled U(1) feature covariance has exact linear dependencies. We therefore
validate the hierarchy with a small scale-aware numerical tolerance while
printing every raw residual. No optimizer calculation is modified by this gate.
"""
import numpy as np
from pennylane import numpy as qnp
from paper_classification import (
    CircuitBundle, Config, FAMILIES, make_data, parameter_count,
    schedule, solve_direction, spectral, tonp, train_method,
)

LOEWNER_RTOL = 1e-5
LOEWNER_ATOL = 1e-10


def maxeig(A_minus_B):
    D = 0.5 * (A_minus_B + A_minus_B.T)
    return float(np.linalg.eigvalsh(D)[-1])


def main():
    cfg = Config(
        dataset="iris01", seed=0, n_qubits=6, n_layers=3,
        n_samples=40, steps=2, lr=0.03, lam=1e-3,
        loss_batch=4, metric_batch=2, metric_every=1,
    )
    data = make_data(cfg.dataset, cfg.n_samples, cfg.n_qubits)
    data2 = make_data(cfg.dataset, cfg.n_samples, cfg.n_qubits)
    if not np.array_equal(data["X_train"], data2["X_train"]):
        raise AssertionError("fixed data split invariant failed")

    Xb = data["X_train"][:2]
    print("parameter counts:", {
        f: parameter_count(f, cfg.n_qubits, cfg.n_layers) for f in FAMILIES
    })
    print(
        f"Loewner tolerance: atol={LOEWNER_ATOL:.1e}, "
        f"rtol={LOEWNER_RTOL:.1e} (raw residuals are printed)"
    )

    for family in FAMILIES:
        b = CircuitBundle(family, cfg)
        rng = np.random.default_rng(444)
        theta = qnp.array(rng.normal(0, 0.1, size=b.p), requires_grad=True)

        vec = tonp(b.pred(theta, Xb)).reshape(-1)
        loop = np.array([float(b.pred(theta, x)) for x in Xb])
        if not np.allclose(vec, loop, atol=1e-10, rtol=1e-10):
            raise AssertionError(f"broadcast mismatch for {family}")

        outside = None
        if family == "u1_rzxy":
            raw_probs = tonp(b.probs(tonp(theta), Xb[0])).reshape(-1)
            outside = sum(
                float(p) for z, p in enumerate(raw_probs)
                if int(z).bit_count() != cfg.n_qubits // 2
            )
            if outside > 1e-10:
                raise AssertionError(f"U(1) support leakage: {outside}")

        Ga, probes = b.aqng_metric(theta, Xb, return_probes=True)
        Gq = b.full_qng_metric(theta, Xb)
        if not np.allclose(Ga, probes["le2"], atol=1e-12, rtol=1e-12):
            raise AssertionError(f"le2 probe differs from main AQNG: {family}")

        matrices = {
            "A1": probes["A1"],
            "local": probes["local"],
            "le2": probes["le2"],
            "QFIM": Gq,
        }
        for name, G in matrices.items():
            if not np.allclose(G, G.T, atol=1e-9):
                raise AssertionError(f"{name} non-symmetric: {family}")
            if np.linalg.eigvalsh(G).min() < -1e-7:
                raise AssertionError(f"{name} not PSD: {family}")

        checks = {
            "A1<=local": maxeig(probes["A1"] - probes["local"]),
            "local<=le2": maxeig(probes["local"] - probes["le2"]),
            "le2<=QFIM": maxeig(probes["le2"] - Gq),
        }
        scale = max(1.0, float(np.linalg.eigvalsh(Gq)[-1]))
        loewner_tol = LOEWNER_ATOL + LOEWNER_RTOL * scale
        for label, val in checks.items():
            if val > loewner_tol:
                raise AssertionError(
                    f"{label} failed for {family}: residual={val:.6e}, "
                    f"tol={loewner_tol:.6e}, scaled={val/scale:.6e}"
                )

        g = np.linspace(-0.2, 0.2, b.p)
        qdir = solve_direction(Gq, g, cfg.lam)
        for name, G in probes.items():
            d = solve_direction(G, g, cfg.lam)
            if not np.all(np.isfinite(d)):
                raise AssertionError(f"nonfinite {name} direction: {family}")
        if not np.all(np.isfinite(qdir)):
            raise AssertionError(f"nonfinite QNG direction: {family}")

        ranks = {k: spectral(v, cfg.lam, cfg.rcond)["metric_rank"]
                 for k, v in matrices.items()}
        extra = f" U1_outside={outside:.3e}" if outside is not None else ""
        print(
            family,
            "broadcast=ok",
            f"A1<=local={checks['A1<=local']:.3e}",
            f"local<=le2={checks['local<=le2']:.3e}",
            f"le2<=QFIM={checks['le2<=QFIM']:.3e}",
            f"tol={loewner_tol:.3e}",
            "ranks=", ranks,
            extra,
        )

    # End-to-end one-step optimizer integration check. This specifically
    # exercises minibatch input/target indexing, gradient evaluation, metric
    # construction, damped solve and parameter update through train_method.
    integ_bundle = CircuitBundle("ryrz_cz", cfg)
    integ_batches = schedule(len(data["X_train"]), cfg)[:1]
    integ = train_method(integ_bundle, "AQNG", cfg, data, integ_batches)
    r = integ["result"]
    if not np.isfinite(r["train_loss"]) or not np.isfinite(r["test_loss"]):
        raise AssertionError("end-to-end training integration produced nonfinite loss")
    if r["loss_examples_total"] != len(integ_batches[0]):
        raise AssertionError("end-to-end minibatch accounting mismatch")
    print("TRAINING INTEGRATION SMOKE PASSED")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
