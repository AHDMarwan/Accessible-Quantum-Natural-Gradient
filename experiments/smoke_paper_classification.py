"""Fast correctness gate for the paper classification benchmark."""
import numpy as np
from pennylane import numpy as qnp
from paper_classification import CircuitBundle, Config, FAMILIES, make_data, parameter_count, tonp


def main():
    cfg = Config(dataset="iris01", seed=0, n_qubits=6, n_layers=3,
                 n_samples=40, steps=2, lr=0.03, lam=1e-3,
                 loss_batch=4, metric_batch=2, metric_every=1)
    data = make_data(cfg.dataset, cfg.seed, cfg.n_samples, cfg.n_qubits)
    Xb = data["X_train"][:2]
    print("parameter counts:", {f: parameter_count(f,cfg.n_qubits,cfg.n_layers) for f in FAMILIES})
    for family in FAMILIES:
        b = CircuitBundle(family,cfg)
        rng = np.random.default_rng(444)
        theta = qnp.array(rng.normal(0,0.1,size=b.p), requires_grad=True)
        vec = tonp(b.pred(theta,Xb)).reshape(-1)
        loop = np.array([float(b.pred(theta,x)) for x in Xb])
        if not np.allclose(vec,loop,atol=1e-10,rtol=1e-10):
            raise AssertionError(f"broadcast mismatch for {family}")
        Ga = b.aqng_metric(theta,Xb); Gq = b.full_qng_metric(theta,Xb)
        if not np.allclose(Ga,Ga.T,atol=1e-9): raise AssertionError(f"AQNG non-symmetric: {family}")
        if not np.allclose(Gq,Gq.T,atol=1e-9): raise AssertionError(f"QFIM non-symmetric: {family}")
        if np.linalg.eigvalsh(Ga).min() < -1e-7: raise AssertionError(f"AQNG not PSD: {family}")
        if np.linalg.eigvalsh(Gq).min() < -1e-7: raise AssertionError(f"QFIM not PSD: {family}")
        diff = 0.5*((Ga-Gq)+(Ga-Gq).T)
        max_loewner = float(np.linalg.eigvalsh(diff)[-1])
        scale = max(1.0,float(np.linalg.eigvalsh(Gq)[-1]))
        if max_loewner > 2e-6*scale:
            raise AssertionError(f"Gacc <= Gq failed for {family}: {max_loewner}")
        if family == "u1_rzxy":
            probs = tonp(b.probs(tonp(theta),Xb[0])).reshape(-1); outside = 0.0
            for z,p in enumerate(probs):
                if int(z).bit_count() != cfg.n_qubits//2: outside += float(p)
            if outside > 1e-10: raise AssertionError(f"U(1) support leakage: {outside}")
        print(family,"broadcast=ok",f"maxeig(Gacc-Gq)={max_loewner:.3e}",f"traces=({np.trace(Ga):.6g},{np.trace(Gq):.6g})")
    print("SMOKE TEST PASSED")


if __name__ == "__main__": main()
