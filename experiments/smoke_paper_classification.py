"""Fast correctness gate for the paper classification benchmark."""
import numpy as np
from pennylane import numpy as qnp
from paper_classification import CircuitBundle, Config, FAMILIES, make_data, parameter_count, solve_direction, spectral, tonp

def maxeig(A_minus_B):
    D=0.5*(A_minus_B+A_minus_B.T)
    return float(np.linalg.eigvalsh(D)[-1])

def main():
    cfg=Config(dataset="iris01",seed=0,n_qubits=6,n_layers=3,n_samples=40,steps=2,lr=0.03,lam=1e-3,loss_batch=4,metric_batch=2,metric_every=1)
    data=make_data(cfg.dataset,cfg.n_samples,cfg.n_qubits)
    data2=make_data(cfg.dataset,cfg.n_samples,cfg.n_qubits)
    if not np.array_equal(data["X_train"],data2["X_train"]): raise AssertionError("fixed data split invariant failed")
    Xb=data["X_train"][:2]
    print("parameter counts:",{f:parameter_count(f,cfg.n_qubits,cfg.n_layers) for f in FAMILIES})
    for family in FAMILIES:
        b=CircuitBundle(family,cfg); rng=np.random.default_rng(444)
        theta=qnp.array(rng.normal(0,0.1,size=b.p),requires_grad=True)
        vec=tonp(b.pred(theta,Xb)).reshape(-1); loop=np.array([float(b.pred(theta,x)) for x in Xb])
        if not np.allclose(vec,loop,atol=1e-10,rtol=1e-10): raise AssertionError(f"broadcast mismatch for {family}")
        Ga,probes=b.aqng_metric(theta,Xb,return_probes=True); Gq=b.full_qng_metric(theta,Xb)
        if not np.allclose(Ga,probes["le2"],atol=1e-12,rtol=1e-12): raise AssertionError(f"le2 probe differs from main AQNG: {family}")
        for name,G in [("A1",probes["A1"]),("local",probes["local"]),("le2",probes["le2"]),("QFIM",Gq)]:
            if not np.allclose(G,G.T,atol=1e-9): raise AssertionError(f"{name} non-symmetric: {family}")
            if np.linalg.eigvalsh(G).min() < -1e-7: raise AssertionError(f"{name} not PSD: {family}")
        checks={"A1<=local":maxeig(probes["A1"]-probes["local"]),"local<=le2":maxeig(probes["local"]-probes["le2"]),"le2<=QFIM":maxeig(probes["le2"]-Gq)}
        scale=max(1.0,float(np.linalg.eigvalsh(Gq)[-1]))
        for label,val in checks.items():
            if val > 2e-6*scale: raise AssertionError(f"{label} failed for {family}: {val}")
        g=np.linspace(-0.2,0.2,b.p); qdir=solve_direction(Gq,g,cfg.lam)
        for name,G in probes.items():
            d=solve_direction(G,g,cfg.lam)
            if not np.all(np.isfinite(d)): raise AssertionError(f"nonfinite {name} direction: {family}")
        if not np.all(np.isfinite(qdir)): raise AssertionError(f"nonfinite QNG direction: {family}")
        if family=="u1_rzxy":
            probs=tonp(b.probs(tonp(theta),Xb[0])).reshape(-1); outside=0.0
            for z,p in enumerate(probs):
                if int(z).bit_count()!=cfg.n_qubits//2: outside+=float(p)
            if outside>1e-10: raise AssertionError(f"U(1) support leakage: {outside}")
        print(family,"broadcast=ok",f"A1<=local={checks['A1<=local']:.3e}",f"local<=le2={checks['local<=le2']:.3e}",f"le2<=QFIM={checks['le2<=QFIM']:.3e}","ranks=",{k:spectral(v,cfg.lam,cfg.rcond)["metric_rank"] for k,v in {**probes,"QFIM":Gq}.items()})
    print("SMOKE TEST PASSED")

if __name__=="__main__": main()
