"""Minimal AQNG vs Full-QNG benchmark: generic VQC versus U(1)-conserving VQC.

Designed to be fetched directly from GitHub by a small Colab launcher notebook.
All experiment hyperparameters are passed from the notebook to run_benchmark().
"""

from functools import lru_cache
import time

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def run_benchmark(
    seeds=(0, 1),
    steps=8,
    fixed_lr=0.03,
    n_samples=40,
    loss_batch=8,
    metric_batch=2,
    metric_every=2,
    lam=1e-3,
    n_qubits=4,
    n_layers=2,
    show_plot=True,
    verbose=True,
):
    """Run the paired AQNG vs exact Full-QNG benchmark."""
    if n_qubits != 4:
        raise ValueError("This minimal benchmark currently fixes n_qubits=4.")

    P = n_layers * n_qubits * 2
    VQCS = ("generic", "u1")
    METHODS = ("AQNG", "Full-QNG")
    seeds = tuple(seeds)

    if verbose:
        print("PennyLane:", qml.__version__)
        print(
            f"runs={len(seeds)*len(VQCS)*len(METHODS)} | steps={steps} | "
            f"lr={fixed_lr} | metric_batch={metric_batch} | metric_every={metric_every}"
        )

    def make_data(seed):
        ds = load_iris()
        mask = ds.target < 2
        X = ds.data[mask].astype(float)
        y = ds.target[mask].astype(int)
        if n_samples > len(X):
            raise ValueError(f"n_samples={n_samples} exceeds available binary Iris samples={len(X)}.")
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, random_state=123, stratify=y
        )
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        Xtr = onp.pi * onp.tanh(Xtr / 2.0)
        Xte = onp.pi * onp.tanh(Xte / 2.0)
        return dict(
            X_train=Xtr, y_train=ytr, y_train_pm=2*ytr-1,
            X_test=Xte, y_test=yte, y_test_pm=2*yte-1,
        )

    def generic_vqc(theta, x):
        k = 0
        for w in range(n_qubits):
            qml.RY(x[..., w], wires=w)
        for _ in range(n_layers):
            for w in range(n_qubits):
                qml.RY(theta[k], wires=w); k += 1
                qml.RZ(theta[k], wires=w); k += 1
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w+1) % n_qubits])

    def u1_vqc(theta, x):
        k = 0
        qml.PauliX(0); qml.PauliX(1)
        for w in range(n_qubits):
            qml.IsingXY(onp.pi/4, wires=[w, (w+1) % n_qubits])
        for _ in range(n_layers):
            for w in range(n_qubits):
                qml.RZ(x[..., w], wires=w)
            for w in range(n_qubits):
                qml.RZ(theta[k], wires=w); k += 1
            for w in range(n_qubits):
                qml.IsingXY(theta[k], wires=[w, (w+1) % n_qubits]); k += 1

    def apply_vqc(name, theta, x):
        if name == "generic":
            generic_vqc(theta, x)
        elif name == "u1":
            u1_vqc(theta, x)
        else:
            raise ValueError(name)

    TERMS = (
        [(i,) for i in range(n_qubits)]
        + [(i,j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
    )

    def sign_matrix(terms):
        basis = onp.arange(2**n_qubits)
        bits = ((basis[:, None] >> onp.arange(n_qubits-1, -1, -1)) & 1)
        z = 1.0 - 2.0 * bits
        return onp.stack([onp.prod(z[:, list(t)], axis=1) for t in terms], axis=1)

    SIGN = sign_matrix(TERMS)
    SIGN_OUTER = onp.einsum("di,dj->dij", SIGN, SIGN)

    @lru_cache(maxsize=None)
    def bundle(vqc):
        try:
            dev_fast = qml.device("lightning.qubit", wires=n_qubits, shots=None, batch_obs=True)
        except Exception:
            dev_fast = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev_fast, interface="autograd", diff_method="adjoint", device_vjp=True, cache=False)
        def pred(theta, x):
            apply_vqc(vqc, theta, x)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_fast, interface="autograd", diff_method="adjoint", device_vjp=True, cache=False)
        def feature_qnode(theta, x):
            apply_vqc(vqc, theta, x)
            obs = []
            for t in TERMS:
                op = qml.PauliZ(t[0])
                for w in t[1:]:
                    op = op @ qml.PauliZ(w)
                obs.append(qml.expval(op))
            return tuple(obs)

        @qml.qnode(dev_fast, interface=None, diff_method=None, cache=False)
        def probs(theta, x):
            apply_vqc(vqc, theta, x)
            return qml.probs(wires=range(n_qubits))

        dev_qng = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev_qng, interface="autograd", cache=False)
        def qng_base(theta, x):
            apply_vqc(vqc, theta, x)
            return qml.expval(qml.PauliZ(0))

        return dict(
            pred=pred,
            feature_qnode=feature_qnode,
            probs=probs,
            full_metric_one=qml.adjoint_metric_tensor(qng_base),
        )

    def tonp(x):
        return onp.asarray(qml.math.toarray(x), dtype=float)

    def make_cost(pred, Xb, yb):
        Xb = onp.asarray(Xb); yb = onp.asarray(yb)
        def cost(theta):
            yhat = pred(theta, Xb)
            return qml.math.mean((yhat-yb)**2)
        return cost

    def aqng_metric(b, theta, Xm):
        Xm = onp.asarray(Xm)
        def features(theta):
            out = b["feature_qnode"](theta, Xm)
            return qml.math.stack(out, axis=-1)
        J = tonp(qml.jacobian(features, argnums=0)(theta)).reshape(len(Xm), len(TERMS), P)
        probs = tonp(b["probs"](tonp(theta), Xm))
        if probs.ndim == 1:
            probs = probs[None, :]
        mu = probs @ SIGN
        second = onp.einsum("bd,dij->bij", probs, SIGN_OUTER)
        Sigma = second - onp.einsum("bi,bj->bij", mu, mu)
        Sigma = 0.5 * (Sigma + onp.swapaxes(Sigma, -1, -2))
        evals, evecs = onp.linalg.eigh(Sigma)
        evals = onp.maximum(evals, 0.0)
        scale = onp.maximum(onp.max(evals, axis=1), 1.0)
        tol = 1e-8 * scale[:, None]
        inv = onp.where(evals > tol, 1.0/onp.maximum(evals, tol), 0.0)
        Sinv = onp.einsum("bik,bk,bjk->bij", evecs, inv, evecs)
        G = onp.einsum("bri,brs,bsj->ij", J, Sinv, J) / len(Xm)
        return 0.5 * (G + G.T)

    def full_qng_metric(b, theta, Xm):
        G = onp.zeros((P, P), dtype=float)
        one = b["full_metric_one"]
        for x in Xm:
            G += tonp(one(theta, x)).reshape(P, P)
        return 4.0 * G / len(Xm)

    def solve_direction(G, g):
        A = 0.5*(G+G.T) + lam*onp.eye(P)
        try:
            return onp.linalg.solve(A, g)
        except onp.linalg.LinAlgError:
            return onp.linalg.pinv(A, rcond=1e-8) @ g

    def init_theta(seed):
        rng = onp.random.default_rng(seed)
        return np.array(rng.normal(0.0, 0.15, size=P), requires_grad=True)

    def schedule(n, seed):
        rng = onp.random.default_rng(seed+1000)
        return [rng.choice(n, size=min(loss_batch, n), replace=False) for _ in range(steps)]

    def evaluate(pred, theta, X, y, ypm):
        yhat = tonp(pred(theta, X)).reshape(-1)
        return dict(
            loss=float(onp.mean((yhat-ypm)**2)),
            acc=float(onp.mean((yhat >= 0).astype(int) == y)),
        )

    def train(vqc, method, seed):
        data = make_data(seed)
        b = bundle(vqc)
        pred = b["pred"]
        theta = init_theta(seed)
        X = data["X_train"]
        ypm = data["y_train_pm"]
        batches = schedule(len(X), seed)
        G = None
        curve = []
        t0 = time.perf_counter()
        for step, ids in enumerate(batches, 1):
            mids = ids[:min(metric_batch, len(ids))]
            if G is None or (step-1) % metric_every == 0:
                G = aqng_metric(b, theta, X[mids]) if method == "AQNG" else full_qng_metric(b, theta, X[mids])
            cost = make_cost(pred, X[ids], ypm[ids])
            grad_fn = qml.grad(cost)
            g = tonp(grad_fn(theta)).reshape(-1)
            curve.append(float(grad_fn.forward))
            d = solve_direction(G, g)
            theta = np.array(tonp(theta) - fixed_lr*d, requires_grad=True)
        elapsed = time.perf_counter() - t0
        test = evaluate(pred, theta, data["X_test"], data["y_test"], data["y_test_pm"])
        return test, elapsed, curve

    rows, curves = [], {}
    jobs = [(vqc, method, seed) for vqc in VQCS for seed in seeds for method in METHODS]
    for vqc, method, seed in tqdm(jobs, desc="AQNG vs Full-QNG"):
        try:
            test, sec, curve = train(vqc, method, seed)
            rows.append(dict(
                vqc=vqc, method=method, seed=seed, lr=fixed_lr,
                test_loss=test["loss"], test_acc=test["acc"], seconds=sec,
            ))
            curves[(vqc, method, seed)] = curve
        except Exception as ex:
            print("FAILED:", vqc, method, seed, repr(ex))

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["vqc", "method"])
        .agg(
            loss_mean=("test_loss", "mean"),
            loss_std=("test_loss", "std"),
            acc_mean=("test_acc", "mean"),
            time_mean=("seconds", "mean"),
        )
        .reset_index()
        if len(results) else pd.DataFrame()
    )

    paired = pd.DataFrame()
    if len(results):
        wide = results.pivot_table(
            index=["vqc", "seed"], columns="method",
            values=["test_loss", "test_acc", "seconds"],
        )
        if (("test_loss", "AQNG") in wide.columns and
            ("test_loss", "Full-QNG") in wide.columns):
            paired = wide.reset_index()
            paired["delta_loss_AQNG_minus_QNG"] = (
                wide[("test_loss", "AQNG")].to_numpy()
                - wide[("test_loss", "Full-QNG")].to_numpy()
            )
            paired["speedup_QNG_over_AQNG"] = (
                wide[("seconds", "Full-QNG")].to_numpy()
                / wide[("seconds", "AQNG")].to_numpy()
            )

    if verbose:
        print("\nSummary")
        print(summary.to_string(index=False) if len(summary) else "No successful runs.")
        if len(paired):
            print("\nPaired comparison")
            cols = ["vqc", "seed", "delta_loss_AQNG_minus_QNG", "speedup_QNG_over_AQNG"]
            print(paired[cols].to_string(index=False))

    if show_plot and curves:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        first_seed = seeds[0]
        for (vqc, method, seed), curve in curves.items():
            if seed == first_seed:
                ax.plot(range(1, steps+1), curve, marker="o", label=f"{vqc}-{method}")
        ax.set_xlabel("step")
        ax.set_ylabel("training loss before update")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return results, summary, paired, curves
