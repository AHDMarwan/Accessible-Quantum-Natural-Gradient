"""Paper-scale AQNG vs Full-QNG supervised benchmark.

Circuit families mirror the tangent-accessibility papers:
1) RY-RZ + nearest-neighbour CZ line
2) SU(2)=RX-RY-RZ + nearest-neighbour CNOT line
3) SU(2)=RX-RY-RZ + fixed Haar-random two-qubit brickwork
4) half-filled U(1): RZ + parameterized IsingXY brickwork

Fairness guardrails:
- fixed dataset subset and fixed train/test split across stochastic seeds;
- paired initialization, minibatches, metric minibatches, refresh, damping, and LR;
- AQNG readout probes A1/local/Z<=2 are extracted from the SAME first
  Jacobian/covariance build, so probes do not change training;
- timing is explicitly backend-specific implementation wall-clock; matched
  logical workload counters are recorded separately.

Optimizer mathematics is unchanged:
G_acc = mean_b J_b^T Sigma_b^+ J_b
Full-QNG = exact QFIM = 4*g_FS
(G+lambda I)d = grad L
theta <- theta - lr*d
"""
from __future__ import annotations

import argparse, json, os, platform, time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

FAMILIES = ("ryrz_cz", "su2_cnot", "su2_haar", "u1_rzxy")
METHODS = ("AQNG", "Full-QNG")
DATASETS = ("iris01", "breast_cancer", "wine01", "digits01")
SUBSET_SEED = 1729
SPLIT_SEED = 314159

@dataclass(frozen=True)
class Config:
    dataset: str
    seed: int
    n_qubits: int = 6
    n_layers: int = 3
    n_samples: int = 80
    steps: int = 20
    lr: float = 0.03
    lam: float = 1e-3
    loss_batch: int = 8
    metric_batch: int = 4
    metric_every: int = 2
    rcond: float = 1e-8

def tonp(x):
    return onp.asarray(qml.math.toarray(x), dtype=float)

def _binary_dataset(name):
    if name == "iris01":
        ds = load_iris(); m = ds.target < 2
        return ds.data[m].astype(float), ds.target[m].astype(int)
    if name == "breast_cancer":
        ds = load_breast_cancer(); return ds.data.astype(float), ds.target.astype(int)
    if name == "wine01":
        ds = load_wine(); m = ds.target < 2
        return ds.data[m].astype(float), ds.target[m].astype(int)
    if name == "digits01":
        ds = load_digits(); m = ds.target < 2
        return ds.data[m].astype(float), ds.target[m].astype(int)
    raise ValueError(name)

@lru_cache(maxsize=None)
def make_data(name, n_samples, n_qubits):
    """Fixed dataset/subset/split; stochastic training seed is absent."""
    X, y = _binary_dataset(name)
    if n_samples > len(X): raise ValueError(f"n_samples={n_samples} exceeds {name} size={len(X)}")
    X, _, y, _ = train_test_split(X, y, train_size=n_samples, random_state=SUBSET_SEED, stratify=y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    if Xtr.shape[1] > n_qubits:
        pca = PCA(n_components=n_qubits, svd_solver="full")
        Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
    elif Xtr.shape[1] < n_qubits:
        pad = n_qubits - Xtr.shape[1]
        Xtr = onp.pad(Xtr, ((0,0),(0,pad))); Xte = onp.pad(Xte, ((0,0),(0,pad)))
    Xtr = onp.pi * onp.tanh(Xtr/2.0); Xte = onp.pi * onp.tanh(Xte/2.0)
    return dict(X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, y_train_pm=2*ytr-1, y_test_pm=2*yte-1)

def brickwork_pairs(n, layer): return [(w,w+1) for w in range(layer % 2, n-1, 2)]
def u1_xy_pairs(n): return ([(w,w+1) for w in range(0,n-1,2)] + [(w,w+1) for w in range(1,n-1,2)])

def parameter_count(family, n, layers):
    if family == "ryrz_cz": return layers*2*n
    if family in ("su2_cnot","su2_haar"): return layers*3*n
    if family == "u1_rzxy": return layers*(n + len(u1_xy_pairs(n)))
    raise ValueError(family)

def haar_unitary4(rng):
    z = (rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))) / onp.sqrt(2.0)
    q, r = onp.linalg.qr(z); d = onp.diag(r)
    ph = onp.ones_like(d, dtype=complex); nz = onp.abs(d) > 0
    ph[nz] = d[nz] / onp.abs(d[nz])
    return q * ph.conj()[None,:]

def make_haar_blocks(seed, n, layers):
    rng = onp.random.default_rng(91771 + int(seed))
    return {(layer,pair): haar_unitary4(rng) for layer in range(layers) for pair in brickwork_pairs(n,layer)}

def generic_input(x, n):
    for w in range(n): qml.RY(x[...,w], wires=w)

def apply_family(family, theta, x, n, layers, haar_blocks=None):
    k = 0
    if family == "ryrz_cz":
        generic_input(x,n)
        for _ in range(layers):
            for w in range(n):
                qml.RY(theta[k],wires=w); k+=1; qml.RZ(theta[k],wires=w); k+=1
            for w in range(n-1): qml.CZ(wires=[w,w+1])
        return
    if family == "su2_cnot":
        generic_input(x,n)
        for _ in range(layers):
            for w in range(n):
                qml.RX(theta[k],wires=w); k+=1; qml.RY(theta[k],wires=w); k+=1; qml.RZ(theta[k],wires=w); k+=1
            for w in range(n-1): qml.CNOT(wires=[w,w+1])
        return
    if family == "su2_haar":
        if haar_blocks is None: raise ValueError("su2_haar requires fixed Haar blocks")
        generic_input(x,n)
        for layer in range(layers):
            for w in range(n):
                qml.RX(theta[k],wires=w); k+=1; qml.RY(theta[k],wires=w); k+=1; qml.RZ(theta[k],wires=w); k+=1
            for pair in brickwork_pairs(n,layer): qml.QubitUnitary(haar_blocks[(layer,pair)], wires=list(pair))
        return
    if family == "u1_rzxy":
        for w in range(n//2): qml.PauliX(w)
        for a,b in u1_xy_pairs(n): qml.IsingXY(onp.pi/4.0,wires=[a,b])
        for w in range(n): qml.RZ(x[...,w],wires=w)
        pairs = u1_xy_pairs(n)
        for _ in range(layers):
            for w in range(n): qml.RZ(theta[k],wires=w); k+=1
            for a,b in pairs: qml.IsingXY(theta[k],wires=[a,b]); k+=1
        return
    raise ValueError(family)

def z_terms_le2(n): return ([(i,) for i in range(n)] + [(i,j) for i in range(n) for j in range(i+1,n)])

def sign_matrix(terms,n):
    basis=onp.arange(2**n); bits=((basis[:,None]>>onp.arange(n-1,-1,-1))&1); z=1.0-2.0*bits
    return onp.stack([onp.prod(z[:,list(t)],axis=1) for t in terms],axis=1)

def _psd_pinv_batch(S,rcond):
    vals,vecs=onp.linalg.eigh(0.5*(S+onp.swapaxes(S,-1,-2))); vals=onp.maximum(vals,0.0)
    scale=onp.maximum(onp.max(vals,axis=1),1.0); tol=rcond*scale[:,None]
    inv=onp.where(vals>tol,1.0/onp.maximum(vals,tol),0.0)
    return onp.einsum("bik,bk,bjk->bij",vecs,inv,vecs,optimize=True)

def _metric_from_subset(J,S,idx,rcond):
    idx=onp.asarray(idx,dtype=int); Js=J[:,idx,:]; Ss=S[:,idx,:][:,:,idx]
    Sinv=_psd_pinv_batch(Ss,rcond)
    G=onp.einsum("bri,brs,bsj->ij",Js,Sinv,Js,optimize=True)/len(J)
    return 0.5*(G+G.T)

class CircuitBundle:
    def __init__(self,family,cfg):
        self.family=family; self.cfg=cfg; self.p=parameter_count(family,cfg.n_qubits,cfg.n_layers)
        self.terms=z_terms_le2(cfg.n_qubits); self.sign=sign_matrix(self.terms,cfg.n_qubits)
        self.sign_outer=onp.einsum("di,dj->dij",self.sign,self.sign)
        self.haar_blocks=make_haar_blocks(cfg.seed,cfg.n_qubits,cfg.n_layers) if family=="su2_haar" else None
        self.fast_backend="lightning.qubit"; self.qng_backend="default.qubit"
        dev_fast=qml.device(self.fast_backend,wires=cfg.n_qubits,shots=None,batch_obs=True)
        def circuit(theta,x): apply_family(family,theta,x,cfg.n_qubits,cfg.n_layers,self.haar_blocks)
        @qml.qnode(dev_fast,interface="autograd",diff_method="adjoint",device_vjp=True,cache=False)
        def pred(theta,x): circuit(theta,x); return qml.expval(qml.PauliZ(0))
        @qml.qnode(dev_fast,interface="autograd",diff_method="adjoint",device_vjp=True,cache=False)
        def feature_qnode(theta,x):
            circuit(theta,x); obs=[]
            for term in self.terms:
                op=qml.PauliZ(term[0])
                for w in term[1:]: op=op@qml.PauliZ(w)
                obs.append(qml.expval(op))
            return tuple(obs)
        @qml.qnode(dev_fast,interface=None,diff_method=None,cache=False)
        def probs(theta,x): circuit(theta,x); return qml.probs(wires=range(cfg.n_qubits))
        dev_qng=qml.device(self.qng_backend,wires=cfg.n_qubits,shots=None)
        @qml.qnode(dev_qng,interface="autograd",cache=False)
        def qng_base(theta,x): circuit(theta,x); return qml.expval(qml.PauliZ(0))
        self.pred=pred; self.feature_qnode=feature_qnode; self.probs=probs
        self.full_metric_one=qml.adjoint_metric_tensor(qng_base)

    def _aqng_j_sigma(self,theta,Xm):
        Xm=onp.asarray(Xm); B=len(Xm); r=len(self.terms)
        def features(th): return qml.math.stack(self.feature_qnode(th,Xm),axis=-1)
        J=tonp(qml.jacobian(features,argnums=0)(theta)).reshape(B,r,self.p)
        probs=tonp(self.probs(tonp(theta),Xm)); probs=probs[None,:] if probs.ndim==1 else probs
        mu=probs@self.sign
        second=onp.einsum("bd,dij->bij",probs,self.sign_outer,optimize=True)
        S=second-onp.einsum("bi,bj->bij",mu,mu,optimize=True); S=0.5*(S+onp.swapaxes(S,-1,-2))
        return J,S

    def aqng_metric(self,theta,Xm,return_probes=False):
        J,S=self._aqng_j_sigma(theta,Xm); n=self.cfg.n_qubits
        idx={"A1":onp.array([0]),"local":onp.arange(n),"le2":onp.arange(len(self.terms))}
        Gle2=_metric_from_subset(J,S,idx["le2"],self.cfg.rcond)
        if not return_probes: return Gle2
        probes={name:(Gle2 if name=="le2" else _metric_from_subset(J,S,ii,self.cfg.rcond)) for name,ii in idx.items()}
        return Gle2,probes

    def full_qng_metric(self,theta,Xm):
        G=onp.zeros((self.p,self.p),dtype=float)
        for x in Xm: G += tonp(self.full_metric_one(theta,x)).reshape(self.p,self.p)
        G *= 4.0/float(len(Xm))
        return 0.5*(G+G.T)

def spectral(G,lam,rcond):
    G=0.5*(G+G.T); eig=onp.maximum(onp.linalg.eigvalsh(G),0.0); mx=float(eig[-1]) if eig.size else 0.0
    pos=eig[eig>rcond*max(1.0,mx)]; rank=len(pos)
    cond=float(pos[-1]/pos[0]) if len(pos)>1 else (1.0 if len(pos)==1 else onp.inf); reg=eig+lam
    return dict(metric_rank=int(rank),metric_trace=float(onp.trace(G)),metric_condition=cond,regularized_metric_condition=float(reg[-1]/reg[0]),metric_min_eig=float(eig[0]),metric_max_eig=mx)

def solve_direction(G,g,lam):
    A=0.5*(G+G.T)+lam*onp.eye(G.shape[0])
    try: return onp.linalg.solve(A,g)
    except onp.linalg.LinAlgError: return onp.linalg.pinv(A,rcond=1e-8)@g

def make_cost(pred,Xb,yb):
    Xb=onp.asarray(Xb); yb=onp.asarray(yb)
    def cost(theta): return qml.math.mean((pred(theta,Xb)-yb)**2)
    return cost

def init_theta(p,seed):
    rng=onp.random.default_rng(10003+int(seed)); return np.array(rng.normal(0,0.15,size=p),requires_grad=True)

def schedule(n,cfg):
    rng=onp.random.default_rng(20003+int(cfg.seed))
    return [rng.choice(n,size=min(cfg.loss_batch,n),replace=False) for _ in range(cfg.steps)]

def evaluate(pred,theta,X,y,ypm):
    yh=tonp(pred(theta,X)).reshape(-1)
    return dict(loss=float(onp.mean((yh-ypm)**2)),acc=float(onp.mean((yh>=0).astype(int)==y)))

def train_method(bundle,method,cfg,data,batches):
    theta=init_theta(bundle.p,cfg.seed); X=data["X_train"]; ypm=data["y_train_pm"]
    G=None; builds=metric_examples=loss_examples=0; mt=gt=st=0.0
    firstG=firstg=firstd=first_probes=None; mdiags=[]; curves=[]; ttot=time.perf_counter()
    for step,ids in enumerate(batches,1):
        mids=ids[:min(cfg.metric_batch,len(ids))]; refresh=G is None or ((step-1)%cfg.metric_every==0)
        if refresh:
            t=time.perf_counter()
            if method=="AQNG":
                if firstG is None: G,first_probes=bundle.aqng_metric(theta,X[mids],return_probes=True)
                else: G=bundle.aqng_metric(theta,X[mids])
            else: G=bundle.full_qng_metric(theta,X[mids])
            mt+=time.perf_counter()-t; builds+=1; metric_examples+=len(mids)
            d=spectral(G,cfg.lam,cfg.rcond); d.update(step=step,method=method); mdiags.append(d)
            if firstG is None: firstG=onp.array(G,copy=True)
        cost=make_cost(bundle.pred,X[ids],ypm); grad_fn=qml.grad(cost)
        t=time.perf_counter(); g=tonp(grad_fn(theta)).reshape(-1); gt+=time.perf_counter()-t; loss_examples+=len(ids); loss=float(grad_fn.forward)
        t=time.perf_counter(); direction=solve_direction(G,g,cfg.lam); st+=time.perf_counter()-t
        if firstg is None: firstg=onp.array(g,copy=True); firstd=onp.array(direction,copy=True)
        theta=np.array(tonp(theta)-cfg.lr*direction,requires_grad=True)
        curves.append(dict(step=step,loss_before=loss,gradient_norm=float(onp.linalg.norm(g)),direction_norm=float(onp.linalg.norm(direction)),parameter_step_norm=float(cfg.lr*onp.linalg.norm(direction)),metric_refreshed=int(refresh)))
    total=time.perf_counter()-ttot
    tr=evaluate(bundle.pred,theta,data["X_train"],data["y_train"],data["y_train_pm"])
    te=evaluate(bundle.pred,theta,data["X_test"],data["y_test"],data["y_test_pm"])
    md=pd.DataFrame(mdiags)
    result=dict(dataset=cfg.dataset,family=bundle.family,method=method,seed=cfg.seed,n_qubits=cfg.n_qubits,n_layers=cfg.n_layers,n_params=bundle.p,n_samples=cfg.n_samples,train_size=len(data["X_train"]),test_size=len(data["X_test"]),subset_seed=SUBSET_SEED,split_seed=SPLIT_SEED,steps=cfg.steps,lr=cfg.lr,lam=cfg.lam,loss_batch=cfg.loss_batch,metric_batch=cfg.metric_batch,metric_every=cfg.metric_every,readout="Z<=2",readout_feature_count=len(bundle.terms),state_dimension=2**cfg.n_qubits,loss_backend=bundle.fast_backend,metric_backend=(bundle.fast_backend if method=="AQNG" else bundle.qng_backend),timing_interpretation="backend_specific_implementation_wall_clock",train_loss=tr["loss"],train_acc=tr["acc"],test_loss=te["loss"],test_acc=te["acc"],total_seconds=total,metric_seconds=mt,gradient_seconds=gt,solve_seconds=st,metric_builds=builds,metric_examples_total=metric_examples,loss_examples_total=loss_examples,metric_rank_first=int(md.iloc[0].metric_rank),metric_trace_first=float(md.iloc[0].metric_trace),metric_condition_first=float(md.iloc[0].metric_condition),regularized_metric_condition_first=float(md.iloc[0].regularized_metric_condition),metric_rank_last=int(md.iloc[-1].metric_rank),metric_trace_last=float(md.iloc[-1].metric_trace))
    return dict(result=result,curves=curves,metric_diags=mdiags,first_metric=firstG,first_grad=firstg,first_direction=firstd,first_probes=first_probes)

def cosine(a,b):
    na=float(onp.linalg.norm(a)); nb=float(onp.linalg.norm(b)); return onp.nan if na==0 or nb==0 else float(onp.dot(a,b)/(na*nb))

def maxeig_diff(A,B):
    D=0.5*((A-B)+(A-B).T); return float(onp.linalg.eigvalsh(D)[-1])

def run_job(cfg,output_dir):
    if cfg.n_qubits%2: raise ValueError("n_qubits must be even for half-filled U(1)")
    output_dir.mkdir(parents=True,exist_ok=True)
    data=make_data(cfg.dataset,cfg.n_samples,cfg.n_qubits); batches=schedule(len(data["X_train"]),cfg)
    rows=[]; curves=[]; mdiags=[]; pairs=[]
    for family in tqdm(FAMILIES,desc=f"{cfg.dataset} seed={cfg.seed}"):
        b=CircuitBundle(family,cfg); outs={}
        for method in METHODS:
            out=train_method(b,method,cfg,data,batches); outs[method]=out; rows.append(out["result"])
            curves += [dict(dataset=cfg.dataset,family=family,method=method,seed=cfg.seed,**r) for r in out["curves"]]
            mdiags += [dict(dataset=cfg.dataset,family=family,seed=cfg.seed,**r) for r in out["metric_diags"]]
        a,q=outs["AQNG"],outs["Full-QNG"]; Ga,Gq=a["first_metric"],q["first_metric"]
        grad_diff=float(onp.max(onp.abs(a["first_grad"]-q["first_grad"])))
        if grad_diff>1e-9: raise RuntimeError(f"paired initial gradients differ for {cfg.dataset}/{family}/seed={cfg.seed}: {grad_diff}")
        ra,rq=a["result"],q["result"]
        for field in ("metric_builds","metric_examples_total","loss_examples_total"):
            if int(ra[field])!=int(rq[field]): raise RuntimeError(f"paired logical workload mismatch {field}: {ra[field]} vs {rq[field]}")
        probes=a["first_probes"]
        if probes is None: raise RuntimeError("AQNG first-step probes were not produced")
        prow=dict(dataset=cfg.dataset,family=family,seed=cfg.seed,initial_gradient_cosine=cosine(a["first_grad"],q["first_grad"]),initial_gradient_max_abs_diff=grad_diff,main_direction_cosine=cosine(a["first_direction"],q["first_direction"]),metric_examples_total_AQNG=int(ra["metric_examples_total"]),metric_examples_total_FullQNG=int(rq["metric_examples_total"]),loss_examples_total_AQNG=int(ra["loss_examples_total"]),loss_examples_total_FullQNG=int(rq["loss_examples_total"]),wall_clock_note="backend_specific_not_intrinsic_cost")
        qdir=q["first_direction"]; qtrace=float(onp.trace(Gq))
        for label,Gp in probes.items():
            ps=spectral(Gp,cfg.lam,cfg.rcond); pdir=solve_direction(Gp,a["first_grad"],cfg.lam)
            prow[f"{label}_rank"]=ps["metric_rank"]; prow[f"{label}_trace"]=ps["metric_trace"]
            prow[f"{label}_trace_ratio_to_QFIM"]=(float(onp.trace(Gp)/qtrace) if qtrace!=0 else onp.nan)
            prow[f"{label}_direction_cosine_to_QNG"]=cosine(pdir,qdir); prow[f"{label}_maxeig_minus_QFIM"]=maxeig_diff(Gp,Gq)
        prow["maxeig_A1_minus_local"]=maxeig_diff(probes["A1"],probes["local"])
        prow["maxeig_local_minus_le2"]=maxeig_diff(probes["local"],probes["le2"])
        prow["maxeig_le2_minus_QFIM"]=maxeig_diff(probes["le2"],Gq); pairs.append(prow)
    stem=f"{cfg.dataset}_seed{cfg.seed:03d}"
    pd.DataFrame(rows).to_csv(output_dir/f"results_{stem}.csv",index=False)
    pd.DataFrame(curves).to_csv(output_dir/f"curves_{stem}.csv",index=False)
    pd.DataFrame(mdiags).to_csv(output_dir/f"metric_diags_{stem}.csv",index=False)
    pd.DataFrame(pairs).to_csv(output_dir/f"pair_diags_{stem}.csv",index=False)
    meta=dict(config=cfg.__dict__,subset_seed=SUBSET_SEED,split_seed=SPLIT_SEED,stochastic_seed_role="parameter initialization + minibatch schedule + fixed Haar instance for su2_haar; NOT the train/test split",families=list(FAMILIES),methods=list(METHODS),readout="all diagonal Pauli-Z strings through weight 2",readout_probes=["A1=Z0","local=all Zi","le2=all Zi and ZiZj"],timing_interpretation="backend-specific implementation wall-clock; use matched logical workload counters for backend-independent fairness checks",pennylane=qml.__version__,numpy=onp.__version__,python=platform.python_version(),github_sha=os.environ.get("GITHUB_SHA"),github_run_id=os.environ.get("GITHUB_RUN_ID"))
    (output_dir/f"config_{stem}.json").write_text(json.dumps(meta,indent=2,sort_keys=True))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True,choices=DATASETS); p.add_argument("--seed",required=True,type=int); p.add_argument("--n-qubits",type=int,default=6); p.add_argument("--n-layers",type=int,default=3); p.add_argument("--n-samples",type=int,default=80); p.add_argument("--steps",type=int,default=20); p.add_argument("--lr",type=float,default=0.03); p.add_argument("--lam",type=float,default=1e-3); p.add_argument("--loss-batch",type=int,default=8); p.add_argument("--metric-batch",type=int,default=4); p.add_argument("--metric-every",type=int,default=2); p.add_argument("--rcond",type=float,default=1e-8); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    cfg=Config(a.dataset,a.seed,a.n_qubits,a.n_layers,a.n_samples,a.steps,a.lr,a.lam,a.loss_batch,a.metric_batch,a.metric_every,a.rcond)
    df=run_job(cfg,a.output_dir); print(df[["dataset","family","method","seed","test_loss","test_acc","total_seconds"]].to_string(index=False))

if __name__=="__main__": main()
