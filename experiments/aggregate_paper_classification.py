"""Aggregate per-job artifacts and compute paired paper statistics."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATASETS=("iris01","breast_cancer","wine01","digits01")
FAMILIES=("ryrz_cz","su2_cnot","su2_haar","u1_rzxy")
METHODS=("AQNG","Full-QNG")

def read_many(root,prefix):
    paths=sorted(root.rglob(f"{prefix}_*.csv"))
    if not paths: raise RuntimeError(f"No {prefix}_*.csv under {root}")
    return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True), paths

def bootstrap_mean_ci(x,n_boot=5000,seed=123456):
    x=np.asarray(x,float)
    if len(x)==0: return np.nan,np.nan
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(n_boot,len(x)))
    return tuple(np.quantile(x[idx].mean(axis=1),[0.025,0.975]))

def holm_adjust(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]); adj[idx]=min(1.0,running)
    return adj

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--seed-count",type=int,required=True); ap.add_argument("--result-tag",required=True); ap.add_argument("--workflow-inputs-json",default="{}"); a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    results,rpaths=read_many(a.input_dir,"results"); curves,_=read_many(a.input_dir,"curves"); mdiags,_=read_many(a.input_dir,"metric_diags"); pdiags,_=read_many(a.input_dir,"pair_diags")
    expected_jobs=len(DATASETS)*a.seed_count; expected_rows=expected_jobs*len(FAMILIES)*len(METHODS)
    if len(rpaths)!=expected_jobs: raise RuntimeError(f"Expected {expected_jobs} job files, found {len(rpaths)}")
    if len(results)!=expected_rows: raise RuntimeError(f"Expected {expected_rows} result rows, found {len(results)}")
    key=["dataset","family","method","seed"]
    if results.duplicated(key).any(): raise RuntimeError("Duplicate result keys detected")
    expected=set(range(a.seed_count))
    for ds in DATASETS:
        got=set(results.loc[results.dataset==ds,"seed"].astype(int))
        if got!=expected: raise RuntimeError(f"Seed mismatch for {ds}: {sorted(got)}")
    summary=(results.groupby(["dataset","family","method"],sort=False)
             .agg(n=("seed","count"),test_loss_mean=("test_loss","mean"),test_loss_std=("test_loss","std"),test_acc_mean=("test_acc","mean"),test_acc_std=("test_acc","std"),total_seconds_mean=("total_seconds","mean"),metric_seconds_mean=("metric_seconds","mean"),metric_rank_first_mean=("metric_rank_first","mean"),metric_trace_first_mean=("metric_trace_first","mean")).reset_index())
    paired=[]
    for ds in DATASETS:
        for fam in FAMILIES:
            g=results[(results.dataset==ds)&(results.family==fam)]
            w=g.pivot(index="seed",columns="method",values=["test_loss","test_acc","total_seconds"])
            if len(w)!=a.seed_count: raise RuntimeError(f"Incomplete pair cell {ds}/{fam}")
            dl=(w[("test_loss","AQNG")]-w[("test_loss","Full-QNG")]).to_numpy(float)
            da=(w[("test_acc","AQNG")]-w[("test_acc","Full-QNG")]).to_numpy(float)
            sp=(w[("total_seconds","Full-QNG")]/w[("total_seconds","AQNG")]).to_numpy(float)
            lo,hi=bootstrap_mean_ci(dl)
            try: p=1.0 if np.allclose(dl,0) else float(wilcoxon(dl,alternative="two-sided",zero_method="wilcox").pvalue)
            except ValueError: p=1.0
            paired.append(dict(dataset=ds,family=fam,n_pairs=len(dl),delta_loss_mean_AQNG_minus_QNG=float(dl.mean()),delta_loss_median_AQNG_minus_QNG=float(np.median(dl)),delta_loss_ci95_lo=float(lo),delta_loss_ci95_hi=float(hi),AQNG_loss_win_rate=float(np.mean(dl<0)),delta_accuracy_mean_AQNG_minus_QNG=float(da.mean()),runtime_speedup_QNG_over_AQNG_mean=float(sp.mean()),runtime_speedup_QNG_over_AQNG_median=float(np.median(sp)),wilcoxon_p=p))
    stats=pd.DataFrame(paired); stats["wilcoxon_p_holm"]=holm_adjust(stats.wilcoxon_p.to_numpy()); stats["reject_0p05_holm"]=stats.wilcoxon_p_holm<0.05
    pair_summary=(pdiags.groupby(["dataset","family"],sort=False)
                  .agg(n=("seed","count"),direction_cosine_mean=("initial_direction_cosine","mean"),direction_cosine_std=("initial_direction_cosine","std"),trace_ratio_mean=("trace_ratio_Gacc_over_Gq","mean"),trace_ratio_std=("trace_ratio_Gacc_over_Gq","std"),max_eig_Gacc_minus_Gq_max=("max_eig_Gacc_minus_Gq","max")).reset_index())
    results.to_csv(a.output_dir/"results.csv",index=False); summary.to_csv(a.output_dir/"summary.csv",index=False); stats.to_csv(a.output_dir/"stats.csv",index=False); curves.to_csv(a.output_dir/"curves.csv",index=False); mdiags.to_csv(a.output_dir/"metric_diagnostics.csv",index=False); pdiags.to_csv(a.output_dir/"pair_diagnostics.csv",index=False); pair_summary.to_csv(a.output_dir/"pair_summary.csv",index=False)
    try: wf=json.loads(a.workflow_inputs_json)
    except json.JSONDecodeError: wf={"raw":a.workflow_inputs_json}
    manifest=dict(result_tag=a.result_tag,seed_count=a.seed_count,expected_jobs=expected_jobs,expected_result_rows=expected_rows,datasets=list(DATASETS),families=list(FAMILIES),methods=list(METHODS),workflow_inputs=wf,github_sha=os.environ.get("GITHUB_SHA"),github_run_id=os.environ.get("GITHUB_RUN_ID"),github_run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"))
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print("Validated rows:",len(results)); print(stats.to_string(index=False))

if __name__=="__main__": main()
