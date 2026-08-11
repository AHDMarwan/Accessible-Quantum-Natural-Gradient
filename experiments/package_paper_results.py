#!/usr/bin/env python3
"""Build a durable, paper-ready result package from completed AQNG Actions artifacts.

Inputs
------
--base-dir:
    Extracted ``aqng-paper-large-scale-summary`` artifact from workflow run
    31480796466. It contains the aggregate raw CSVs for the frozen 470-cell
    paper-scale matrix.
--followup-dir:
    Directory containing the extracted artifacts from workflow run 31522334431,
    the 15-cell SGD/Adam scaling/depth follow-up.

The script fixes only reporting/aggregation. It does not rerun or modify any
experiment outcome.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

CORE_METHODS = [
    "AQNG-physical", "AQNG-random", "AQNG-aligned", "AQNG-Z0",
    "Full-QNG", "Block-QNG", "SGD", "Adam",
]
BASE_RUN_ID = 31480796466
BASE_SHA = "1aad0e2413436084d84c9b013ef4b75a32771d06"
FOLLOWUP_RUN_ID = 31522334431
FOLLOWUP_SHA = "82616cabacec2f8740a8955e0c750e169112c9eb"


def block_from_tag(tag: str) -> str:
    tag = str(tag)
    for name in ("primary", "finite", "scaling", "depth", "order"):
        if f"paper-run-{name}-" in tag or f"paper-{name}-" in tag:
            return name
    return "unknown"


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return out
    vals = p[idx]
    order = np.argsort(vals)
    m = len(vals)
    adj = np.empty(m)
    running = 0.0
    for rank, pos in enumerate(order):
        running = max(running, (m-rank)*vals[pos])
        adj[pos] = min(1.0, running)
    out[idx] = adj
    return out


def safe_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b-a
    if len(d) == 0: return np.nan
    if np.allclose(d, 0.0, atol=1e-14, rtol=0.0): return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def summary(df, group_cols):
    if df.empty: return pd.DataFrame()
    cols = {
        "n": ("seed", "nunique"),
        "test_loss_mean": ("test_loss", "mean"),
        "test_loss_std": ("test_loss", "std"),
        "test_acc_mean": ("test_acc", "mean"),
        "test_acc_std": ("test_acc", "std"),
        "train_loss_mean": ("train_loss", "mean"),
        "total_seconds_mean": ("total_seconds", "mean"),
        "first_g_dot_d_mean": ("first_g_dot_d", "mean"),
        "first_actual_decrease_mean": ("first_actual_same_batch_decrease", "mean"),
    }
    return df.groupby(group_cols, dropna=False).agg(**cols).reset_index()


def cal_summary(cal, group_cols):
    if cal.empty: return pd.DataFrame()
    fields = [
        "physical_rank", "centered_score_dimension", "rank_baseline",
        "reference_support_size", "physical_heldout_retention",
        "physical_heldout_rho", "aligned_crossfit_heldout_retention",
        "aligned_crossfit_heldout_rho", "random_rank_heldout_retention",
        "random_rank_heldout_rho",
    ]
    aggs = {c:(c,"mean") for c in fields if c in cal.columns}
    aggs["n"] = ("seed","nunique")
    return cal.groupby(group_cols, dropna=False).agg(**aggs).reset_index()


def paired_primary(primary):
    rows=[]
    for (dataset,family),g in primary.groupby(["dataset","family"], sort=True):
        a0=g[g.method=="AQNG-aligned"].set_index("seed")
        if a0.empty: continue
        for ref in [m for m in CORE_METHODS if m!="AQNG-aligned"]:
            b0=g[g.method==ref].set_index("seed")
            seeds=a0.index.intersection(b0.index)
            if len(seeds)==0: continue
            a=a0.loc[seeds,"test_loss"].to_numpy(float)
            b=b0.loc[seeds,"test_loss"].to_numpy(float)
            d=b-a
            rows.append(dict(dataset=dataset,family=family,method_a="AQNG-aligned",method_b=ref,
                n_pairs=len(seeds),mean_loss_a=a.mean(),mean_loss_b=b.mean(),
                mean_loss_advantage_a=d.mean(),median_loss_advantage_a=np.median(d),
                wins_a=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),
                wins_b=int((d<-1e-12).sum()),p_raw=safe_wilcoxon(a,b)))
    out=pd.DataFrame(rows)
    if len(out): out["p_holm"] = holm(out.p_raw)
    return out


def relation(primary, cal_primary):
    if primary.empty or cal_primary.empty: return {}
    piv=primary.pivot_table(index=["dataset","family","seed"], columns="method", values="test_loss", aggfunc="first").reset_index()
    keep=["dataset","family","seed","physical_heldout_retention","aligned_crossfit_heldout_retention","random_rank_heldout_retention"]
    m=cal_primary[keep].merge(piv,on=["dataset","family","seed"],how="inner")
    need={"AQNG-physical","AQNG-aligned","AQNG-random"}
    if not need.issubset(m.columns): return {}
    m["aligned_retention_gain"]=m.aligned_crossfit_heldout_retention-m.physical_heldout_retention
    m["aligned_loss_gain"]=m["AQNG-physical"]-m["AQNG-aligned"]
    m["random_retention_gain"]=m.random_rank_heldout_retention-m.physical_heldout_retention
    m["random_loss_gain"]=m["AQNG-physical"]-m["AQNG-random"]
    def corr(x,y):
        x=np.asarray(x,float); y=np.asarray(y,float); mask=np.isfinite(x)&np.isfinite(y)
        if mask.sum()<3 or np.std(x[mask])==0 or np.std(y[mask])==0:
            return {"rho":None,"p":None,"n":int(mask.sum())}
        r=spearmanr(x[mask],y[mask]); return {"rho":float(r.statistic),"p":float(r.pvalue),"n":int(mask.sum())}
    return {
        "aligned_retention_vs_loss_gain":corr(m.aligned_retention_gain,m.aligned_loss_gain),
        "random_retention_vs_loss_gain":corr(m.random_retention_gain,m.random_loss_gain),
        "aligned_by_family":{str(f):corr(g.aligned_retention_gain,g.aligned_loss_gain) for f,g in m.groupby("family")},
        "n_rows":int(len(m)),
    }


def concat_followup(root: Path, pattern: str):
    frames=[]
    for p in sorted(root.rglob(pattern)):
        try:
            df=pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if len(df)==0: continue
        artifact=p.parent.name
        df["source_artifact"] = artifact
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def setting_label(row):
    n=int(row.n_qubits); L=int(row.n_layers)
    if n in (8,10) and L==3: return f"n{n}_L3"
    if n==6 and L==5: return "n6_L5"
    return f"n{n}_L{L}"


def large_deep_package(base_results, follow_results):
    b=base_results[(base_results.dataset=="iris01")&(base_results.family=="su2_haar")].copy()
    b=b[((b.paper_block=="scaling") & b.n_qubits.isin([8,10]) & (b.n_layers==3)) |
        ((b.paper_block=="depth") & (b.n_qubits==6) & (b.n_layers==5))]
    b=b[b.method.isin(["AQNG-physical","AQNG-random","AQNG-aligned","Full-QNG"])]
    f=follow_results.copy()
    f=f[f.method.isin(["SGD","Adam"])]
    merged=pd.concat([b,f], ignore_index=True, sort=False)
    merged["setting"]=[setting_label(r) for r in merged.itertuples()]
    merged=merged.sort_values(["setting","seed","method"]).reset_index(drop=True)
    summ=summary(merged,["setting","method"])

    rows=[]
    refs=["AQNG-physical","AQNG-random","Full-QNG","SGD","Adam"]
    for setting,g in list(merged.groupby("setting"))+[("pooled_15",merged)]:
        a0=g[g.method=="AQNG-aligned"].set_index(["setting","seed"] if setting=="pooled_15" else ["seed"])
        for ref in refs:
            b0=g[g.method==ref].set_index(["setting","seed"] if setting=="pooled_15" else ["seed"])
            idx=a0.index.intersection(b0.index)
            a=a0.loc[idx,"test_loss"].to_numpy(float); b_=b0.loc[idx,"test_loss"].to_numpy(float)
            if len(a)==0: continue
            d=b_-a
            rows.append(dict(setting=setting,method_a="AQNG-aligned",method_b=ref,n_pairs=len(a),
                mean_loss_a=a.mean(),mean_loss_b=b_.mean(),mean_loss_advantage_a=d.mean(),
                wins_a=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),wins_b=int((d<-1e-12).sum()),
                p_raw=safe_wilcoxon(a,b_)))
    tests=pd.DataFrame(rows)
    if len(tests): tests["p_holm"] = holm(tests.p_raw)
    return merged,summ,tests


def copy_or_gzip(src: Path, dst: Path, gzip_it=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if gzip_it:
        with src.open("rb") as fi, gzip.open(dst,"wb",compresslevel=9) as fo:
            shutil.copyfileobj(fi,fo)
    else:
        shutil.copy2(src,dst)


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-dir",type=Path,required=True)
    ap.add_argument("--followup-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); out=a.output_dir
    if out.exists(): shutil.rmtree(out)
    (out/"raw").mkdir(parents=True)
    (out/"summaries").mkdir(parents=True)
    (out/"followup").mkdir(parents=True)

    r=pd.read_csv(a.base_dir/"results_all.csv")
    cal=pd.read_csv(a.base_dir/"calibration_all.csv")
    ori=pd.read_csv(a.base_dir/"orientation_all.csv")
    for df in (r,cal,ori): df["paper_block"]=df.paper_tag.map(block_from_tag)

    copy_or_gzip(a.base_dir/"results_all.csv", out/"raw"/"results_all.csv")
    copy_or_gzip(a.base_dir/"calibration_all.csv", out/"raw"/"calibration_all.csv")
    copy_or_gzip(a.base_dir/"orientation_all.csv", out/"raw"/"orientation_all.csv")
    copy_or_gzip(a.base_dir/"curves_all.csv", out/"raw"/"curves_all.csv.gz", True)
    copy_or_gzip(a.base_dir/"metric_diags_all.csv", out/"raw"/"metric_diags_all.csv.gz", True)
    for name in ["baseline_hparams.json","baseline_candidates.csv","manifest.json","paper_scale_report.json"]:
        p=a.base_dir/name
        if p.exists(): copy_or_gzip(p,out/"raw"/f"source_{name}")

    blocks={k:r[r.paper_block==k].copy() for k in ["primary","finite","scaling","depth","order"]}
    cblocks={k:cal[cal.paper_block==k].copy() for k in ["primary","finite","scaling","depth","order"]}
    outputs={
        "primary_summary.csv":summary(blocks["primary"],["dataset","family","method"]),
        "finite_shot_summary.csv":summary(blocks["finite"],["dataset","family","shots","method"]),
        "scaling_summary.csv":summary(blocks["scaling"],["dataset","family","n_qubits","method"]),
        "depth_summary.csv":summary(blocks["depth"],["dataset","family","n_layers","method"]),
        "readout_order_summary.csv":summary(blocks["order"],["dataset","family","readout_order","method"]),
        "primary_geometry_summary.csv":cal_summary(cblocks["primary"],["dataset","family"]),
        "finite_geometry_summary.csv":cal_summary(cblocks["finite"],["dataset","family","shots"]),
        "scaling_geometry_summary.csv":cal_summary(cblocks["scaling"],["dataset","family","n_qubits"]),
        "depth_geometry_summary.csv":cal_summary(cblocks["depth"],["dataset","family","n_layers"]),
        "readout_order_geometry_summary.csv":cal_summary(cblocks["order"],["dataset","family","readout_order"]),
        "paired_primary_tests.csv":paired_primary(blocks["primary"]),
    }
    resource_cols=[c for c in ["training_shots_total","calibration_shots_shared","end_to_end_shots_including_calibration","training_executions_total","calibration_executions_shared","end_to_end_executions_including_calibration"] if c in blocks["finite"].columns]
    outputs["finite_resource_summary.csv"]=blocks["finite"].groupby(["dataset","family","shots","method"],dropna=False)[resource_cols].mean().reset_index()
    generic=blocks["primary"][blocks["primary"].family!="u1_rzxy"]
    outputs["generic_primary_summary.csv"]=summary(generic,["method"])
    if len(outputs["generic_primary_summary.csv"]):
        counts=generic.groupby("method").size().rename("n_cases").reset_index()
        outputs["generic_primary_summary.csv"]=outputs["generic_primary_summary.csv"].merge(counts,on="method",how="left")
    for name,df in outputs.items(): df.to_csv(out/"summaries"/name,index=False)

    fres=concat_followup(a.followup_dir,"results_*.csv")
    fcur=concat_followup(a.followup_dir,"curves_*.csv")
    fcal=concat_followup(a.followup_dir,"calibration_diags_*.csv")
    fori=concat_followup(a.followup_dir,"orientation_diags_*.csv")
    if len(fres):
        fres["paper_block"]="followup"
        fres["paper_tag"]=[f"paper-run-baselines-{setting_label(x)}-s{int(x.seed)}" for x in fres.itertuples()]
    fres.to_csv(out/"followup"/"sgd_adam_results_all.csv",index=False)
    fcur.to_csv(out/"followup"/"sgd_adam_curves_all.csv",index=False)
    fcal.to_csv(out/"followup"/"sgd_adam_calibration_all.csv",index=False)
    fori.to_csv(out/"followup"/"sgd_adam_orientation_all.csv",index=False)

    configs=[]
    for p in sorted(a.followup_dir.rglob("config_*.json")):
        d=json.loads(p.read_text()); d["source_artifact"]=p.parent.name; configs.append(d)
    (out/"followup"/"sgd_adam_configs.json").write_text(json.dumps(configs,indent=2,allow_nan=True))

    merged,msum,mtests=large_deep_package(r.assign(paper_block=r.paper_tag.map(block_from_tag)),fres)
    merged.to_csv(out/"followup"/"large_deep_merged_results.csv",index=False)
    msum.to_csv(out/"followup"/"large_deep_summary.csv",index=False)
    mtests.to_csv(out/"followup"/"large_deep_paired_tests.csv",index=False)

    rel=relation(blocks["primary"],cblocks["primary"])
    report={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_runs":{"paper_scale":{"run_id":BASE_RUN_ID,"head_sha":BASE_SHA},"sgd_adam_followup":{"run_id":FOLLOWUP_RUN_ID,"head_sha":FOLLOWUP_SHA}},
        "completeness":{
            "paper_scale_cells":int(r.paper_tag.nunique()),
            "paper_scale_terminal_rows":int(len(r)),
            "paper_scale_calibration_rows":int(len(cal)),
            "primary_cells":int(blocks["primary"][["dataset","family","seed"]].drop_duplicates().shape[0]),
            "scaling_cells":int(blocks["scaling"][["family","n_qubits","seed"]].drop_duplicates().shape[0]),
            "depth_cells":int(blocks["depth"][["family","n_layers","seed"]].drop_duplicates().shape[0]),
            "order_cells":int(blocks["order"][["dataset","family","readout_order","seed"]].drop_duplicates().shape[0]),
            "finite_cells":int(blocks["finite"][["dataset","family","shots","seed"]].drop_duplicates().shape[0]),
            "followup_cells":int(fres[["n_qubits","n_layers","seed"]].drop_duplicates().shape[0]),
            "followup_terminal_rows":int(len(fres)),
        },
        "expected":{"primary_cells":320,"scaling_cells":40,"depth_cells":30,"order_cells":40,"finite_cells":40,"followup_cells":15},
        "orientation_task_relation":rel,
        "note":"All summaries are regenerated from raw aggregate tables. No experiment outcome is altered. The source summary artifact had empty block summaries because its parser matched 'paper-primary' rather than the actual 'paper-run-primary' tags.",
    }
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))

    gsum=outputs["generic_primary_summary.csv"].sort_values("test_loss_mean")
    lsum=msum.copy()
    lines=[
        "# AQNG paper-scale results (frozen campaign + SGD/Adam follow-up)","",
        "This directory is the durable result package for the frozen AQNG paper-scale campaign.",
        "It contains the complete aggregate terminal results, calibration/orientation diagnostics, lossless-compressed training curves and metric diagnostics, corrected paper summaries, and the matched SGD/Adam large/deep follow-up.","",
        "## Provenance","",
        f"- Paper-scale workflow run: `{BASE_RUN_ID}` at `{BASE_SHA}`.",
        f"- SGD/Adam follow-up workflow run: `{FOLLOWUP_RUN_ID}` at `{FOLLOWUP_SHA}`.",
        "- Paper-scale matrix: 470 cells = 320 primary + 40 scaling + 30 depth + 40 readout-order + 40 finite-shot.",
        "- Terminal paper-scale rows: 3160.",
        "- Follow-up: 15 matched cells, 30 terminal rows (SGD and Adam only).","",
        "## Generic primary benchmark (U(1) negative control excluded)","",
        "| method | mean test loss | mean test accuracy |", "|---|---:|---:|",
    ]
    for x in gsum.itertuples(): lines.append(f"| {x.method} | {x.test_loss_mean:.6f} | {x.test_acc_mean:.4f} |")
    lines += ["","## Large/deep matched comparison","","| setting | method | mean test loss | mean test accuracy |","|---|---|---:|---:|"]
    for x in lsum.sort_values(["setting","test_loss_mean"]).itertuples():
        lines.append(f"| {x.setting} | {x.method} | {x.test_loss_mean:.6f} | {x.test_acc_mean:.4f} |")
    lines += ["","## Files","",
        "- `raw/results_all.csv`: all 3160 terminal rows from the frozen paper-scale matrix.",
        "- `raw/calibration_all.csv`: all 470 calibration rows.",
        "- `raw/orientation_all.csv`: orientation diagnostics.",
        "- `raw/curves_all.csv.gz`: complete training curves (lossless gzip).",
        "- `raw/metric_diags_all.csv.gz`: complete metric diagnostics (lossless gzip).",
        "- `summaries/`: corrected block summaries, geometry summaries, resource accounting, and planned paired tests.",
        "- `followup/`: complete SGD/Adam follow-up aggregates plus the merged AQNG/Full-QNG/SGD/Adam comparison.",
        "- `report.json`: provenance, completeness checks, and orientation/task correlation diagnostics.","",
        "## Reporting fix","",
        "The completed source workflow produced valid raw aggregate CSVs but empty block-summary CSVs because the reporting parser searched for tags such as `paper-primary`, while the actual artifact tags are `paper-run-primary-...`. This package regenerates those summaries from the unchanged raw tables.",
    ]
    (out/"README.md").write_text("\n".join(lines)+"\n")

    files=sorted(p for p in out.rglob("*") if p.is_file())
    with (out/"SHA256SUMS").open("w") as f:
        for p in files: f.write(f"{sha256(p)}  {p.relative_to(out).as_posix()}\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
