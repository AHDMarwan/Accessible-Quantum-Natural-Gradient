"""Aggregate AQNG medium-pilot outputs into explicit pre-scale go/no-go checks."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ["AQNG-physical", "AQNG-random", "AQNG-aligned", "Full-QNG"]


def _spearman_rank(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 1.0 if np.allclose(ra, rb) else 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(results: pd.DataFrame, cal: pd.DataFrame):
    checks = {}

    finite_cal = cal[cal["calibration_finite_shot"].astype(str).str.lower().isin(["true", "1"])]
    stable = True
    details = {}
    for fam, g in finite_cal.groupby("family"):
        dims = sorted(set(g["centered_score_dimension"].astype(int)))
        ranks = sorted(set(g["physical_rank"].astype(int)))
        supports = sorted(set(g["reference_support_size"].astype(int)))
        ok = len(dims) == len(ranks) == len(supports) == 1
        stable &= ok
        details[fam] = {"dims": dims, "ranks": ranks, "supports": supports, "pass": ok}
    checks["stable_support_rank_across_shots"] = {"pass": bool(stable), "details": details}

    generic = cal[cal.family == "su2_haar"]
    rrho = generic["random_rank_heldout_rho"].astype(float)
    med_rrho = float(rrho.median()) if len(rrho) else np.nan
    checks["random_rank_near_baseline"] = {"pass": bool(np.isfinite(med_rrho) and 0.5 <= med_rrho <= 1.5), "median_rho": med_rrho}

    if len(generic):
        ratio = generic["aligned_crossfit_heldout_retention"].astype(float) / generic["physical_heldout_retention"].astype(float)
        med = float(ratio.median())
    else:
        med = np.nan
    checks["generic_aligned_retention_above_physical"] = {"pass": bool(np.isfinite(med) and med >= 1.05), "median_ratio": med}

    u1 = cal[cal.family == "u1_rzxy"]
    if len(u1):
        ratio = u1["physical_heldout_retention"].astype(float) / u1["aligned_crossfit_heldout_retention"].astype(float)
        med = float(ratio.median())
    else:
        med = np.nan
    checks["u1_physical_already_aligned"] = {"pass": bool(np.isfinite(med) and med >= 0.90), "median_physical_over_aligned": med}

    numeric_cols = ["test_loss", "test_acc", "first_g_dot_d", "first_actual_same_batch_decrease"]
    finite_ok = bool(np.isfinite(results[numeric_cols].to_numpy(dtype=float)).all())
    checks["numerically_finite"] = {"pass": finite_ok}

    shot = results[(results.family == "su2_haar") & results.shots.notna()]
    rank_corr = np.nan
    if len(shot):
        means = shot.groupby(["shots", "method"])["test_loss"].mean().unstack("method")
        if 1000 in means.index and 10000 in means.index:
            common = [m for m in METHODS if m in means.columns]
            rank_corr = _spearman_rank(means.loc[1000, common].to_numpy(), means.loc[10000, common].to_numpy())
    checks["finite_shot_ranking_stable_1k_10k"] = {"pass": bool(np.isfinite(rank_corr) and rank_corr >= 0.5), "spearman": rank_corr}

    ana = results[(results.family == "su2_haar") & results.shots.isna()]
    wins = total = 0
    if len(ana):
        p = ana[ana.method == "AQNG-physical"].set_index("seed")["test_loss"]
        a = ana[ana.method == "AQNG-aligned"].set_index("seed")["test_loss"]
        common = p.index.intersection(a.index)
        total = len(common)
        wins = int(np.sum(a.loc[common].to_numpy() <= p.loc[common].to_numpy()))
    checks["aligned_optimizer_effect_repeats_across_seeds"] = {"pass": bool(total >= 5 and wins >= 3), "aligned_loss_wins": wins, "paired_seeds": total}

    all_pass = all(x["pass"] for x in checks.values())
    return {"go_for_large_scale": bool(all_pass), "checks": checks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = evaluate(pd.read_csv(a.results), pd.read_csv(a.calibration))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
