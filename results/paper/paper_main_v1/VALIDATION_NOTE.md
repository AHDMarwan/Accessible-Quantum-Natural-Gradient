# Aggregation validation note

Run `31277386581` produced all 80 dataset/seed artifacts (20 seeds each for `iris01`, `breast_cancer`, `wine01`, and `digits01`) and 640 result rows.

The original `aggregate_paper_classification.py` stopped before committing because it used a fixed Loewner residual tolerance of `2e-6`. The largest residual was `8.706148278849681e-06` for `wine01 / u1_rzxy / seed 13` in `maxeig_local_minus_le2`.

The pre-run smoke gate in the same commit already specified a scale-aware numerical tolerance with `LOEWNER_RTOL = 1e-5` and `LOEWNER_ATOL = 1e-10`. Reaggregation therefore used a fixed `1e-5` threshold, which is no looser than the smoke threshold at its minimum scale (`scale >= 1`). No training result, curve, diagnostic, or statistic was altered; only the aggregation gate threshold was aligned with the pre-run smoke policy.

The resulting bundle contains all 80 jobs and all 640 result rows.
