# AQNG paper v2 plan

Working title: **Accessible Quantum Natural Gradient: Readout-Induced Geometry for Variational Optimization**

## Scientific claim hierarchy

1. **Definition / geometry.** For retained commuting outcome features, the accessible metric
   \(G_{\rm acc}=B^{-1}\sum_b J_b^T\Sigma_b^+J_b\) is the exact pullback/Gram matrix of the Fisher-score projection onto the retained feature span. AQNG uses this interface-induced geometry rather than attempting to reconstruct the full QFIM.
2. **Rank versus orientation.** Physical, Haar-random, and cross-fitted tangent-aligned readouts are compared at exactly matched rank. Random readouts provide the rank-only null; cross-fitted alignment isolates orientation gain without same-sample optimism.
3. **Optimization evidence.** In the 240 generic primary cases, aligned AQNG has the best pooled mean test loss/accuracy among the eight frozen methods, but random AQNG and SGD are close; there is no universal optimizer winner.
4. **Large/deep follow-up.** In the 15 matched SU(2)-Haar scaling/depth cells, aligned AQNG beats SGD 15/15 and Full-QNG 14/15; pooled Holm-adjusted p-values are 0.00122 and 0.00580 respectively. The aligned-vs-Adam difference is not statistically resolved.
5. **Accessibility is not trainability.** U(1) provides a structured negative control: physical/aligned retention is near unity while the effective descent signal collapses and terminal accuracy remains chance-like at larger n.
6. **More retained geometry is not always better.** Weight-2 readouts can strongly increase retention while worsening metric conditioning and terminal task loss. Readout selection should therefore be condition-aware, not retention-maximizing alone.
7. **Finite-shot scope.** AQNG remains usable under the tested finite-shot protocol, but finite-shot Full-QNG keeps an analytic metric oracle; hardware-cost superiority is therefore not claimed.

## Main-text structure

### I. Introduction
- Full quantum geometry versus the geometry exposed by a measurement/readout interface.
- Measurement-accessible tangent retention: rank, spectrum, orientation.
- Relation to Gross--Rieser QELM/PTM feature decodability: common projection skeleton, different differential/optimization object.
- Contributions and deliberately limited claims.

### II. Accessible quantum natural gradient
- Fixed measurement and commuting retained features.
- Probability Jacobian, feature Jacobian, covariance, pseudoinverse.
- Score-projection identity and \(G_{\rm acc}=J^T\Sigma^+J\).
- Damping, normalization, rank deficiency, primal/dual/SVD solves.
- Full-QNG convention and what AQNG does *not* approximate.

### III. Rank-matched readout controls
- Physical low-weight readouts.
- Haar-random equal-rank readout.
- Cross-fitted tangent-aligned equal-rank readout.
- Retention \(R\), normalized retention \(\rho\), and frozen calibration.

### IV. Experimental protocol
- Four binary datasets; four circuit families.
- Frozen n=6, L=3 primary benchmark with 20 paired seeds and eight optimizers.
- Baseline tuning protocol and no test/U(1) leakage.
- Scaling, depth, readout order, finite-shot, and SGD/Adam follow-up.
- Shot/execution accounting and caveats.

### V. Results
A. Generic primary benchmark.
B. Geometry: random rank law and cross-fitted orientation gain.
C. Qubit/depth scaling and matched Adam/SGD follow-up.
D. Accessibility != trainability: U(1) control.
E. More accessibility != better optimization: readout-order/conditioning ablation.
F. Finite-shot robustness and resource accounting.

### VI. Discussion
- What AQNG explains that Adam/SGD do not: poor accessibility vs poor conditioning vs weak task gradient.
- Why aligned retention and task gain are only weakly correlated globally.
- Relation to barren plateaus: no claim of solving them; U(1) is a concentration-like negative regime, not a formal barren-plateau theorem.
- Noncommuting readouts require explicit measurement grouping/interface models.
- Calibration cost is separate and not free.

### VII. Conclusion
One-sentence thesis: **measurement-accessible tangent geometry is a controllable and interpretable optimization resource, but it is neither equivalent to the full QFIM nor to task trainability.**

## Main figures

### Fig. 1 — AQNG construction and geometric layers
Conceptual diagram: parameter tangent -> full QFIM -> fixed measurement score space -> retained readout subspace -> accessible metric -> AQNG step. Visually separate *full state geometry*, *measurement-accessible geometry*, and *task gradient*.

### Fig. 2 — Frozen generic primary benchmark
Two panels using `generic_primary_summary.csv`: pooled test loss and accuracy for the eight methods over the 240 generic cases. Prefer point + uncertainty presentation over bars. Highlight that aligned is best descriptively while random/SGD remain close.

### Fig. 3 — Rank law and orientation under scaling
Use `scaling_geometry_summary.csv`: for SU(2)-Haar n=4,6,8,10 show \(R\) and/or \(\rho\) for physical, aligned, random. Random should stay near \(\rho=1\); aligned normalized retention grows strongly. Include U(1) as a separate panel or inset to show near-unity physical/aligned retention.

### Fig. 4 — Larger/deeper matched optimizer comparison
Use `followup/large_deep_summary.csv`: n=8 L3, n=10 L3, n=6 L5; methods aligned/random/physical/Adam/SGD/Full-QNG. Main quantity: terminal test loss with five-seed uncertainty. Annotate pooled paired results: aligned vs SGD Holm p=0.00122; aligned vs Full-QNG p=0.00580; aligned vs Adam not significant.

### Fig. 5 — Accessibility is not trainability
Two-part diagnostic. (a) U(1) scaling: near-unity physical/aligned retention while first-step descent diagnostics collapse and accuracy remains 46.7%. (b) SU(2)-Haar readout order 1->2: aligned retention rises sharply but loss can worsen. This is the paper's main interpretability/failure-mode figure.

### Fig. 6 — Finite-shot robustness and resource scope
Use finite-shot summaries/resources. Show 1k vs 10k terminal loss and explicitly label Full-QNG as analytic-metric oracle in this protocol. Put detailed shot accounting in the appendix if main-text space is tight.

## Appendix figures/tables
- Per-dataset/per-family primary results.
- Training curves.
- Metric spectra/condition numbers and clipping incidence.
- Calibration support/rank stability.
- Complete paired tests with Holm correction.
- Full shot/execution resource accounting.
- Hyperparameter tuning grid and frozen selections.
- Reproducibility manifest and result provenance.

## Claim language to avoid
- "AQNG universally outperforms Adam/SGD."
- "AQNG solves barren plateaus."
- "Higher tangent retention necessarily improves supervised learning."
- "Finite-shot Full-QNG is hardware-fair in the current implementation."
- "The U(1) finite-size collapse establishes an asymptotic barren plateau."
