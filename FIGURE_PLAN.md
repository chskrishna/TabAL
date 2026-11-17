# Final paper figure plan

The repository currently includes four publication-style figures for the
single-split precursor in `results/figures_precursor/`. The final paper figures
must be regenerated from the three-split 2,700-trajectory artifacts.

## Main figures

1. **Representation changes label efficiency and model ranking.** Paired
   RDKit2D-minus-ECFP4 AULC forest plot, learner–policy heatmap, and rank
   slopegraph. Replace precursor Figure 1 using split-aware cluster bootstrap
   intervals.
2. **Learning curves across representations.** A 2 × 4 grid for two
   representations and four learners, with acquisition policy encoded by line
   style and color. Replace precursor Figure 2.
3. **The descriptor advantage is not dimensionality alone.** Paired AULC
   effects for ECFP4-1024, ECFP4-256, Gaussian projection-210, and RDKit2D-210.
   This figure is blocked until compression-control results are supplied.
4. **Prediction features dominate query-sequence differences.** Acquisition
   replay decomposition separating training-representation effects from
   sequence effects for TabPFN and Random Forest. This figure is blocked until
   replay results are supplied.

## Supplementary figures

- Acquisition-policy effects relative to random acquisition.
- Per-task/per-split representation effects.
- Runtime distributions by learner, representation, and policy.
- Probability-quality and query-diversity diagnostics, if retained in the
  final manuscript.

## Required final data fields

The final per-budget metrics should contain at least:

```text
dataset, split_seed, seed, representation, model, acquisition,
round, budget, test_auroc, test_auprc, brier, nll,
fit_runtime, prediction_runtime, acquisition_prediction_runtime
```

Compression-control data additionally need a representation/control identifier
(`ecfp4_1024`, `ecfp4_256`, `gaussian_210`, or `rdkit2d_210`). Replay data need
both `acquisition_representation` and `training_representation`, plus a stable
query-sequence identifier. Figures will use paired or split-aware cluster
bootstrap intervals; no panel will be generated from summary values alone.
