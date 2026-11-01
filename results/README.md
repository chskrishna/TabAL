# Result artifacts

The current single-split precursor benchmark has passed
`analyze_results.py --validate-only` for both representations: 600 trajectories
and 15 budgets per trajectory. These files are not the paper's final
three-split 2,700-trajectory release.

## Raw schemas

`*_metrics.csv` has one row per trajectory and budget:

- identity: `dataset`, `seed`, `model`, `acquisition`;
- progress: `round`, `budget`;
- performance: `test_auroc`, `test_auprc`, `brier`, `nll`;
- timing: `fit_runtime`, `prediction_runtime`,
  `acquisition_prediction_runtime`.

`*_queries.csv` has one row per revealed label:

- trajectory identity and acquisition round;
- stable `queried_molecule_id` (not the molecular structure);
- acquisition score and entropy;
- revealed binary label;
- mean within-batch ECFP4/Tanimoto similarity;
- `is_initial` flag.

## Derived outputs

`tables_<representation>/` contains CSV and LaTeX versions of:

- `aulc_trajectories`: one AULC and endpoint summary per trajectory;
- `per_dataset_aulc`: mean AULC by dataset/model/acquisition;
- `macro_summary`: cross-dataset mean AULC, rank, and dataset wins;
- `paired_comparisons`: paired bootstrap differences with 95% intervals;
- `query_diversity`: per-trajectory mean within-batch similarity.

`figures_<representation>/` contains the learning curves and paired acquisition
difference plots in PNG and PDF formats.

The `ecfp4_*` and `rdkit2d_*` prefixes identify the predictive representation;
both sets use the same four models and frozen public protocol.
