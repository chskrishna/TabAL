# Reproducibility guide

## Frozen protocol

`config_representation.yaml` is the authoritative full benchmark:

- five TDC binary ADMET datasets;
- scaffold split with seed 42 and 70/10/20 train/validation/test fractions;
- train and validation combined into the active-learning pool;
- ten paired active-learning seeds;
- initial label budget 32, batches of 16, final budget 256;
- TabPFN, random forest, XGBoost, and LightGBM;
- random, predictive-entropy, and entropy-then-diversity acquisition;
- ECFP4 and precomputed RDKit2D predictive representations.

All model-policy combinations share the same initial labeled set and random
ordering within each dataset/seed pair. Pool and test scaffolds are asserted to
be disjoint during preprocessing.

## Expected scale

Each representation contains:

```text
5 datasets × 10 seeds × 4 models × 3 acquisitions = 600 trajectories
600 trajectories × 15 budgets = 9,000 metric rows
600 trajectories × 256 labels = 153,600 query-history rows
```

The full run takes several hours on a CUDA workstation because TabPFN inference
is repeated inside the acquisition loop. The published manifests report an
NVIDIA RTX A4000 and the exact package versions used for the archived results.

## Integrity checks

The following commands must both succeed before interpreting or publishing a
run:

```bash
python analyze_results.py --config config_representation.yaml \
  --representation ecfp4 --validate-only
python analyze_results.py --config config_representation.yaml \
  --representation rdkit2d --validate-only
```

Validation checks the complete Cartesian product of datasets, seeds, models,
and acquisitions; exact budget sequences; finite metrics; AUROC bounds; 256
unique queried molecules per trajectory; 32 initial labels; and 16-member
acquisition batches.

## Published artifacts

For each representation, the repository contains:

- `*_metrics.csv`: per-budget predictive metrics and runtimes;
- `*_queries.csv`: complete molecule-level acquisition history;
- `*_manifest.json`: configuration, checkpoint, environment, and hardware;
- `*_validation.json`: strict validation receipt;
- `tables_*`: trajectory, dataset, macro, paired-bootstrap, and diversity tables;
- `figures_*`: publication-ready PNG and PDF figures.

Generated datasets (`data/raw`, `data/processed/*.npz`) and the TabPFN
checkpoint are intentionally excluded because their providers govern
redistribution. Recreate them with `prepare_data.py` and `fetch_checkpoint.py`.
