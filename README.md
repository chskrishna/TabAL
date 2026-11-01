# TabAL

Official code and results for **“Do Tabular Foundation Models Improve Molecular
Active Learning? A Controlled Study on Scaffold-Split ADMET Tasks.”**

TabAL compares TabPFN, random forests, XGBoost, and LightGBM in pool-based
molecular active learning. The benchmark uses five TDC ADMET classification
tasks, scaffold-held-out test sets, two molecular representations (ECFP4 and
RDKit2D), three acquisition policies, and ten paired seeds.

## Main result

The public result files contain **1,200 validated learning trajectories**
(600 per representation), each evaluated at 15 label budgets from 32 to 256.

| Representation | Best model | Acquisition | Macro AULC |
|---|---|---|---:|
| ECFP4 | Random forest | Entropy + diversity | 0.7959 |
| RDKit2D | TabPFN | Entropy + diversity | 0.8575 |

The representation matters: TabPFN is not competitive on ECFP4 in this study,
but becomes the best macro-AULC configuration with precomputed RDKit2D
descriptors. See [`results/README.md`](results/README.md) for schemas and
artifact provenance.

## Repository layout

```text
prepare_data.py              download, clean, split, and featurize TDC data
run_al.py                    run resumable active-learning trajectories
analyze_results.py           validate runs and regenerate tables/figures
config_representation.yaml   frozen ECFP4/RDKit2D + LightGBM protocol
results/                     validated raw runs, tables, and figures
tests/                       protocol and checkpoint regression tests
```

RDKit2D descriptors are computed once in `prepare_data.py`, median-imputed and
standardized using the active-learning pool only, then stored in processed NPZ
files. Diversity selection always uses ECFP4/Tanimoto, isolating the effect of
the predictive representation.

## Installation

Python 3.11 is recommended. The commands below use Conda only for isolation;
all Python dependencies remain pinned in `requirements.in`.

```bash
conda env create -f environment.yml
conda activate tabal

# Install the PyTorch build appropriate for your CPU/CUDA platform first:
# https://pytorch.org/get-started/locally/
python -m pip install -r requirements.in
```

The archived CUDA environment used for the reported runs is recorded in
`requirements.cuda.lock.txt`; it is evidence, not a portable installer.

## Quick verification

```bash
python fetch_checkpoint.py --config config_representation.yaml
python prepare_data.py --config config_representation.yaml
python -m pytest -q

python run_al.py --config config_representation.yaml --smoke \
  --representations ecfp4 rdkit2d
python analyze_results.py --config config_representation.yaml \
  --representation ecfp4 --smoke --validate-only
python analyze_results.py --config config_representation.yaml \
  --representation rdkit2d --smoke --validate-only
```

Smoke mode uses one seed and budgets 32, 48, and 64. Use `--datasets`,
`--models`, `--acquisitions`, or `--seeds` to run a smaller targeted check.

## Reproduce the paper benchmark

```bash
python run_al.py --config config_representation.yaml \
  --representations ecfp4 rdkit2d

python analyze_results.py --config config_representation.yaml \
  --representation ecfp4 --validate-only
python analyze_results.py --config config_representation.yaml \
  --representation rdkit2d --validate-only

python analyze_results.py --config config_representation.yaml \
  --representation ecfp4
python analyze_results.py --config config_representation.yaml \
  --representation rdkit2d
```

Each run checkpoints after every complete trajectory and safely replaces a retried
trajectory, so the same command resumes after interruption. Full details,
hardware notes, and integrity checks are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Data and model provenance

- Datasets are downloaded by PyTDC; raw data is not redistributed here.
- Processed arrays are locally generated and ignored by Git.
- The TabPFN v2.5 checkpoint is downloaded from `Prior-Labs/tabpfn_2_5` and
  accepted only when its SHA-256 matches the frozen configuration.
- Manifests record the configuration hash, package versions, checkpoint hash,
  and execution hardware for every published representation.

## Citation

Please cite the paper title above. The archival BibTeX entry will be added when
the paper identifier is public.

## License

Code is released under the [MIT License](LICENSE). Dataset and model checkpoint
licenses remain with their respective providers.
