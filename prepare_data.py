from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tdc.single_pred import ADME, Tox
from tdc.utils import create_scaffold_split

from study import canonicalize, fingerprint, load_config, rdkit2d, scaffold, write_json


LOADERS = {"ADME": ADME, "Tox": Tox}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"Drug", "Y"}
    if not required <= set(raw.columns):
        raise ValueError(f"Expected TDC columns {required}; got {list(raw.columns)}")
    frame = raw.reset_index(drop=False).rename(columns={"index": "source_row"}).copy()
    frame["canonical_smiles"] = frame["Drug"].map(canonicalize)
    invalid = int(frame["canonical_smiles"].isna().sum())
    frame = frame.dropna(subset=["canonical_smiles"]).copy()
    label_counts = frame.groupby("canonical_smiles")["Y"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    conflict_rows = int(frame["canonical_smiles"].isin(conflicts).sum())
    frame = frame[~frame["canonical_smiles"].isin(conflicts)].copy()
    before_dedup = len(frame)
    frame = frame.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    frame["Y"] = pd.to_numeric(frame["Y"], errors="raise").astype(np.int8)
    if set(frame["Y"].unique()) != {0, 1}:
        raise ValueError(f"Native labels must be binary 0/1; got {sorted(frame['Y'].unique())}")
    stats = {
        "raw_rows": int(len(raw)), "invalid_rows_removed": invalid,
        "identical_duplicate_rows_collapsed": int(before_dedup - len(frame)),
        "conflicting_duplicate_groups": int(len(conflicts)),
        "conflicting_duplicate_rows_removed": conflict_rows,
    }
    return frame, stats


def prepare_one(key: str, spec: dict, cfg: dict) -> dict:
    dataset = LOADERS[spec["loader"]](name=spec["name"])
    cleaned, stats = clean(dataset.get_data())
    split = create_scaffold_split(
        cleaned, seed=cfg["study"]["split_seed"],
        frac=cfg["study"]["split_fractions"], entity="canonical_smiles",
    )
    assignments = {}
    for split_name, part in split.items():
        for source_row in part["source_row"]:
            assignments[int(source_row)] = split_name
    cleaned["split"] = cleaned["source_row"].map(assignments)
    if cleaned["split"].isna().any():
        raise AssertionError("TDC scaffold split omitted cleaned molecules")
    cleaned["scaffold"] = cleaned["canonical_smiles"].map(scaffold)
    pool = cleaned[cleaned["split"].isin(["train", "valid"])]
    test = cleaned[cleaned["split"] == "test"]
    overlap = set(pool["scaffold"]) & set(test["scaffold"])
    if overlap:
        raise AssertionError(f"Pool/test scaffold leakage: {len(overlap)} scaffolds")
    for name, part in (("pool", pool), ("test", test)):
        if set(part["Y"].unique()) != {0, 1}:
            raise AssertionError(f"{key} {name} does not contain both classes")
    fp_cfg = cfg["fingerprint"]
    fps = np.stack([fingerprint(s, fp_cfg["radius"], fp_cfg["n_bits"])
                    for s in cleaned["canonical_smiles"]])
    if fps.dtype != np.uint8 or fps.shape[1] != 1024 or not np.isin(fps, [0, 1]).all():
        raise AssertionError("Fingerprint acceptance test failed")
    descriptor_rows = [rdkit2d(s) for s in cleaned["canonical_smiles"]]
    descriptor_names = descriptor_rows[0][1]
    descriptors = np.stack([row[0] for row in descriptor_rows])
    pool_mask = cleaned["split"].isin(["train", "valid"]).to_numpy()
    medians = np.nanmedian(descriptors[pool_mask], axis=0)
    medians[~np.isfinite(medians)] = 0.0
    descriptors = np.where(np.isfinite(descriptors), descriptors, medians)
    means = descriptors[pool_mask].mean(axis=0)
    scales = descriptors[pool_mask].std(axis=0)
    scales[scales == 0] = 1.0
    descriptors = ((descriptors - means) / scales).astype(np.float32)
    if not np.isfinite(descriptors).all() or descriptors.shape[1] != len(descriptor_names):
        raise AssertionError("RDKit2D acceptance test failed")
    molecule_ids = np.array([f"{key}:{i}" for i in range(len(cleaned))])
    out = Path(cfg["paths"]["processed_dir"]) / f"{key}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, fingerprints=fps, rdkit2d=descriptors, rdkit2d_names=descriptor_names,
        labels=cleaned["Y"].to_numpy(np.int8),
        canonical_smiles=cleaned["canonical_smiles"].to_numpy(str),
        molecule_ids=molecule_ids, split=cleaned["split"].to_numpy(str),
        scaffolds=cleaned["scaffold"].to_numpy(str),
    )
    stats.update({
        "final_samples": int(len(cleaned)),
        "positive_prevalence": float(cleaned["Y"].mean()),
        "pool_samples": int(len(pool)), "test_samples": int(len(test)),
        "scaffold_count": int(cleaned["scaffold"].nunique()),
        "pool_scaffold_count": int(pool["scaffold"].nunique()),
        "test_scaffold_count": int(test["scaffold"].nunique()),
        "rdkit2d_features": int(descriptors.shape[1]),
    })
    write_json(out.with_suffix(".json"), stats)
    return {"dataset": key, **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_representation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = [prepare_one(key, spec, cfg) for key, spec in cfg["datasets"].items()]
    pd.DataFrame(rows).to_csv(Path(cfg["paths"]["processed_dir"]) / "dataset_stats.csv", index=False)


if __name__ == "__main__":
    main()
