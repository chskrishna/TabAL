from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from study import config_hash, hardware, load_config, load_processed, package_versions, sha256, write_json


BASE_MODELS = ("tabpfn", "random_forest", "xgboost")
MODELS = BASE_MODELS + ("lightgbm",)
ACQUISITIONS = ("random", "entropy", "entropy_diversity")
REPRESENTATIONS = ("ecfp4", "rdkit2d")


def select_seeds(requested: list[int] | None, configured: list[int], smoke: bool) -> list[int]:
    return requested if requested is not None else configured[:1] if smoke else configured


def initial_and_random_order(labels: np.ndarray, seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels))
    while True:
        initial = rng.choice(all_indices, size=size, replace=False)
        if np.unique(labels[initial]).size == 2:
            break
    remaining = np.setdiff1d(all_indices, initial, assume_unique=False)
    return initial, rng.permutation(remaining)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), np.finfo(float).eps,
                1 - np.finfo(float).eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def seeded_rank(indices: np.ndarray, scores: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tie_keys = rng.random(len(indices))
    return indices[np.lexsort((indices, tie_keys, -scores))]


def tanimoto_to_many(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    intersection = matrix @ vector
    union = matrix.sum(axis=1) + vector.sum() - intersection
    return np.divide(intersection, union, out=np.ones(len(matrix), dtype=float), where=union != 0)


def entropy_diversity(
    unlabeled: np.ndarray, probabilities: np.ndarray, fingerprints: np.ndarray,
    labeled: np.ndarray, batch_size: int, shortlist_size: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ent = entropy(probabilities)
    shortlist = seeded_rank(unlabeled, ent, seed)[:shortlist_size]
    entropy_by_index = dict(zip(unlabeled.tolist(), ent.tolist()))
    rng = np.random.default_rng(seed)
    tie_key = {int(i): float(v) for i, v in zip(shortlist, rng.random(len(shortlist)))}
    selected, scores = [], []
    reference = fingerprints[labeled]
    candidates = shortlist.tolist()
    while len(selected) < min(batch_size, len(candidates) + len(selected)):
        best = None
        for idx in candidates:
            min_distance = float(np.min(1 - tanimoto_to_many(fingerprints[idx], reference)))
            key = (min_distance, -tie_key[idx], -idx)
            if best is None or key > best[0]:
                best = (key, idx, min_distance)
        assert best is not None
        _, idx, score = best
        selected.append(idx)
        scores.append(score)
        candidates.remove(idx)
        reference = np.vstack([reference, fingerprints[idx]])
    chosen = np.asarray(selected, dtype=int)
    if not set(chosen) <= set(shortlist):
        raise AssertionError("Hybrid acquisition escaped entropy shortlist")
    return chosen, np.asarray(scores), np.asarray([entropy_by_index[int(i)] for i in chosen])


def mean_pairwise_similarity(batch_fps: np.ndarray) -> float:
    if len(batch_fps) < 2:
        return 0.0
    values = []
    for i in range(len(batch_fps)):
        values.extend(tanimoto_to_many(batch_fps[i], batch_fps[i + 1:]).tolist())
    return float(np.mean(values))


def make_model(name: str, cfg: dict, seed: int):
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **cfg["models"][name])
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, **cfg["models"][name])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, **cfg["models"][name])
    if name == "tabpfn":
        import torch
        from tabpfn import TabPFNClassifier
        params = cfg["models"][name]
        checkpoint = Path(cfg["paths"]["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Exact synthetic-only checkpoint missing: {checkpoint}")
        actual_sha = sha256(checkpoint)
        if actual_sha != cfg["paths"]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {actual_sha}")
        device = params["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNClassifier(
            model_path=checkpoint, n_estimators=params["n_estimators"],
            balance_probabilities=params["balance_probabilities"],
            device=device, random_state=seed,
        )
    raise ValueError(name)


def predict_positive(model, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    positive = probabilities[:, int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])]
    if not np.isfinite(positive).all() or not ((0 <= positive) & (positive <= 1)).all():
        raise AssertionError("Model returned invalid probabilities")
    return positive


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "test_auroc": float(roc_auc_score(y, p)),
        "test_auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
    }


def trajectory(dataset: str, seed: int, model_name: str, acquisition: str,
               representation: str, cfg: dict):
    data = load_processed(Path(cfg["paths"]["processed_dir"]) / f"{dataset}.npz")
    pool_global = np.flatnonzero(np.isin(data["split"], ["train", "valid"]))
    test_global = np.flatnonzero(data["split"] == "test")
    feature_key = "fingerprints" if representation == "ecfp4" else "rdkit2d"
    if feature_key not in data:
        raise KeyError(f"Precomputed representation missing: {feature_key}")
    x_pool, y_pool = data[feature_key][pool_global], data["labels"][pool_global]
    x_test, y_test = data[feature_key][test_global], data["labels"][test_global]
    diversity_pool = data["fingerprints"][pool_global]
    al = cfg["active_learning"]
    labeled, random_order = initial_and_random_order(y_pool, seed, al["initial_size"])
    queried = set(labeled.tolist())
    metrics_rows, query_rows = [], []
    initial_similarity = mean_pairwise_similarity(diversity_pool[labeled])
    for idx in labeled:
        global_idx = pool_global[idx]
        query_rows.append({
            "dataset": dataset, "seed": seed, "model": model_name,
            "acquisition": acquisition, "round": 0,
            "queried_molecule_id": str(data["molecule_ids"][global_idx]),
            "acquisition_score": np.nan, "entropy": np.nan,
            "revealed_label": int(y_pool[idx]),
            "mean_within_batch_tanimoto": initial_similarity,
            "is_initial": True,
        })
    random_cursor = 0
    budgets = list(range(al["initial_size"], al["final_budget"] + 1, al["batch_size"]))
    for round_index, budget in enumerate(budgets):
        model = make_model(model_name, cfg, seed)
        started = time.perf_counter()
        model.fit(x_pool[labeled], y_pool[labeled])
        fit_runtime = time.perf_counter() - started
        started = time.perf_counter()
        test_p = predict_positive(model, x_test)
        prediction_runtime = time.perf_counter() - started
        row = {"dataset": dataset, "seed": seed, "model": model_name,
               "acquisition": acquisition, "round": round_index, "budget": budget,
               "fit_runtime": fit_runtime, "prediction_runtime": prediction_runtime,
               "acquisition_prediction_runtime": 0.0,
               **evaluate(y_test, test_p)}
        if not all(math.isfinite(v) for k, v in row.items() if isinstance(v, float)):
            raise AssertionError("Non-finite metric")
        metrics_rows.append(row)
        if budget == al["final_budget"]:
            break
        unlabeled = np.asarray([i for i in range(len(y_pool)) if i not in queried], dtype=int)
        if acquisition == "random":
            batch = random_order[random_cursor:random_cursor + al["batch_size"]]
            random_cursor += al["batch_size"]
            scores = np.arange(len(batch), 0, -1, dtype=float)
            entropies = np.full(len(batch), np.nan)
        else:
            started = time.perf_counter()
            pool_p = predict_positive(model, x_pool[unlabeled])
            row["acquisition_prediction_runtime"] = time.perf_counter() - started
            ent = entropy(pool_p)
            if acquisition == "entropy":
                batch = seeded_rank(unlabeled, ent, seed + round_index)[:al["batch_size"]]
                score_map = dict(zip(unlabeled.tolist(), ent.tolist()))
                scores = np.asarray([score_map[int(i)] for i in batch])
                entropies = scores.copy()
            else:
                batch, scores, entropies = entropy_diversity(
                    unlabeled, pool_p, diversity_pool, labeled, al["batch_size"],
                    min(al["entropy_shortlist"], len(unlabeled)), seed + round_index,
                )
        if len(batch) != al["batch_size"] or queried.intersection(batch.tolist()):
            raise AssertionError("Invalid acquisition batch")
        similarity = mean_pairwise_similarity(diversity_pool[batch])
        for idx, score, ent_score in zip(batch, scores, entropies):
            global_idx = pool_global[idx]
            query_rows.append({
                "dataset": dataset, "seed": seed, "model": model_name,
                "acquisition": acquisition, "round": round_index + 1,
                "queried_molecule_id": str(data["molecule_ids"][global_idx]),
                "acquisition_score": float(score), "entropy": float(ent_score),
                "revealed_label": int(y_pool[idx]),
                "mean_within_batch_tanimoto": similarity,
                "is_initial": False,
            })
        queried.update(batch.tolist())
        labeled = np.concatenate([labeled, batch])
    return pd.DataFrame(metrics_rows), pd.DataFrame(query_rows)


def append_checkpoint(path: Path, new: pd.DataFrame) -> None:
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        keys = ["dataset", "seed", "model", "acquisition"]
        replaced = pd.MultiIndex.from_frame(new[keys].drop_duplicates())
        old = old[~pd.MultiIndex.from_frame(old[keys]).isin(replaced)]
    combined = pd.concat([old, new], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def write_manifest(config_path: str, cfg: dict, representation: str, prefix: str) -> None:
    checkpoint = Path(cfg["paths"]["checkpoint"])
    write_json(Path(cfg["paths"]["results_dir"]) / f"{prefix}manifest.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_hash": config_hash(config_path),
        "representation": representation,
        "package_versions": package_versions(), "hardware": hardware(),
        "tabpfn_checkpoint_identity": checkpoint.name,
        "tabpfn_checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_representation.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run two acquisition rounds (budgets 32,48,64)")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*", choices=MODELS)
    parser.add_argument("--acquisitions", nargs="*", choices=ACQUISITIONS)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS,
                        default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["active_learning"]["final_budget"] = 64
    datasets = args.datasets or list(cfg["datasets"])
    representations = args.representations or ["ecfp4"]
    models = args.models or list(MODELS)
    acquisitions = args.acquisitions or list(ACQUISITIONS)
    seeds = select_seeds(args.seeds, cfg["active_learning"]["seeds"], args.smoke)
    results = Path(cfg["paths"]["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    for representation in representations:
        prefix = f"{representation}_"
        smoke = "smoke_" if args.smoke else ""
        metrics_path = results / f"{prefix}{smoke}metrics.csv"
        queries_path = results / f"{prefix}{smoke}queries.csv"
        complete = set()
        if metrics_path.exists():
            prior = pd.read_csv(metrics_path)
            expected = 3 if args.smoke else 15
            counts = prior.groupby(["dataset", "seed", "model", "acquisition"]).size()
            complete = set(counts[counts == expected].index)
            if queries_path.exists():
                prior_queries = pd.read_csv(queries_path)
                query_counts = prior_queries.groupby(
                    ["dataset", "seed", "model", "acquisition"]
                ).size()
                complete &= set(query_counts[
                    query_counts == cfg["active_learning"]["final_budget"]
                ].index)
            else:
                complete = set()
        for dataset in datasets:
            for seed in seeds:
                for model in models:
                    for acquisition in acquisitions:
                        key = (dataset, seed, model, acquisition)
                        if key in complete:
                            continue
                        metric_rows, query_rows = trajectory(
                            dataset, seed, model, acquisition, representation, cfg
                        )
                        append_checkpoint(metrics_path, metric_rows)
                        append_checkpoint(queries_path, query_rows)
                        print(f"completed {(representation,) + key}", flush=True)
        write_manifest(args.config, cfg, representation, prefix)


if __name__ == "__main__":
    main()
