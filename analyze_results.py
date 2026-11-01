from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from run_al import ACQUISITIONS, MODELS
from study import load_config, write_json


EXPECTED_BUDGETS = list(range(32, 257, 16))


def validate(metrics: pd.DataFrame, queries: pd.DataFrame, datasets: list[str],
             seeds: list[int], budgets: list[int], models: tuple[str, ...]) -> None:
    keys = ["dataset", "seed", "model", "acquisition"]
    expected = len(datasets) * len(seeds) * len(models) * len(ACQUISITIONS)
    groups = metrics.groupby(keys, sort=False)
    if groups.ngroups != expected:
        raise AssertionError(f"Expected {expected} trajectories, found {groups.ngroups}")
    for key, group in groups:
        if sorted(group["budget"].tolist()) != budgets:
            raise AssertionError(f"Bad budgets for {key}")
    metric_columns = ["test_auroc", "test_auprc", "brier", "nll",
                      "fit_runtime", "prediction_runtime", "acquisition_prediction_runtime"]
    if not np.isfinite(metrics[metric_columns].to_numpy()).all():
        raise AssertionError("Non-finite metrics")
    if not metrics["test_auroc"].between(0, 1).all():
        raise AssertionError("Invalid AUROC")
    qgroups = queries.groupby(keys, sort=False)
    if qgroups.ngroups != expected:
        raise AssertionError("Missing query histories")
    for key, group in qgroups:
        if len(group) != budgets[-1] or group["queried_molecule_id"].duplicated().any():
            raise AssertionError(f"Invalid query history for {key}")
        acquired = group[group["round"] > 0]
        if len(group[group["round"] == 0]) != 32 or not (acquired.groupby("round").size() == 16).all():
            raise AssertionError(f"Invalid query batch for {key}")


def add_aulc(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "seed", "model", "acquisition"]
    rows = []
    for key, group in metrics.groupby(keys, sort=False):
        group = group.sort_values("budget")
        rows.append(dict(zip(keys, key)) | {
            "aulc": float(np.trapz(group["test_auroc"], group["budget"]) / 224),
            "final_auroc": float(group.iloc[-1]["test_auroc"]),
            "macro_auprc_over_budgets": float(group["test_auprc"].mean()),
            "macro_brier_over_budgets": float(group["brier"].mean()),
            "macro_nll_over_budgets": float(group["nll"].mean()),
            "total_fit_runtime": float(group["fit_runtime"].sum()),
            "total_prediction_runtime": float(group["prediction_runtime"].sum()),
            "total_acquisition_prediction_runtime": float(group["acquisition_prediction_runtime"].sum()),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(frame: pd.DataFrame, left: tuple[str, str], right: tuple[str, str],
                     samples: int, seed: int) -> dict:
    index = ["dataset", "seed"]
    a = frame[(frame.model == left[0]) & (frame.acquisition == left[1])].set_index(index)["aulc"]
    b = frame[(frame.model == right[0]) & (frame.acquisition == right[1])].set_index(index)["aulc"]
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(paired) != 50:
        raise AssertionError(f"Expected 50 paired dataset/seed observations, got {len(paired)}")
    differences = (paired["a"] - paired["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (samples, len(differences)))].mean(axis=1)
    return {
        "left_model": left[0], "left_acquisition": left[1],
        "right_model": right[0], "right_acquisition": right[1],
        "mean_paired_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)), "n_pairs": len(differences),
    }


def comparisons(aulc: pd.DataFrame, cfg: dict, models: tuple[str, ...]) -> pd.DataFrame:
    pairs = []
    for acquisition in ACQUISITIONS:
        pairs += [(('tabpfn', acquisition), (model, acquisition))
                  for model in models if model != 'tabpfn']
    for model in models:
        pairs += [((model, 'entropy'), (model, 'random')),
                  ((model, 'entropy_diversity'), (model, 'random')),
                  ((model, 'entropy_diversity'), (model, 'entropy'))]
    return pd.DataFrame([
        paired_bootstrap(aulc, left, right, cfg["evaluation"]["bootstrap_samples"],
                         cfg["evaluation"]["bootstrap_seed"])
        for left, right in pairs
    ])


def make_figures(metrics: pd.DataFrame, comparisons_df: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    datasets = metrics["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3.2), sharey=True)
    for ax, dataset in zip(axes, datasets):
        part = metrics[metrics.dataset == dataset]
        sns.lineplot(data=part, x="budget", y="test_auroc", hue="model",
                     style="acquisition", estimator="mean", errorbar=("ci", 95), ax=ax)
        ax.set_title(dataset)
        if ax is not axes[0]:
            ax.get_legend().remove()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure1_learning_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    acquisition = comparisons_df[
        comparisons_df.left_model == comparisons_df.right_model
    ].copy()
    labels = (acquisition.left_model + ": " + acquisition.left_acquisition +
              " − " + acquisition.right_acquisition)
    y = np.arange(len(acquisition))
    fig, ax = plt.subplots(figsize=(7, 5))
    x = acquisition.mean_paired_difference.to_numpy()
    ax.errorbar(x, y, xerr=[x - acquisition.ci95_low, acquisition.ci95_high - x], fmt="o")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(yticks=y, yticklabels=labels, xlabel="Paired AULC difference")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure2_acquisition_differences.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_representation.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--representation", choices=("ecfp4", "rdkit2d"), default="ecfp4")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = Path(cfg["paths"]["results_dir"])
    representation = args.representation
    models = MODELS
    rep_prefix = f"{args.representation}_"
    prefix = rep_prefix + ("smoke_" if args.smoke else "")
    metrics = pd.read_csv(results / f"{prefix}metrics.csv")
    queries = pd.read_csv(results / f"{prefix}queries.csv")
    budgets = [32, 48, 64] if args.smoke else EXPECTED_BUDGETS
    seeds = [0] if args.smoke else cfg["active_learning"]["seeds"]
    validate(metrics, queries, list(cfg["datasets"]), seeds, budgets, models)
    if args.validate_only:
        output = results / f"{prefix}validation.json"
        write_json(output, {"valid": True,
                            "representation": representation,
                            "trajectories": len(cfg["datasets"]) * len(seeds) * len(models) * len(ACQUISITIONS),
                            "budgets_per_trajectory": len(budgets)})
        print(f"validated {output}")
        return
    aulc = add_aulc(metrics)
    comparison_table = comparisons(aulc, cfg, models)
    suffix = f"_{args.representation}"
    tables, figures = results / f"tables{suffix}", results / f"figures{suffix}"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    per_dataset = aulc.groupby(["dataset", "model", "acquisition"], as_index=False).aulc.mean()
    macro = aulc.groupby(["model", "acquisition"], as_index=False).aulc.mean()
    ranks = per_dataset.copy()
    ranks["rank"] = ranks.groupby("dataset")["aulc"].rank(ascending=False, method="average")
    summary = macro.merge(ranks.groupby(["model", "acquisition"], as_index=False)["rank"].mean(),
                          on=["model", "acquisition"])
    winners = per_dataset.loc[per_dataset.groupby("dataset").aulc.idxmax()]
    win_counts = winners.groupby(["model", "acquisition"]).size().rename("dataset_wins").reset_index()
    summary = summary.merge(win_counts, how="left", on=["model", "acquisition"]).fillna({"dataset_wins": 0})
    diversity = queries[~queries["is_initial"]].groupby(
        ["dataset", "seed", "model", "acquisition"], as_index=False)[
        "mean_within_batch_tanimoto"].mean()
    outputs = {"aulc_trajectories": aulc, "per_dataset_aulc": per_dataset,
               "macro_summary": summary, "paired_comparisons": comparison_table,
               "query_diversity": diversity}
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
        (tables / f"{name}.tex").write_text(frame.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    make_figures(metrics, comparison_table, figures)
    write_json(results / f"{prefix}validation.json", {
        "valid": True, "representation": representation,
        "trajectories": len(cfg["datasets"]) * len(seeds) * len(models) * len(ACQUISITIONS),
        "budgets_per_trajectory": len(budgets),
    })


if __name__ == "__main__":
    main()
