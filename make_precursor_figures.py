from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODELS = ("tabpfn", "random_forest", "xgboost", "lightgbm")
MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
MODEL_COLORS = {
    "tabpfn": "#0077BB",
    "random_forest": "#EE7733",
    "xgboost": "#009988",
    "lightgbm": "#CC3311",
}
POLICIES = ("random", "entropy", "entropy_diversity")
POLICY_LABELS = {
    "random": "Random",
    "entropy": "Entropy",
    "entropy_diversity": "Entropy + diversity",
}
POLICY_COLORS = {
    "random": "#666666",
    "entropy": "#0077BB",
    "entropy_diversity": "#EE7733",
}
POLICY_STYLES = {"random": "-", "entropy": "--", "entropy_diversity": ":"}
REPRESENTATIONS = ("ecfp4", "rdkit2d")
REP_LABELS = {"ecfp4": "ECFP4", "rdkit2d": "RDKit2D"}
KEYS = ["dataset", "seed", "model", "acquisition"]
EXPECTED_BUDGETS = list(range(32, 257, 16))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_metrics(results: Path, representation: str) -> pd.DataFrame:
    frame = pd.read_csv(results / f"{representation}_metrics.csv")
    required = set(KEYS + ["budget", "test_auroc", "fit_runtime",
                           "prediction_runtime", "acquisition_prediction_runtime"])
    if not required <= set(frame.columns):
        raise ValueError(f"{representation}: missing columns {sorted(required - set(frame.columns))}")
    groups = frame.groupby(KEYS, sort=False)
    if groups.ngroups != 600:
        raise ValueError(f"{representation}: expected 600 trajectories, found {groups.ngroups}")
    bad = [key for key, group in groups if sorted(group["budget"].tolist()) != EXPECTED_BUDGETS]
    if bad:
        raise ValueError(f"{representation}: invalid budget sequence for {bad[0]}")
    return frame


def trajectory_aulc(metrics: pd.DataFrame, representation: str) -> pd.DataFrame:
    rows = []
    for key, group in metrics.groupby(KEYS, sort=False):
        group = group.sort_values("budget")
        rows.append(dict(zip(KEYS, key)) | {
            "representation": representation,
            "aulc": float(np.trapz(group["test_auroc"], group["budget"]) / 224),
        })
    return pd.DataFrame(rows)


def paired_aulc(aulc: pd.DataFrame) -> pd.DataFrame:
    wide = aulc.pivot(index=KEYS, columns="representation", values="aulc").reset_index()
    if len(wide) != 600 or wide[list(REPRESENTATIONS)].isna().any().any():
        raise ValueError("Representations do not contain the same 600 trajectories")
    wide["delta"] = wide["rdkit2d"] - wide["ecfp4"]
    return wide


def bootstrap_mean(values: np.ndarray, seed: int, samples: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def effect_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    units = paired.groupby(["dataset", "seed"], as_index=False)["delta"].mean()
    mean, low, high = bootstrap_mean(units["delta"].to_numpy(), 42)
    rows.append({"level": "overall", "model": "overall", "acquisition": "all",
                 "mean": mean, "ci95_low": low, "ci95_high": high, "n_pairs": len(units)})
    for index, model in enumerate(MODELS, start=1):
        units = paired[paired.model == model].groupby(
            ["dataset", "seed"], as_index=False
        )["delta"].mean()
        mean, low, high = bootstrap_mean(units["delta"].to_numpy(), 42 + index)
        rows.append({"level": "model", "model": model, "acquisition": "all",
                     "mean": mean, "ci95_low": low, "ci95_high": high, "n_pairs": len(units)})
        for policy_index, policy in enumerate(POLICIES, start=1):
            values = paired[(paired.model == model) & (paired.acquisition == policy)]["delta"]
            mean, low, high = bootstrap_mean(values.to_numpy(), 100 + 10 * index + policy_index)
            rows.append({"level": "model_policy", "model": model, "acquisition": policy,
                         "mean": mean, "ci95_low": low, "ci95_high": high,
                         "n_pairs": len(values)})
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    plt.close(fig)


def representation_overview(paired: pd.DataFrame, effects: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(7.2, 2.65), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.2, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    forest = effects[effects.level.isin(["overall", "model"])].copy()
    order = ["overall", *MODELS]
    forest["order"] = forest.model.map({name: i for i, name in enumerate(order)})
    forest = forest.sort_values("order", ascending=False)
    y = np.arange(len(forest))
    for ypos, row in zip(y, forest.itertuples()):
        color = "#000000" if row.model == "overall" else MODEL_COLORS[row.model]
        marker = "D" if row.model == "overall" else "o"
        ax.errorbar(row.mean, ypos,
                    xerr=[[row.mean - row.ci95_low], [row.ci95_high - row.mean]],
                    fmt=marker, color=color, markersize=4.5, capsize=2, linewidth=1.1)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, ["Overall" if name == "overall" else MODEL_LABELS[name]
                      for name in forest.model])
    ax.set_xlabel("RDKit2D − ECFP4 AULC")
    ax.set_title("a  Representation gain", loc="left", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    heat = effects[effects.level == "model_policy"].pivot(
        index="model", columns="acquisition", values="mean"
    ).reindex(index=MODELS, columns=POLICIES)
    image = ax.imshow(heat.to_numpy(), cmap="viridis", vmin=0, vmax=0.10, aspect="auto")
    for i in range(len(MODELS)):
        for j in range(len(POLICIES)):
            value = heat.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > 0.065 else "black", fontsize=7)
    ax.set_xticks(range(len(POLICIES)), ["Random", "Entropy", "Ent. + div."], rotation=25, ha="right")
    ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.set_title("b  Gain by learner and policy", loc="left", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("AULC gain", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(grid[0, 2])
    model_means = paired.groupby("model")[["ecfp4", "rdkit2d"]].mean().reindex(MODELS)
    label_offsets = {"random_forest": 0.0025, "tabpfn": 0.0005,
                     "lightgbm": -0.0015, "xgboost": -0.0030}
    for model, row in model_means.iterrows():
        ax.plot([0, 1], row, marker="o", color=MODEL_COLORS[model], linewidth=1.4,
                markersize=4, label=MODEL_LABELS[model])
        ax.text(1.035, row.rdkit2d + label_offsets[model], MODEL_LABELS[model], color=MODEL_COLORS[model],
                va="center", fontsize=7)
    ax.set_xticks([0, 1], ["ECFP4", "RDKit2D"])
    ax.set_xlim(-0.12, 1.55)
    ax.set_ylim(0.74, 0.87)
    ax.set_ylabel("Macro AULC")
    ax.set_title("c  Model ranking", loc="left", fontweight="bold")
    save_figure(fig, out, "figure_1_representation_overview_precursor")


def learning_curves(metrics_by_rep: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.25), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, representation in enumerate(REPRESENTATIONS):
        data = metrics_by_rep[representation]
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            for policy_index, policy in enumerate(POLICIES):
                subset = data[(data.model == model) & (data.acquisition == policy)]
                means, lows, highs = [], [], []
                for budget in EXPECTED_BUDGETS:
                    values = subset[subset.budget == budget]["test_auroc"].to_numpy()
                    mean, low, high = bootstrap_mean(
                        values, seed=1000 + row * 500 + col * 50 + policy_index * 17 + budget,
                        samples=2_000,
                    )
                    means.append(mean); lows.append(low); highs.append(high)
                color = POLICY_COLORS[policy]
                ax.plot(EXPECTED_BUDGETS, means, color=color, linestyle=POLICY_STYLES[policy],
                        linewidth=1.35)
                ax.fill_between(EXPECTED_BUDGETS, lows, highs, color=color, alpha=0.10,
                                linewidth=0)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{REP_LABELS[representation]}\nMean test AUROC")
            if row == 1:
                ax.set_xlabel("Label budget")
            ax.set_ylim(0.67, 0.93)
            ax.set_xticks([32, 96, 160, 224])
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    handles = [Line2D([0], [0], color=POLICY_COLORS[p], linestyle=POLICY_STYLES[p],
                      linewidth=1.5, label=POLICY_LABELS[p]) for p in POLICIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Learning curves by representation, learner, and acquisition policy",
                 fontsize=10, fontweight="bold")
    save_figure(fig, out, "figure_2_learning_curves_precursor")


def acquisition_effects(aulc: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, sharey=True,
                             constrained_layout=True)
    comparisons = ("entropy", "entropy_diversity")
    offsets = {"entropy": -0.12, "entropy_diversity": 0.12}
    markers = {"entropy": "o", "entropy_diversity": "s"}
    for ax, representation in zip(axes, REPRESENTATIONS):
        rep = aulc[aulc.representation == representation]
        for model_index, model in enumerate(MODELS):
            base = rep[(rep.model == model) & (rep.acquisition == "random")].set_index(
                ["dataset", "seed"]
            )["aulc"]
            for policy_index, policy in enumerate(comparisons):
                informed = rep[(rep.model == model) & (rep.acquisition == policy)].set_index(
                    ["dataset", "seed"]
                )["aulc"]
                values = (informed - base).dropna().to_numpy()
                mean, low, high = bootstrap_mean(
                    values, seed=2000 + model_index * 10 + policy_index + 100 * REPRESENTATIONS.index(representation)
                )
                y = model_index + offsets[policy]
                ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]],
                            fmt=markers[policy], color=POLICY_COLORS[policy], markersize=4,
                            capsize=2, linewidth=1.1)
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set_xlabel("AULC difference vs random")
        ax.set_title(REP_LABELS[representation], fontweight="bold")
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    handles = [Line2D([0], [0], color=POLICY_COLORS[p], marker=markers[p], linestyle="none",
                      label=POLICY_LABELS[p]) for p in comparisons]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Acquisition-policy effects are smaller and model-dependent",
                 fontsize=10, fontweight="bold")
    save_figure(fig, out, "figure_3_acquisition_effects_precursor")


def runtime_supplement(metrics_by_rep: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, constrained_layout=True)
    for ax, representation in zip(axes, REPRESENTATIONS):
        frame = metrics_by_rep[representation].copy()
        frame["runtime"] = (frame["fit_runtime"] + frame["prediction_runtime"]
                            + frame["acquisition_prediction_runtime"])
        totals = frame.groupby(KEYS, as_index=False)["runtime"].sum()
        positions, values, colors = [], [], []
        for model_index, model in enumerate(MODELS):
            for policy_index, policy in enumerate(POLICIES):
                positions.append(model_index + (policy_index - 1) * 0.22)
                values.append(totals[(totals.model == model) &
                                     (totals.acquisition == policy)]["runtime"].to_numpy())
                colors.append(POLICY_COLORS[policy])
        boxes = ax.boxplot(values, positions=positions, widths=0.18, patch_artist=True,
                           showfliers=False, medianprops={"color": "black", "linewidth": 0.9})
        for box, color in zip(boxes["boxes"], colors):
            box.set(facecolor=color, edgecolor=color, alpha=0.65, linewidth=0.7)
        ax.set_yscale("log")
        ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS], rotation=20, ha="right")
        ax.set_title(REP_LABELS[representation], fontweight="bold")
        ax.set_xlabel("Learner")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Total trajectory runtime (s, log scale)")
    handles = [Line2D([0], [0], color=POLICY_COLORS[p], linewidth=5,
                      label=POLICY_LABELS[p]) for p in POLICIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Observed runtime on the archived RTX A4000 host",
                 fontsize=10, fontweight="bold")
    save_figure(fig, out, "figure_s1_runtime_precursor")


def write_captions(out: Path, effects: pd.DataFrame) -> None:
    overall = effects[effects.level == "overall"].iloc[0]
    text = f"""# Figure captions (single-split precursor)

These figures summarize the archived 1,200-trajectory precursor dataset. They
must not be presented as the final three-split, 2,700-trajectory paper results.

**Figure 1. Representation effects on active-learning efficiency.** (a) Paired
RDKit2D-minus-ECFP4 normalized AULC differences, averaged within each of 50
dataset–seed units; intervals are 95% paired bootstrap intervals. The overall
precursor gain is {overall['mean']:.4f} [{overall['ci95_low']:.4f},
{overall['ci95_high']:.4f}]. (b) Mean paired gain for every learner–policy
combination. (c) Macro AULC averaged over acquisition policies. Positive values
favor RDKit2D.

**Figure 2. Learning curves by representation, learner, and acquisition
policy.** Lines show mean test AUROC across 50 dataset–seed units; shaded bands
show 95% bootstrap intervals. Label budgets range from 32 to 256.

**Figure 3. Acquisition-policy effects relative to random acquisition.** Points
show paired normalized AULC differences for entropy and entropy-plus-diversity
relative to random acquisition; intervals are 95% paired bootstrap intervals
over 50 dataset–seed units. Positive values favor the informed policy.

**Figure S1. Runtime of complete active-learning trajectories.** Box plots show
the distribution of total recorded fit, prediction, and acquisition-scoring
time over dataset–seed trajectories. The logarithmic axis reflects the large
model-dependent runtime range. Measurements are specific to the archived
Windows/NVIDIA RTX A4000 execution environment.
"""
    (out / "CAPTIONS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures_precursor"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    metrics = {rep: load_metrics(args.results, rep) for rep in REPRESENTATIONS}
    aulc = pd.concat([trajectory_aulc(metrics[rep], rep) for rep in REPRESENTATIONS],
                     ignore_index=True)
    paired = paired_aulc(aulc)
    effects = effect_summary(paired)
    print(f"validated {len(paired)} paired trajectories; mean gain={paired.delta.mean():.6f}")
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    representation_overview(paired, effects, args.output)
    learning_curves(metrics, args.output)
    acquisition_effects(aulc, args.output)
    runtime_supplement(metrics, args.output)
    effects.to_csv(args.output / "representation_effects.csv", index=False)
    write_captions(args.output, effects)
    print(f"wrote figures to {args.output}")


if __name__ == "__main__":
    main()
