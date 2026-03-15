"""
Results Analysis Script
=======================
Generates tables, visualisations, and statistical tests from multi-run
experiment results for AML model comparison.

Toggle sections on/off via the SECTIONS dict below.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path(r"C:\Users\lambe\Desktop\Results master")
OUTPUT_DIR = Path(r"C:\Users\lambe\Desktop\Results_analysis\output")

DATASETS = [
    "AMLSim",
    "Elliptic",
    "IBM_AML_HiSmall",
    "IBM_AML_LiSmall",
    "IBM_AML_HiMedium",
    "IBM_AML_LiMedium",
]

MODELS = ["MLP", "GCN", "GAT", "GIN", "RF", "SVM", "XGB"]

# All scalar metrics stored per run
METRIC_COLS = [
    "accuracy",
    "precision",
    "precision_illicit",
    "recall",
    "recall_illicit",
    "f1",
    "f1_illicit",
    "roc_auc",
    "PRAUC",
    "kappa",
]

# Subset used for focused visualisations (box/violin/heatmap/rank/CD)
KEY_METRICS = ["f1_illicit", "PRAUC", "roc_auc", "kappa"]

# Metrics shown on radar charts
RADAR_METRICS = [
    "accuracy",
    "precision_illicit",
    "recall_illicit",
    "f1_illicit",
    "roc_auc",
    "kappa",
]

N_RUNS = 30  # expected runs per model-dataset combination

# ── Toggle individual sections ────────────────────────────────────────────────
SECTIONS = {
    "summary_tables": True,  # 1  Mean (std) tables per dataset
    "pr_curves": True,  # 2  PR curves per model-dataset
    "confusion_matrices": True,  # 3  Confusion matrices
    "box_plots": True,  # 4  Box plots of metric distributions
    "heatmap": True,  # 5  Cross-dataset performance heatmap
    "violin_plots": True,  # 6  Violin plots
    "statistical_tests": True,  # 7  Friedman + Wilcoxon + Nemenyi
    "cv_analysis": True,  # 8  Coefficient of variation
    "rank_analysis": True,  # 9  Rank stability across runs
    "radar_charts": True,  # 10 Radar / spider charts
    "critical_difference": True,  # 11 Critical difference diagrams
    "convergence": True,  # 12 Metric convergence over runs
}

# ── Plot aesthetics ──────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)

MODEL_COLOURS = {
    "MLP": "#1f77b4",
    "GCN": "#ff7f0e",
    "GAT": "#2ca02c",
    "GIN": "#d62728",
    "RF": "#9467bd",
    "SVM": "#8c564b",
    "XGB": "#e377c2",
}

# Nemenyi critical values q_{alpha=0.05, k} for k groups (infinity df)
_NEMENYI_Q = {
    2: 1.960,
    3: 2.344,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


def _load_pkl(path):
    """Load a pickle file; return None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_run_metrics():
    """Load scalar metrics for every run into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: [dataset, model, run] + METRIC_COLS
    """
    records = []
    for ds in DATASETS:
        for model in MODELS:
            for run in range(1, N_RUNS + 1):
                path = (
                    RESULTS_DIR
                    / ds
                    / model
                    / "pkl_files"
                    / f"{model}_run_{run}_metrics.pkl"
                )
                m = _load_pkl(path)
                if m is None:
                    continue
                row = {k: m.get(k, np.nan) for k in METRIC_COLS}
                row.update(dataset=ds, model=model, run=run)
                records.append(row)
    df = pd.DataFrame(records)
    print(
        f"Loaded {len(df)} run records  "
        f"({df['dataset'].nunique()} datasets, {df['model'].nunique()} models)"
    )
    return df


def load_predictions(dataset, model, run):
    """Return (y_true, y_pred, y_probs) for one run.

    Tries the metrics pkl first (wrapper models store preds/y_true),
    then falls back to the pr_data pkl (threshold probs at 0.5).
    """
    # -- Try metrics pkl (wrapper models) --
    m = _load_pkl(
        RESULTS_DIR / dataset / model / "pkl_files" / f"{model}_run_{run}_metrics.pkl"
    )
    if m is not None and "y_true" in m and "preds" in m:
        y_true = np.asarray(m["y_true"]).ravel()
        y_pred = np.asarray(m["preds"]).ravel()
        y_probs = np.asarray(m.get("probs")) if "probs" in m else None
        return y_true, y_pred, y_probs

    # -- Fallback: pr_data pkl (sklearn models) --
    pr = _load_pkl(
        RESULTS_DIR / dataset / model / "pkl_files" / f"{model}_run_{run}_pr_data.pkl"
    )
    if pr is not None and "y_true" in pr and "y_probs" in pr:
        y_true = np.asarray(pr["y_true"]).ravel()
        y_probs_pos = np.asarray(pr["y_probs"]).ravel()
        y_pred = (y_probs_pos >= 0.5).astype(int)
        return y_true, y_pred, y_probs_pos

    return None, None, None


def load_pr_curve(dataset, model, run):
    """Load PR curve arrays from npz (preferred) or pr_data pkl.

    Returns (precision, recall, thresholds, auc).
    """
    npz = RESULTS_DIR / dataset / model / "pr_curves" / f"{model}_run_{run}_pr_data.npz"
    if npz.exists():
        d = np.load(npz)
        return (
            d["precision"],
            d["recall"],
            d.get("thresholds", None),
            float(d.get("auc", np.nan)),
        )

    pr = _load_pkl(
        RESULTS_DIR / dataset / model / "pkl_files" / f"{model}_run_{run}_pr_data.pkl"
    )
    if pr is not None:
        return (
            np.asarray(pr["precision"]),
            np.asarray(pr["recall"]),
            np.asarray(pr.get("thresholds")),
            float(pr.get("auc", np.nan)),
        )

    return None, None, None, np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  1. SUMMARY TABLES
# ══════════════════════════════════════════════════════════════════════════════


def generate_summary_tables(df):
    out = OUTPUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # Per-dataset tables: each cell shows "mean (std)"
    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]
        rows = []
        for model in MODELS:
            m_df = ds_df[ds_df["model"] == model]
            if m_df.empty:
                continue
            row = {"Model": model}
            for metric in METRIC_COLS:
                vals = m_df[metric].dropna()
                if len(vals) > 0:
                    row[metric] = f"{vals.mean():.4f} ({vals.std():.4f})"
                else:
                    row[metric] = "N/A"
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"{ds}_summary.csv", index=False)
        print(f"  Saved {ds}_summary.csv")

    # Overall flat table with separate mean / std columns
    all_rows = []
    for ds in DATASETS:
        for model in MODELS:
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            if sub.empty:
                continue
            row = {"Dataset": ds, "Model": model}
            for metric in METRIC_COLS:
                vals = sub[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = vals.mean()
                    row[f"{metric}_std"] = vals.std()
                    row[f"{metric}_display"] = f"{vals.mean():.4f} ({vals.std():.4f})"
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
                    row[f"{metric}_display"] = "N/A"
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(out / "overall_summary.csv", index=False)
    print("  Saved overall_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  2. PR CURVES
# ══════════════════════════════════════════════════════════════════════════════


def generate_pr_curves():
    out = OUTPUT_DIR / "pr_curves"
    out.mkdir(parents=True, exist_ok=True)

    common_recall = np.linspace(0, 1, 1001)

    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(8, 6))

        for model in MODELS:
            interp_precisions = []
            aucs = []

            for run in range(1, N_RUNS + 1):
                prec, rec, _, auc_val = load_pr_curve(ds, model, run)
                if prec is None:
                    continue
                prec = np.asarray(prec).ravel()
                rec = np.asarray(rec).ravel()

                # Sort by recall ascending for np.interp
                order = np.argsort(rec)
                rec_s = rec[order]
                prec_s = prec[order]

                # Make precision monotonically decreasing (right to left)
                # standard for PR curves: precision = max(precision[i:]) at each recall
                prec_s = np.maximum.accumulate(prec_s[::-1])[::-1]

                interp_prec = np.interp(common_recall, rec_s, prec_s)
                interp_precisions.append(interp_prec)
                aucs.append(auc_val)

            if not interp_precisions:
                continue

            mean_prec = np.mean(interp_precisions, axis=0)
            std_prec = np.std(interp_precisions, axis=0)
            mean_auc = np.nanmean(aucs)

            c = MODEL_COLOURS.get(model, "#333333")
            ax.plot(
                common_recall,
                mean_prec,
                label=f"{model} (PRAUC={mean_auc:.4f})",
                color=c,
                linewidth=1.5,
            )
            ax.fill_between(
                common_recall,
                np.clip(mean_prec - std_prec, 0, 1),
                np.clip(mean_prec + std_prec, 0, 1),
                color=c,
                alpha=0.12,
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves \u2014 {ds}")
        ax.legend(loc="best", fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{ds}_pr_curves.png")
        plt.close(fig)
        print(f"  Saved {ds}_pr_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  3. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════


def generate_confusion_matrices():
    out = OUTPUT_DIR / "confusion_matrices"
    out.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        n_models = len(MODELS)
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for i, model in enumerate(MODELS):
            cms_norm = []

            for run in range(1, N_RUNS + 1):
                y_true, y_pred, _ = load_predictions(ds, model, run)
                if y_true is None:
                    continue
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                cms_norm.append(cm / row_sums)

            ax = axes[i]
            if not cms_norm:
                ax.set_title(f"{model}\n(no data)")
                ax.axis("off")
                continue

            mean_norm = np.mean(cms_norm, axis=0)
            std_norm = np.std(cms_norm, axis=0)

            annot = np.array(
                [
                    [
                        f"{mean_norm[r, c] * 100:.1f}%\n\u00b1{std_norm[r, c] * 100:.1f}%"
                        for c in range(2)
                    ]
                    for r in range(2)
                ]
            )

            sns.heatmap(
                mean_norm,
                annot=annot,
                fmt="",
                xticklabels=["Licit", "Illicit"],
                yticklabels=["Licit", "Illicit"],
                cmap="Blues",
                vmin=0,
                vmax=1,
                ax=ax,
                cbar=False,
            )
            ax.set_title(f"{model} (n={len(cms_norm)})")
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")

        for j in range(len(MODELS), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Mean Normalised Confusion Matrices \u2014 {ds}\n"
            f"(averaged over {N_RUNS} runs)",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out / f"{ds}_confusion_matrices.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {ds}_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
#  4. BOX PLOTS
# ══════════════════════════════════════════════════════════════════════════════


def generate_box_plots(df):
    out = OUTPUT_DIR / "box_plots"
    out.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]
        n_m = len(KEY_METRICS)
        fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5))
        if n_m == 1:
            axes = [axes]

        for ax, metric in zip(axes, KEY_METRICS):
            plot_data, labels, colors = [], [], []
            for model in MODELS:
                vals = ds_df[ds_df["model"] == model][metric].dropna().values
                if len(vals) > 0:
                    plot_data.append(vals)
                    labels.append(model)
                    colors.append(MODEL_COLOURS.get(model, "#333"))

            bp = ax.boxplot(
                plot_data,
                labels=labels,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)

            ax.set_title(metric)
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Metric Distributions \u2014 {ds}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / f"{ds}_box_plots.png")
        plt.close(fig)
        print(f"  Saved {ds}_box_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5. CROSS-DATASET HEATMAP
# ══════════════════════════════════════════════════════════════════════════════


def generate_heatmaps(df):
    out = OUTPUT_DIR / "heatmaps"
    out.mkdir(parents=True, exist_ok=True)

    for metric in KEY_METRICS:
        pivot = df.groupby(["dataset", "model"])[metric].mean().unstack("model")
        pivot = pivot.reindex(index=DATASETS, columns=MODELS)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"Mean {metric} Across Datasets and Models")
        ax.set_ylabel("Dataset")
        ax.set_xlabel("Model")
        fig.tight_layout()
        fig.savefig(out / f"{metric}_heatmap.png")
        plt.close(fig)
        print(f"  Saved {metric}_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  6. VIOLIN PLOTS
# ══════════════════════════════════════════════════════════════════════════════


def generate_violin_plots(df):
    out = OUTPUT_DIR / "violin_plots"
    out.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]

        for metric in KEY_METRICS:
            fig, ax = plt.subplots(figsize=(10, 5))

            rows_for_plot = []
            for model in MODELS:
                vals = ds_df[ds_df["model"] == model][metric].dropna()
                for v in vals:
                    rows_for_plot.append({"Model": model, metric: v})

            if not rows_for_plot:
                plt.close(fig)
                continue

            plot_df = pd.DataFrame(rows_for_plot)
            present = [m for m in MODELS if m in plot_df["Model"].unique()]
            palette = [MODEL_COLOURS.get(m, "#333") for m in present]

            sns.violinplot(
                data=plot_df,
                x="Model",
                y=metric,
                ax=ax,
                hue="Model",
                palette=dict(zip(present, palette)),
                inner="box",
                order=present,
                legend=False,
                cut=0,
            )
            ax.set_title(f"{metric} Distribution \u2014 {ds}")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out / f"{ds}_{metric}_violin.png")
            plt.close(fig)

        print(f"  Saved violin plots for {ds}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════


def run_statistical_tests(df):
    out = OUTPUT_DIR / "statistical_tests"
    out.mkdir(parents=True, exist_ok=True)

    # ── 7a. Friedman test (cross-dataset: datasets are blocks) ────────────
    friedman_rows = []
    for metric in KEY_METRICS:
        pivot = df.groupby(["dataset", "model"])[metric].mean().unstack("model")
        pivot = pivot.reindex(columns=MODELS).dropna(axis=1, how="all")
        pivot = pivot.dropna(axis=0, how="any")

        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue

        groups = [pivot[col].values for col in pivot.columns]
        stat, p = stats.friedmanchisquare(*groups)
        friedman_rows.append(
            {
                "metric": metric,
                "test_type": "cross_dataset",
                "chi2": stat,
                "p_value": p,
                "significant_0.05": p < 0.05,
                "n_blocks": pivot.shape[0],
                "n_models": pivot.shape[1],
            }
        )

    # ── 7a-bis. Friedman within each dataset (runs are blocks) ────────────
    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]
        for metric in KEY_METRICS:
            pivot = ds_df.pivot(index="run", columns="model", values=metric)
            pivot = pivot.reindex(columns=MODELS).dropna(axis=1, how="all")
            pivot = pivot.dropna(axis=0, how="any")

            if pivot.shape[0] < 3 or pivot.shape[1] < 3:
                continue

            groups = [pivot[col].values for col in pivot.columns]
            stat, p = stats.friedmanchisquare(*groups)
            friedman_rows.append(
                {
                    "metric": metric,
                    "test_type": f"within_{ds}",
                    "chi2": stat,
                    "p_value": p,
                    "significant_0.05": p < 0.05,
                    "n_blocks": pivot.shape[0],
                    "n_models": pivot.shape[1],
                }
            )

    if friedman_rows:
        pd.DataFrame(friedman_rows).to_csv(out / "friedman_tests.csv", index=False)
        print("  Saved friedman_tests.csv")

    # ── 7b. Wilcoxon signed-rank tests (pairwise, per dataset) ────────────
    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]
        wilcox_rows = []

        for metric in KEY_METRICS:
            for m1, m2 in combinations(MODELS, 2):
                v1 = ds_df[ds_df["model"] == m1][metric].dropna().values
                v2 = ds_df[ds_df["model"] == m2][metric].dropna().values
                n = min(len(v1), len(v2))
                if n < 10:
                    continue
                v1, v2 = v1[:n], v2[:n]
                if np.all(v1 == v2):
                    continue
                try:
                    stat, p = stats.wilcoxon(v1, v2, alternative="two-sided")
                    r = 1 - (2 * stat) / (n * (n + 1) / 2)  # rank-biserial
                    wilcox_rows.append(
                        {
                            "metric": metric,
                            "model_1": m1,
                            "model_2": m2,
                            "statistic": stat,
                            "p_value": p,
                            "effect_size_r": r,
                            "mean_1": v1.mean(),
                            "mean_2": v2.mean(),
                            "n_pairs": n,
                        }
                    )
                except Exception:
                    pass

        if wilcox_rows:
            wd = pd.DataFrame(wilcox_rows)
            # Holm-Bonferroni correction
            n_tests = len(wd)
            sorted_idx = wd["p_value"].argsort()
            corrected = np.ones(n_tests)
            for rank_i, orig_i in enumerate(sorted_idx):
                corrected[orig_i] = min(
                    wd["p_value"].iloc[orig_i] * (n_tests - rank_i), 1.0
                )
            wd["p_value_holm"] = corrected
            wd["significant_holm_0.05"] = wd["p_value_holm"] < 0.05
            wd.to_csv(out / f"wilcoxon_{ds}.csv", index=False)

    print("  Saved Wilcoxon test results")

    # ── 7c. Nemenyi post-hoc (cross-dataset) ──────────────────────────────
    nemenyi_rows = []
    for metric in KEY_METRICS:
        pivot = df.groupby(["dataset", "model"])[metric].mean().unstack("model")
        pivot = pivot.reindex(columns=MODELS).dropna(axis=1, how="all")
        pivot = pivot.dropna(axis=0, how="any")

        if pivot.shape[0] < 2:
            continue

        ranks = pivot.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean(axis=0)
        k = len(mean_ranks)
        n = pivot.shape[0]
        q = _NEMENYI_Q.get(k, 3.0)
        cd = q * np.sqrt(k * (k + 1) / (6 * n))

        for m1, m2 in combinations(mean_ranks.index, 2):
            diff = abs(mean_ranks[m1] - mean_ranks[m2])
            nemenyi_rows.append(
                {
                    "metric": metric,
                    "model_1": m1,
                    "model_2": m2,
                    "mean_rank_1": mean_ranks[m1],
                    "mean_rank_2": mean_ranks[m2],
                    "rank_diff": diff,
                    "critical_difference": cd,
                    "significant": diff > cd,
                }
            )

    if nemenyi_rows:
        pd.DataFrame(nemenyi_rows).to_csv(out / "nemenyi_posthoc.csv", index=False)
        print("  Saved nemenyi_posthoc.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  8. COEFFICIENT OF VARIATION
# ══════════════════════════════════════════════════════════════════════════════


def generate_cv_analysis(df):
    out = OUTPUT_DIR / "cv_analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in DATASETS:
        for model in MODELS:
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            if sub.empty:
                continue
            row = {"dataset": ds, "model": model}
            for metric in METRIC_COLS:
                vals = sub[metric].dropna()
                if len(vals) > 1 and vals.mean() != 0:
                    row[f"{metric}_cv"] = vals.std() / abs(vals.mean()) * 100
                else:
                    row[f"{metric}_cv"] = np.nan
            rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(out / "coefficient_of_variation.csv", index=False)

    # CV heatmaps for key metrics
    for metric in KEY_METRICS:
        col = f"{metric}_cv"
        if col not in cv_df.columns:
            continue
        pivot = cv_df.pivot(index="dataset", columns="model", values=col)
        pivot = pivot.reindex(index=DATASETS, columns=MODELS)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(
            f"Coefficient of Variation (%) \u2014 {metric}\n"
            f"(lower = more reproducible)"
        )
        fig.tight_layout()
        fig.savefig(out / f"{metric}_cv_heatmap.png")
        plt.close(fig)

    print("  Saved CV analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  9. RANK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def generate_rank_analysis(df):
    out = OUTPUT_DIR / "rank_analysis"
    out.mkdir(parents=True, exist_ok=True)

    for metric in KEY_METRICS:
        rows = []
        for ds in DATASETS:
            ds_df = df[df["dataset"] == ds]

            # Rank models within each run independently
            run_ranks = {}
            for run in range(1, N_RUNS + 1):
                run_df = ds_df[ds_df["run"] == run]
                if run_df.empty:
                    continue
                vals = run_df.set_index("model")[metric].dropna()
                if len(vals) < 2:
                    continue
                ranks = vals.rank(ascending=False)
                for model, rank in ranks.items():
                    run_ranks.setdefault(model, []).append(rank)

            for model in MODELS:
                if model not in run_ranks or len(run_ranks[model]) == 0:
                    continue
                r = run_ranks[model]
                rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "mean_rank": np.mean(r),
                        "std_rank": np.std(r),
                        "median_rank": np.median(r),
                        "best_rank_count": sum(1 for x in r if x == 1),
                        "n_runs": len(r),
                    }
                )

        pd.DataFrame(rows).to_csv(out / f"{metric}_rank_stability.csv", index=False)

    print("  Saved rank analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  10. RADAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════


def generate_radar_charts(df):
    out = OUTPUT_DIR / "radar_charts"
    out.mkdir(parents=True, exist_ok=True)

    n_ax = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for model in MODELS:
            m_df = ds_df[ds_df["model"] == model]
            if m_df.empty:
                continue
            values = [m_df[m].mean() for m in RADAR_METRICS]
            values += values[:1]
            c = MODEL_COLOURS.get(model, "#333")
            ax.plot(angles, values, "o-", linewidth=1.5, label=model, color=c, markersize=4)
            ax.fill(angles, values, alpha=0.06, color=c)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_METRICS, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Model Performance Profile \u2014 {ds}", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{ds}_radar.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {ds}_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  11. CRITICAL DIFFERENCE DIAGRAMS
# ══════════════════════════════════════════════════════════════════════════════


def _find_cliques(sorted_ranks, cd):
    """Find maximal cliques of models whose pairwise rank diffs are < cd."""
    n = len(sorted_ranks)
    cliques = []
    for i in range(n):
        clique = [i]
        for j in range(i + 1, n):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                clique.append(j)
            else:
                break
        if len(clique) > 1:
            cliques.append(clique)

    # Remove cliques that are subsets of others
    maximal = []
    for c in cliques:
        cs = set(c)
        if not any(cs < set(other) for other in cliques):
            maximal.append(c)
    return maximal


def generate_critical_difference_diagram(df):
    out = OUTPUT_DIR / "critical_difference"
    out.mkdir(parents=True, exist_ok=True)

    for metric in KEY_METRICS:
        pivot = df.groupby(["dataset", "model"])[metric].mean().unstack("model")
        pivot = pivot.reindex(columns=MODELS).dropna(axis=1, how="all")
        pivot = pivot.dropna(axis=0, how="any")

        if pivot.shape[0] < 2:
            continue

        ranks = pivot.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean(axis=0).sort_values()

        k = len(mean_ranks)
        n = pivot.shape[0]
        q = _NEMENYI_Q.get(k, 3.0)
        cd = q * np.sqrt(k * (k + 1) / (6 * n))

        sorted_models = mean_ranks.index.tolist()
        sorted_vals = mean_ranks.values

        cliques = _find_cliques(sorted_vals, cd)

        # ── Draw ──
        fig, ax = plt.subplots(figsize=(10, max(3, 1 + 0.5 * len(cliques))))
        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(0, 1)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Average Rank (lower is better)")

        # Model markers and labels
        for i, (model, rank) in enumerate(zip(sorted_models, sorted_vals)):
            c = MODEL_COLOURS.get(model, "#333")
            y_pos = 0.72 if i % 2 == 0 else 0.28
            ax.plot(rank, 0.5, "o", color=c, markersize=9, zorder=5)
            ax.annotate(
                f"{model}\n({rank:.2f})",
                xy=(rank, 0.5),
                xytext=(rank, y_pos),
                ha="center",
                va="center",
                fontsize=8,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            )

        # CD bar
        ax.plot([0.7, 0.7 + cd], [0.94, 0.94], "k-", linewidth=2)
        ax.text(0.7 + cd / 2, 0.98, f"CD={cd:.2f}", ha="center", fontsize=8)

        # Non-significant clique bars
        bar_ys = np.linspace(0.42, 0.58, max(len(cliques), 1))
        for gi, clique in enumerate(cliques):
            lo = sorted_vals[clique[0]]
            hi = sorted_vals[clique[-1]]
            y = bar_ys[gi % len(bar_ys)]
            ax.plot([lo - 0.05, hi + 0.05], [y, y], "k-", linewidth=3, alpha=0.4)

        ax.set_title(f"Critical Difference Diagram \u2014 {metric}", pad=25)
        fig.tight_layout()
        fig.savefig(out / f"{metric}_cd_diagram.png", bbox_inches="tight")
        plt.close(fig)

    print("  Saved critical difference diagrams")


# ══════════════════════════════════════════════════════════════════════════════
#  12. CONVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def generate_convergence_analysis(df):
    """Show how the cumulative mean of a metric stabilises as runs increase.

    This is directly relevant for a reproducibility paper: it answers
    'are N=30 runs sufficient for stable estimates?'
    """
    out = OUTPUT_DIR / "convergence"
    out.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds]

        for metric in KEY_METRICS:
            fig, ax = plt.subplots(figsize=(10, 5))

            for model in MODELS:
                vals = ds_df[ds_df["model"] == model].sort_values("run")[metric].dropna().values
                if len(vals) < 3:
                    continue
                cum_mean = np.cumsum(vals) / np.arange(1, len(vals) + 1)
                c = MODEL_COLOURS.get(model, "#333")
                ax.plot(range(1, len(vals) + 1), cum_mean, label=model, color=c, linewidth=1.5)

            ax.set_xlabel("Number of Runs")
            ax.set_ylabel(f"Cumulative Mean {metric}")
            ax.set_title(f"Metric Convergence \u2014 {metric} on {ds}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"{ds}_{metric}_convergence.png")
            plt.close(fig)

        print(f"  Saved convergence plots for {ds}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

SECTION_MAP = [
    ("summary_tables", "1/12", "Generating summary tables", lambda df: generate_summary_tables(df)),
    ("pr_curves", "2/12", "Generating PR curves", lambda df: generate_pr_curves()),
    ("confusion_matrices", "3/12", "Generating confusion matrices", lambda df: generate_confusion_matrices()),
    ("box_plots", "4/12", "Generating box plots", lambda df: generate_box_plots(df)),
    ("heatmap", "5/12", "Generating heatmaps", lambda df: generate_heatmaps(df)),
    ("violin_plots", "6/12", "Generating violin plots", lambda df: generate_violin_plots(df)),
    ("statistical_tests", "7/12", "Running statistical tests", lambda df: run_statistical_tests(df)),
    ("cv_analysis", "8/12", "Generating CV analysis", lambda df: generate_cv_analysis(df)),
    ("rank_analysis", "9/12", "Generating rank analysis", lambda df: generate_rank_analysis(df)),
    ("radar_charts", "10/12", "Generating radar charts", lambda df: generate_radar_charts(df)),
    ("critical_difference", "11/12", "Generating CD diagrams", lambda df: generate_critical_difference_diagram(df)),
    ("convergence", "12/12", "Generating convergence plots", lambda df: generate_convergence_analysis(df)),
]


def main():
    print("=" * 70)
    print("  Results Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("\nLoading run metrics...")
    df = load_all_run_metrics()

    enabled = [name for name, *_ in SECTION_MAP if SECTIONS.get(name, False)]
    disabled = [name for name, *_ in SECTION_MAP if not SECTIONS.get(name, False)]
    print(f"\nEnabled sections : {', '.join(enabled)}")
    if disabled:
        print(f"Disabled sections: {', '.join(disabled)}")

    for name, label, desc, func in SECTION_MAP:
        if SECTIONS.get(name, False):
            print(f"\n[{label}] {desc}...")
            func(df)

    print("\n" + "=" * 70)
    print(f"  Analysis complete. Outputs saved to:\n  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
