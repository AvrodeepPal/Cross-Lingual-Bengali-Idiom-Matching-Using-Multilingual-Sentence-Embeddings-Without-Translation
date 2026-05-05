"""
STEP 7: Evaluation Aggregator
================================
Pulls all results from Steps 4, 5, 6 into one unified evaluation report.

Outputs:
  results/full_evaluation_report.csv   — all metrics in one table
  results/plots/evaluation_radar.png   — radar chart: mSBERT vs LaBSE
  results/plots/score_distributions.png — similarity score histograms

Run: python src/07_evaluate.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("../results")
PLOTS_DIR   = Path("../results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS      = ["mSBERT", "LaBSE"]


# ─────────────────────────────────────────────────────────────────────────────
# Load result files
# ─────────────────────────────────────────────────────────────────────────────
def load_all_results():
    results = {}

    sim_path = RESULTS_DIR / "similarity_scores.csv"
    if sim_path.exists():
        results["similarity"] = pd.read_csv(sim_path)

    ret_path = RESULTS_DIR / "retrieval_metrics.csv"
    if ret_path.exists():
        results["retrieval"] = pd.read_csv(ret_path)

    clu_path = RESULTS_DIR / "clustering_metrics.csv"
    if clu_path.exists():
        results["clustering"] = pd.read_csv(clu_path)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Similarity distribution plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_score_distributions(sim_df: pd.DataFrame):
    """
    Plots histograms of similarity scores for both models and both language pairs.
    """
    lang_pairs = sim_df["lang_pair"].unique()
    fig, axes = plt.subplots(len(lang_pairs), 2, figsize=(14, 5 * len(lang_pairs)))
    if len(lang_pairs) == 1:
        axes = [axes]

    colours = {"mSBERT": "#2a9d8f", "LaBSE": "#e63946"}

    for row_i, lp in enumerate(lang_pairs):
        subset = sim_df[sim_df["lang_pair"] == lp]
        for col_i, model_key in enumerate(MODELS):
            ax  = axes[row_i][col_i]
            col = f"{model_key}_score"
            if col not in subset.columns:
                ax.set_visible(False)
                continue
            scores = subset[col].dropna()
            ax.hist(scores, bins=30, color=colours[model_key], edgecolor="white", alpha=0.85)
            ax.axvline(scores.mean(), color="black", linestyle="--",
                       linewidth=1.5, label=f"Mean: {scores.mean():.3f}")
            ax.set_title(f"{model_key} — {lp}", fontsize=12)
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Count")
            ax.legend()

    plt.suptitle("Similarity Score Distributions by Model & Language Pair", fontsize=14)
    plt.tight_layout()
    path = PLOTS_DIR / "score_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Radar chart comparing both models across all metrics
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_chart(report_df: pd.DataFrame):
    """
    Radar chart: each spoke is one metric, each line is one model.
    """
    # Metrics to show on radar (all should be 0–1 scale)
    metric_cols = [c for c in report_df.columns
                   if c not in ("model", "lang_pair", "n_queries", "n_idioms",
                                "n_clusters", "metric_type")]

    if not metric_cols:
        print("[!] No numeric metrics found for radar chart.")
        return

    # Average across language pairs per model
    numeric_df = report_df.groupby("model")[metric_cols].mean().reset_index()

    labels  = metric_cols
    N       = len(labels)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colours = {"mSBERT": "#2a9d8f", "LaBSE": "#e63946"}

    for _, row in numeric_df.iterrows():
        values  = [row[m] for m in labels]
        values += values[:1]
        colour  = colours.get(row["model"], "#555")
        ax.plot(angles, values, colour, linewidth=2, label=row["model"])
        ax.fill(angles, values, colour, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison: mSBERT vs LaBSE\n(all metrics, averaged across language pairs)",
                 size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = PLOTS_DIR / "evaluation_radar.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Radar chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Build unified report
# ─────────────────────────────────────────────────────────────────────────────
def build_full_report(results: dict) -> pd.DataFrame:
    rows = []

    # ── Similarity stats ──────────────────────────────────────────────────────
    if "similarity" in results:
        sim = results["similarity"]
        for lp in sim["lang_pair"].unique():
            sub = sim[sim["lang_pair"] == lp]
            for m in MODELS:
                col = f"{m}_score"
                if col not in sub.columns:
                    continue
                scores = sub[col].dropna()
                rows.append({
                    "model":       m,
                    "lang_pair":   lp,
                    "metric_type": "similarity",
                    "mean_cosine": round(scores.mean(), 4),
                    "std_cosine":  round(scores.std(), 4),
                    "median_cosine": round(scores.median(), 4),
                })

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    if "retrieval" in results:
        for _, row in results["retrieval"].iterrows():
            rows.append({
                "model":       row["model"],
                "lang_pair":   row["lang_pair"],
                "metric_type": "retrieval",
                "MRR":         row.get("MRR", np.nan),
                "P@1":         row.get("P@1", np.nan),
                "P@5":         row.get("P@5", np.nan),
                "Hit@10":      row.get("Hit@10", np.nan),
            })

    # ── Clustering metrics ────────────────────────────────────────────────────
    if "clustering" in results:
        for _, row in results["clustering"].iterrows():
            rows.append({
                "model":            row["model"],
                "lang_pair":        "all",
                "metric_type":      "clustering",
                "silhouette":       row.get("silhouette", np.nan),
                "purity":           row.get("purity", np.nan),
                "triplet_cohesion": row.get("triplet_cohesion", np.nan),
            })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(RESULTS_DIR / "full_evaluation_report.csv", index=False)
    print(f"[✓] Full report saved → results/full_evaluation_report.csv")
    return report_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 7: Full Evaluation")
    print("=" * 60 + "\n")

    results   = load_all_results()
    report_df = build_full_report(results)

    # Print full report
    print("\n── Full Evaluation Report ────────────────────────────────────")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(report_df.fillna("").to_string(index=False))

    # Plots
    if "similarity" in results:
        plot_score_distributions(results["similarity"])

    plot_radar_chart(report_df)

    print("\nNext: python src/08_compare_models.py\n")
