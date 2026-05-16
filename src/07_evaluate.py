"""
STEP 7: Evaluation Aggregator
================================
Pulls all results from Steps 4, 5, 6 into one unified evaluation report.

*** NEW: Also loads monolingual similarity, retrieval, and clustering results
    and includes them in the report and separate radar chart. ***

Outputs:
  results/full_evaluation_report.csv      — all metrics in one table
  results/plots/evaluation_radar.png      — radar chart: mSBERT vs LaBSE (cross-lingual)
  results/plots/monolingual_radar.png     — radar chart: all models (Bengali-only)
  results/plots/score_distributions.png   — similarity score histograms (cross-lingual only)

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
MODELS      = ["mSBERT", "LaBSE"]                     # cross-lingual models
ALL_MODELS  = ["mSBERT", "LaBSE", "BanglaBERT"]       # all models for monolingual


# ─────────────────────────────────────────────────────────────────────────────
# Load result files
# ─────────────────────────────────────────────────────────────────────────────
def load_all_results():
    results = {}

    # Original cross-lingual similarity
    sim_path = RESULTS_DIR / "similarity_scores.csv"
    if sim_path.exists():
        results["similarity"] = pd.read_csv(sim_path)

    # Cross-lingual retrieval
    ret_path = RESULTS_DIR / "retrieval_metrics.csv"
    if ret_path.exists():
        results["retrieval"] = pd.read_csv(ret_path)

    # Cross-lingual clustering
    clu_path = RESULTS_DIR / "clustering_metrics.csv"
    if clu_path.exists():
        results["clustering"] = pd.read_csv(clu_path)

    # NEW: Monolingual files
    mono_sim_path = RESULTS_DIR / "monolingual_similarity.csv"
    if mono_sim_path.exists():
        results["mono_sim"] = pd.read_csv(mono_sim_path)

    mono_ret_path = RESULTS_DIR / "monolingual_retrieval.csv"
    if mono_ret_path.exists():
        results["mono_ret"] = pd.read_csv(mono_ret_path)

    mono_clu_path = RESULTS_DIR / "monolingual_clustering.csv"
    if mono_clu_path.exists():
        results["mono_clu"] = pd.read_csv(mono_clu_path)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Similarity distribution plots (cross-lingual only)
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
# Cross‑lingual radar chart (mSBERT vs LaBSE)
# ─────────────────────────────────────────────────────────────────────────────
def plot_crosslingual_radar(report_df: pd.DataFrame):
    """
    Radar chart comparing mSBERT and LaBSE across cross‑lingual metrics.
    """
    # Select only rows that are cross‑lingual (metric_type = similarity, retrieval, clustering)
    cross = report_df[report_df["lang_pair"].isin(["bn-en", "bn-hi", "all"])].copy()

    # Define which metrics to plot (0–1 scale)
    metric_cols = [
        "mean_cosine",  # similarity (we'll average over lang pairs)
        "MRR",
        "P@1",
        "P@5",
        "Hit@10",
        "silhouette",
        "purity",
        "triplet_cohesion",
    ]

    # Keep only metrics that exist in the dataframe
    available = [col for col in metric_cols if col in cross.columns]
    if not available:
        print("[!] No cross‑lingual metrics found for radar chart.")
        return

    # Average across language pairs per model
    # For similarity, we have one row per model per lang_pair; we'll average 'mean_cosine'
    # For retrieval, we'll average over lang pairs. For clustering, lang_pair='all', just take it.
    numeric_df = cross.groupby("model")[available].mean().reset_index()

    # Ensure we only have mSBERT and LaBSE
    numeric_df = numeric_df[numeric_df["model"].isin(MODELS)]

    labels  = available
    N       = len(labels)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

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
    ax.set_title("Cross‑Lingual Comparison: mSBERT vs LaBSE\n(metrics averaged across language pairs)",
                 size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = PLOTS_DIR / "evaluation_radar.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Cross‑lingual radar chart saved → {path}")


# NEW: Monolingual radar chart (all models with Bengali embeddings)
def plot_monolingual_radar(report_df: pd.DataFrame):
    """
    Radar chart for monolingual Bengali metrics, comparing all models
    that have Bengali embeddings (mSBERT, LaBSE, BanglaBERT).
    """
    # Select monolingual rows
    mono = report_df[report_df["lang_pair"].isin(["bn-bn", "bn"])].copy()

    # Metrics we want: mean_cosine (from monolingual similarity), silhouette (from monolingual clustering)
    # Skip retrieval because it's a trivial sanity check (all perfect)
    metric_cols = [
        "mean_cosine",   # self-similarity
        "silhouette",     # monolingual clustering
    ]

    available = [col for col in metric_cols if col in mono.columns]
    if not available:
        print("[!] No monolingual metrics found for radar chart.")
        return

    # Some models may have missing values, but we only plot what's there
    numeric_df = mono.groupby("model")[available].mean().reset_index()

    labels  = available
    N       = len(labels)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colours = {"mSBERT": "#2a9d8f", "LaBSE": "#e63946", "BanglaBERT": "#6a4c93"}

    for _, row in numeric_df.iterrows():
        values  = [row[m] for m in labels]
        values += values[:1]
        colour  = colours.get(row["model"], "#555")
        ax.plot(angles, values, colour, linewidth=2, label=row["model"])
        ax.fill(angles, values, colour, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Monolingual Bengali Comparison\n(Self‑Similarity & Clustering Silhouette)",
                 size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = PLOTS_DIR / "monolingual_radar.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Monolingual radar chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Build unified report
# ─────────────────────────────────────────────────────────────────────────────
def build_full_report(results: dict) -> pd.DataFrame:
    rows = []

    # ── Cross‑lingual similarity stats ────────────────────────────────────────
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

    # ── Cross‑lingual retrieval metrics ───────────────────────────────────────
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

    # ── Cross‑lingual clustering metrics ──────────────────────────────────────
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

    # ── NEW: Monolingual Bengali similarity ───────────────────────────────────
    if "mono_sim" in results:
        mono_sim = results["mono_sim"]
        for _, row in mono_sim.iterrows():
            rows.append({
                "model":       row["model"],
                "lang_pair":   "bn-bn",       # indicates monolingual
                "metric_type": "monolingual_similarity",
                "mean_cosine": row.get("mean_cosine", np.nan),
            })

    # ── NEW: Monolingual retrieval (sanity check – all should be 1.0) ─────────
    if "mono_ret" in results:
        mono_ret = results["mono_ret"]
        for _, row in mono_ret.iterrows():
            rows.append({
                "model":       row["model"],
                "lang_pair":   "bn-bn",
                "metric_type": "monolingual_retrieval",
                "MRR":         row.get("MRR", np.nan),
                "P@1":         row.get("P@1", np.nan),
                "P@5":         row.get("P@5", np.nan),
                "Hit@10":      row.get("Hit@10", np.nan),
            })

    # ── NEW: Monolingual Bengali clustering ───────────────────────────────────
    if "mono_clu" in results:
        mono_clu = results["mono_clu"]
        for _, row in mono_clu.iterrows():
            rows.append({
                "model":       row["model"],
                "lang_pair":   "bn",
                "metric_type": "clustering_monolingual",
                "silhouette":  row.get("silhouette", np.nan),
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

    # Cross‑lingual similarity distributions (unchanged)
    if "similarity" in results:
        plot_score_distributions(results["similarity"])

    # Cross‑lingual radar
    plot_crosslingual_radar(report_df)

    # NEW: Monolingual radar
    plot_monolingual_radar(report_df)

    print("\nNext: python src/08_compare_models.py\n")
