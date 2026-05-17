"""
STEP 8: Final Model Comparison
================================
Now compares mSBERT, LaBSE, XLM-R and BanglaBERT.

Output:
  results/plots/final_comparison_crosslingual.png
  results/plots/final_comparison_monolingual.png
  results/model_comparison_summary.txt

Run: python src/08_compare_models.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("../results")
PLOTS_DIR   = Path("../results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cross‑lingual models (multilingual)
CROSS_MODELS = ["mSBERT", "LaBSE", "XLM-R"]
# All models (including BanglaBERT)
ALL_MODELS   = ["mSBERT", "LaBSE", "XLM-R", "BanglaBERT"]

COLOURS = {
    "mSBERT": "#2a9d8f",
    "LaBSE": "#e63946",
    "XLM-R": "#f4a261",
    "BanglaBERT": "#6a4c93",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load all results (cross‑lingual + monolingual)
# ─────────────────────────────────────────────────────────────────────────────
def load_results():
    sim = pd.read_csv(RESULTS_DIR / "similarity_scores.csv") \
        if (RESULTS_DIR / "similarity_scores.csv").exists() else pd.DataFrame()
    ret = pd.read_csv(RESULTS_DIR / "retrieval_metrics.csv") \
        if (RESULTS_DIR / "retrieval_metrics.csv").exists() else pd.DataFrame()
    clu = pd.read_csv(RESULTS_DIR / "clustering_metrics.csv") \
        if (RESULTS_DIR / "clustering_metrics.csv").exists() else pd.DataFrame()

    # NEW: monolingual results
    mono_sim = pd.read_csv(RESULTS_DIR / "monolingual_similarity.csv") \
        if (RESULTS_DIR / "monolingual_similarity.csv").exists() else pd.DataFrame()
    mono_ret = pd.read_csv(RESULTS_DIR / "monolingual_retrieval.csv") \
        if (RESULTS_DIR / "monolingual_retrieval.csv").exists() else pd.DataFrame()
    mono_clu = pd.read_csv(RESULTS_DIR / "monolingual_clustering.csv") \
        if (RESULTS_DIR / "monolingual_clustering.csv").exists() else pd.DataFrame()

    return sim, ret, clu, mono_sim, mono_ret, mono_clu


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance test on cross‑lingual similarity scores
# ─────────────────────────────────────────────────────────────────────────────
def significance_test(sim_df: pd.DataFrame) -> dict:
    """
    Paired t-test: are the similarity scores of mSBERT and LaBSE
    statistically significantly different?
    """
    results = {}
    for lp in sim_df["lang_pair"].unique():
        sub = sim_df[sim_df["lang_pair"] == lp]
        a = sub["mSBERT_score"].dropna()
        b = sub["LaBSE_score"].dropna()
        n = min(len(a), len(b))
        if n < 5:
            results[lp] = {"t_stat": np.nan, "p_value": np.nan, "significant": False}
            continue
        t_stat, p_val = stats.ttest_rel(a.values[:n], b.values[:n])
        results[lp] = {
            "t_stat":      round(t_stat, 4),
            "p_value":     round(p_val, 4),
            "significant": p_val < 0.05,
            "better":      "LaBSE" if b.mean() > a.mean() else "mSBERT",
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bar charts
# ─────────────────────────────────────────────────────────────────────────────
def plot_crosslingual_comparison(sim_df, ret_df, clu_df):
    """
    Three subplots: similarity, retrieval, clustering – mSBERT, LaBSE, XLM-R.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Cross‑Lingual Comparison: mSBERT vs LaBSE vs XLM-R", fontsize=15, fontweight="bold")
    bar_width = 0.25

    # ── Plot 1: Similarity ──────────────────────────────────────────────────
    ax = axes[0]
    if not sim_df.empty:
        lang_pairs = sim_df["lang_pair"].unique()
        x = np.arange(len(lang_pairs))
        for i, m in enumerate(CROSS_MODELS):
            col    = f"{m}_score"
            # In case a model's column is missing, skip it
            if col not in sim_df.columns:
                continue
            means  = [sim_df[sim_df["lang_pair"]==lp][col].mean() for lp in lang_pairs]
            offset = (i - 1) * bar_width  # center three bars
            bars   = ax.bar(x + offset, means, bar_width,
                            label=m, color=COLOURS[m], alpha=0.85)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(lang_pairs)
        ax.set_ylim(0, 1)
        ax.set_title("Mean Cosine Similarity", fontsize=12)
        ax.set_ylabel("Score")
        ax.legend()

    # ── Plot 2: Retrieval ───────────────────────────────────────────────────
    ax = axes[1]
    if not ret_df.empty:
        metrics = ["MRR", "P@1", "P@5", "Hit@10"]
        metrics = [m for m in metrics if m in ret_df.columns]
        agg = ret_df.groupby("model")[metrics].mean()
        x   = np.arange(len(metrics))
        for i, m in enumerate(CROSS_MODELS):
            if m not in agg.index:
                continue
            offset = (i - 1) * bar_width
            vals   = [agg.loc[m, met] if met in agg.columns else np.nan for met in metrics]
            # Replace NaN with 0 for plotting
            vals   = [v if not np.isnan(v) else 0 for v in vals]
            bars   = ax.bar(x + offset, vals, bar_width,
                            label=m, color=COLOURS[m], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Retrieval Metrics (avg over lang pairs)", fontsize=12)
        ax.set_ylabel("Score")
        ax.legend()

    # ── Plot 3: Clustering ──────────────────────────────────────────────────
    ax = axes[2]
    if not clu_df.empty:
        metrics = ["silhouette", "purity", "triplet_cohesion"]
        metrics = [m for m in metrics if m in clu_df.columns]
        x       = np.arange(len(metrics))
        for i, m in enumerate(CROSS_MODELS):
            sub    = clu_df[clu_df["model"] == m]
            if sub.empty:
                continue
            offset = (i - 1) * bar_width
            vals   = [sub[met].values[0] for met in metrics]
            bars   = ax.bar(x + offset, vals, bar_width,
                            label=m, color=COLOURS[m], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title("Clustering Metrics", fontsize=12)
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    path = PLOTS_DIR / "final_comparison_crosslingual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Cross‑lingual comparison chart saved → {path}")


def plot_monolingual_comparison(mono_sim_df, mono_clu_df):
    """
    Two subplots: monolingual similarity & monolingual clustering.
    Compares mSBERT, LaBSE, XLM-R, and BanglaBERT.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Monolingual Bengali Comparison: All Models", fontsize=14, fontweight="bold")
    bar_width = 0.25

    # ── Plot 1: Monolingual Self‑Similarity ────────────────────────────────
    ax = axes[0]
    if not mono_sim_df.empty:
        models_here = mono_sim_df["model"].tolist()
        means = mono_sim_df["mean_cosine"].tolist()
        x = np.arange(len(models_here))
        bars = ax.bar(x, means, bar_width, color=[COLOURS[m] for m in models_here], alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models_here)
        ax.set_ylim(0, 1)
        ax.set_title("Bengali Self‑Similarity", fontsize=12)
        ax.set_ylabel("Mean Cosine Similarity")

    # ── Plot 2: Monolingual Clustering Silhouette ──────────────────────────
    ax = axes[1]
    if not mono_clu_df.empty:
        models_here = mono_clu_df["model"].tolist()
        sil_scores  = mono_clu_df["silhouette"].tolist()
        x = np.arange(len(models_here))
        bars = ax.bar(x, sil_scores, bar_width, color=[COLOURS[m] for m in models_here], alpha=0.85)
        for bar, val in zip(bars, sil_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models_here)
        ax.set_ylim(0, 1)
        ax.set_title("Bengali Clustering Silhouette", fontsize=12)
        ax.set_ylabel("Silhouette Score")

    plt.tight_layout()
    path = PLOTS_DIR / "final_comparison_monolingual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Monolingual comparison chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Determine winner and write verdict
# ─────────────────────────────────────────────────────────────────────────────
def compute_verdict(sim_df, ret_df, clu_df,
                    mono_sim_df, mono_clu_df,
                    sig_tests) -> str:
    """
    Tallies wins per model for cross‑lingual and monolingual tasks,
    and writes a plain‑English verdict.
    """
    # Include all possible models (dynamic from data + known ones)
    scores = {"mSBERT": 0, "LaBSE": 0, "XLM-R": 0, "BanglaBERT": 0}
    lines = []
    lines.append("=" * 65)
    lines.append("  MODEL COMPARISON VERDICT")
    lines.append("=" * 65)

    # ── Cross‑lingual evaluation ─────────────────────────────────────────────
    lines.append("\n[1] CROSS‑LINGUAL SEMANTIC SIMILARITY")
    if not sim_df.empty:
        # Determine which models have similarity columns
        sim_models = [m for m in CROSS_MODELS if f"{m}_score" in sim_df.columns]
        for lp in sim_df["lang_pair"].unique():
            sub = sim_df[sim_df["lang_pair"] == lp]
            # Find best model for this language pair
            best_model = None
            best_mean = -1
            means_list = []
            for m in sim_models:
                col = f"{m}_score"
                mean_val = sub[col].mean()
                means_list.append(f"{m}={mean_val:.4f}")
                if mean_val > best_mean:
                    best_mean = mean_val
                    best_model = m
            if best_model:
                scores[best_model] += 1
                lines.append(f"  {lp}: {'  '.join(means_list)}  → {best_model} wins")
        for lp, t in sig_tests.items():
            if t["significant"]:
                lines.append(f"  [{lp}] Difference is statistically significant "
                              f"(t={t['t_stat']}, p={t['p_value']})")
            else:
                lines.append(f"  [{lp}] No significant difference (p={t.get('p_value','N/A')})")
    else:
        lines.append("  No similarity data found.")

    lines.append("\n[2] CROSS‑LINGUAL RETRIEVAL")
    if not ret_df.empty:
        # Determine which models are present in retrieval data
        ret_models = ret_df["model"].unique()
        for metric in ["MRR", "P@1", "P@5", "Hit@10"]:
            if metric not in ret_df.columns:
                continue
            agg = ret_df.groupby("model")[metric].mean()
            if len(agg) < 2:
                continue
            winner = agg.idxmax()
            scores[winner] += 1
            parts = [f"{m}={agg[m]:.4f}" for m in ret_models if m in agg.index]
            lines.append(f"  {metric}: {'  '.join(parts)}  → {winner} wins")
    else:
        lines.append("  No retrieval metrics found.")

    lines.append("\n[3] CROSS‑LINGUAL CLUSTERING")
    if not clu_df.empty:
        clu_models = clu_df["model"].unique()
        for metric in ["silhouette", "purity", "triplet_cohesion"]:
            if metric not in clu_df.columns:
                continue
            agg = clu_df.groupby("model")[metric].mean()
            if len(agg) < 2:
                continue
            winner = agg.idxmax()
            scores[winner] += 1
            parts = [f"{m}={agg[m]:.4f}" for m in clu_models if m in agg.index]
            lines.append(f"  {metric}: {'  '.join(parts)}  → {winner} wins")
    else:
        lines.append("  No clustering metrics found.")

    # ── Monolingual Bengali evaluation ───────────────────────────────────────
    lines.append("\n[4] MONOLINGUAL BENGALI EVALUATION")
    if not mono_sim_df.empty:
        best = mono_sim_df.loc[mono_sim_df["mean_cosine"].idxmax()]
        scores[best["model"]] += 1
        for _, row in mono_sim_df.iterrows():
            lines.append(f"  Self‑similarity ({row['model']}): {row['mean_cosine']:.4f}")
        lines.append(f"    → Winner: {best['model']}")
    if not mono_clu_df.empty:
        best = mono_clu_df.loc[mono_clu_df["silhouette"].idxmax()]
        scores[best["model"]] += 1
        for _, row in mono_clu_df.iterrows():
            lines.append(f"  Clustering silhouette ({row['model']}): {row['silhouette']:.4f}")
        lines.append(f"    → Winner: {best['model']}")
    if mono_sim_df.empty and mono_clu_df.empty:
        lines.append("  No monolingual evaluation data found.")

    # ── Final tally ──────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 65)
    score_str = "  ".join([f"{m}={scores[m]}" for m in scores])
    lines.append(f"  SCORE TALLY: {score_str}")
    overall_winner = max(scores, key=scores.get)
    lines.append(f"\n  OVERALL WINNER: {overall_winner}")
    lines.append("=" * 65)

    # Interpretive text
    if overall_winner == "BanglaBERT":
        lines.append("""
  INTERPRETATION:
  BanglaBERT outperforms the multilingual models on the monolingual Bengali
  tasks (self‑similarity and clustering). This demonstrates its strong
  representation of Bengali idioms. However, BanglaBERT cannot be used
  for cross‑lingual retrieval (it lacks English/Hindi embeddings).
  For cross‑lingual tasks, please refer to the cross‑lingual winner below.
""")
    # Determine cross‑lingual winner among multilingual models
    multilingual_models = [m for m in ["mSBERT", "LaBSE", "XLM-R"] if m in scores]
    if len(multilingual_models) > 0:
        cross_winner = max(multilingual_models, key=lambda m: scores[m])
        lines.append(f"\n  CROSS‑LINGUAL WINNER: {cross_winner}")
        if cross_winner == "LaBSE":
            lines.append("""
  INTERPRETATION (cross‑lingual):
  LaBSE (Language‑Agnostic BERT Sentence Embeddings) performs better for
  cross‑lingual idiom matching, which aligns with its dual‑encoder training
  specifically designed for cross‑lingual alignment. mSBERT, while strong
  on paraphrases, may not align languages as tightly as LaBSE.
""")
        elif cross_winner == "mSBERT":
            lines.append("""
  INTERPRETATION (cross‑lingual):
  mSBERT (Multilingual Sentence‑BERT) is the top cross‑lingual model.
  Its training on paraphrase data across many languages likely gives it
  an edge in capturing idiomatic equivalence, a skill closer to paraphrase
  detection than to literal translation alignment.
""")
        elif cross_winner == "XLM-R":
            lines.append("""
  INTERPRETATION (cross‑lingual):
  XLM-R (XLM‑RoBERTa) shows the strongest cross‑lingual performance.
  Its large‑scale multilingual pretraining with cross‑lingual masked language
  modelling provides robust cross‑lingual alignment, which seems to benefit
  idiomatic matching.
""")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 8: Final Model Comparison")
    print("=" * 60 + "\n")

    sim_df, ret_df, clu_df, mono_sim_df, mono_ret_df, mono_clu_df = load_results()

    # Statistical significance (cross‑lingual only)
    sig_tests = {}
    if not sim_df.empty:
        sig_tests = significance_test(sim_df)

    # Cross‑lingual bar chart
    plot_crosslingual_comparison(sim_df, ret_df, clu_df)

    # Monolingual bar chart
    plot_monolingual_comparison(mono_sim_df, mono_clu_df)

    # Verdict
    verdict = compute_verdict(sim_df, ret_df, clu_df,
                              mono_sim_df, mono_clu_df,
                              sig_tests)
    print(verdict)

    # Save verdict to file
    verdict_path = RESULTS_DIR / "model_comparison_summary.txt"
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write(verdict)
    print(f"\n[✓] Verdict saved → {verdict_path}")
    print("\n✅  PROJECT PIPELINE COMPLETE\n")
    print("Final outputs:")
    print("  results/similarity_scores.csv")
    print("  results/monolingual_similarity.csv")
    print("  results/retrieval_metrics.csv")
    print("  results/monolingual_retrieval.csv")
    print("  results/clustering_metrics.csv")
    print("  results/monolingual_clustering.csv")
    print("  results/full_evaluation_report.csv")
    print("  results/model_comparison_summary.txt")
    print("  results/plots/  (all charts)\n")
