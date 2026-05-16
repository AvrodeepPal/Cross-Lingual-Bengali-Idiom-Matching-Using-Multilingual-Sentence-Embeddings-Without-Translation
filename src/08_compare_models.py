"""
STEP 8: Final Model Comparison
================================
Produces the final comparison between mSBERT, LaBSE, and BanglaBERT with:
  - Cross‑lingual side‑by‑side metrics (similarity, retrieval, clustering)
  - Monolingual Bengali metrics (self‑similarity, clustering silhouette)
  - Bar charts for each group
  - Statistical significance test on cross‑lingual similarity scores
  - A written verdict: which model is better and why

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

# Cross‑lingual models (those with hi/en embeddings)
CROSS_MODELS = ["mSBERT", "LaBSE"]
# All models (including monolingual BanglaBERT)
ALL_MODELS   = ["mSBERT", "LaBSE", "BanglaBERT"]
COLOURS = {"mSBERT": "#2a9d8f", "LaBSE": "#e63946", "BanglaBERT": "#6a4c93"}


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
    Three subplots: similarity, retrieval, clustering – only mSBERT vs LaBSE.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Cross‑Lingual Comparison: mSBERT vs LaBSE", fontsize=15, fontweight="bold")
    bar_width = 0.35

    # ── Plot 1: Similarity ──────────────────────────────────────────────────
    ax = axes[0]
    if not sim_df.empty:
        lang_pairs = sim_df["lang_pair"].unique()
        x = np.arange(len(lang_pairs))
        for i, m in enumerate(CROSS_MODELS):
            col    = f"{m}_score"
            means  = [sim_df[sim_df["lang_pair"]==lp][col].mean() for lp in lang_pairs]
            offset = (i - 0.5) * bar_width
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
            offset = (i - 0.5) * bar_width
            vals   = [agg.loc[m, met] for met in metrics]
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
            offset = (i - 0.5) * bar_width
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
    Compares mSBERT, LaBSE, and BanglaBERT.
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
    scores = {"mSBERT": 0, "LaBSE": 0, "BanglaBERT": 0}
    lines = []
    lines.append("=" * 65)
    lines.append("  MODEL COMPARISON VERDICT")
    lines.append("=" * 65)

    # ── Cross‑lingual evaluation ─────────────────────────────────────────────
    lines.append("\n[1] CROSS‑LINGUAL SEMANTIC SIMILARITY")
    if not sim_df.empty:
        for lp in sim_df["lang_pair"].unique():
            sub = sim_df[sim_df["lang_pair"] == lp]
            s_m = sub["mSBERT_score"].mean()
            s_l = sub["LaBSE_score"].mean()
            winner = "LaBSE" if s_l > s_m else "mSBERT"
            scores[winner] += 1
            lines.append(f"  {lp}: mSBERT={s_m:.4f}  LaBSE={s_l:.4f}  → {winner} wins")
        for lp, t in sig_tests.items():
            if t["significant"]:
                lines.append(f"  [{lp}] Difference is statistically significant "
                              f"(t={t['t_stat']}, p={t['p_value']})")
            else:
                lines.append(f"  [{lp}] No significant difference (p={t.get('p_value','N/A')})")

    lines.append("\n[2] CROSS‑LINGUAL RETRIEVAL")
    if not ret_df.empty:
        for metric in ["MRR", "P@1", "P@5", "Hit@10"]:
            if metric not in ret_df.columns:
                continue
            agg = ret_df.groupby("model")[metric].mean()
            if len(agg) < 2:
                continue
            winner = agg.idxmax()
            scores[winner] += 1
            lines.append(f"  {metric}: mSBERT={agg.get('mSBERT', np.nan):.4f}  "
                         f"LaBSE={agg.get('LaBSE', np.nan):.4f}  → {winner} wins")

    lines.append("\n[3] CROSS‑LINGUAL CLUSTERING")
    if not clu_df.empty:
        for metric in ["silhouette", "purity", "triplet_cohesion"]:
            if metric not in clu_df.columns:
                continue
            agg = clu_df.groupby("model")[metric].mean()
            if len(agg) < 2:
                continue
            winner = agg.idxmax()
            scores[winner] += 1
            lines.append(f"  {metric}: mSBERT={agg.get('mSBERT', np.nan):.4f}  "
                         f"LaBSE={agg.get('LaBSE', np.nan):.4f}  → {winner} wins")

    # ── Monolingual Bengali evaluation ───────────────────────────────────────
    lines.append("\n[4] MONOLINGUAL BENGALI EVALUATION")
    if not mono_sim_df.empty:
        # winner for self‑similarity is the one with the highest mean
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

    # ── Final tally ──────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 65)
    lines.append(f"  SCORE TALLY:  mSBERT={scores['mSBERT']}  "
                 f"LaBSE={scores['LaBSE']}  BanglaBERT={scores['BanglaBERT']}")
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
    # Also mention the cross‑lingual winner separately
    if "mSBERT" in scores and "LaBSE" in scores:
        cross_winner = "LaBSE" if scores["LaBSE"] > scores["mSBERT"] else "mSBERT"
        lines.append(f"\n  CROSS‑LINGUAL WINNER: {cross_winner}")
        if cross_winner == "LaBSE":
            lines.append("""
  INTERPRETATION (cross‑lingual):
  LaBSE (Language‑Agnostic BERT Sentence Embeddings) performs better for
  cross‑lingual idiom matching, which aligns with its dual‑encoder training
  specifically designed for cross‑lingual alignment. mSBERT, while strong
  on paraphrases, may not align languages as tightly as LaBSE.
""")
        else:
            lines.append("""
  INTERPRETATION (cross‑lingual):
  mSBERT (Multilingual Sentence‑BERT) is the top cross‑lingual model.
  Its training on paraphrase data across many languages likely gives it
  an edge in capturing idiomatic equivalence, a skill closer to paraphrase
  detection than to literal translation alignment.
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
