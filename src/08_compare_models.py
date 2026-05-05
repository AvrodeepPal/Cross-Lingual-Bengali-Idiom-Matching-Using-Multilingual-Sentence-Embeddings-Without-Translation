"""
STEP 8: Final Model Comparison
================================
Produces the final comparison between mSBERT and LaBSE with:
  - Side-by-side metric tables (similarity, retrieval, clustering)
  - Bar charts for every metric group
  - Statistical significance test (paired t-test on similarity scores)
  - A written verdict: which model is better and why

Output:
  results/plots/final_comparison.png
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
MODELS = ["mSBERT", "LaBSE"]
COLOURS = {"mSBERT": "#2a9d8f", "LaBSE": "#e63946"}


# ─────────────────────────────────────────────────────────────────────────────
# Load all results
# ─────────────────────────────────────────────────────────────────────────────
def load_results():
    sim = pd.read_csv(RESULTS_DIR / "similarity_scores.csv") \
        if (RESULTS_DIR / "similarity_scores.csv").exists() else pd.DataFrame()
    ret = pd.read_csv(RESULTS_DIR / "retrieval_metrics.csv") \
        if (RESULTS_DIR / "retrieval_metrics.csv").exists() else pd.DataFrame()
    clu = pd.read_csv(RESULTS_DIR / "clustering_metrics.csv") \
        if (RESULTS_DIR / "clustering_metrics.csv").exists() else pd.DataFrame()
    return sim, ret, clu


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance test on similarity scores
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
        # Align lengths
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
def plot_comparison_bars(sim_df, ret_df, clu_df):
    """
    Three subplots:
      1. Mean cosine similarity (bn-en and bn-hi)
      2. Retrieval metrics (MRR, P@1, P@5, Hit@10)
      3. Clustering metrics (Silhouette, Purity, Triplet Cohesion)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("mSBERT vs LaBSE — Full Model Comparison", fontsize=15, fontweight="bold")
    bar_width = 0.35

    # ── Plot 1: Similarity ────────────────────────────────────────────────────
    ax = axes[0]
    if not sim_df.empty:
        lang_pairs = sim_df["lang_pair"].unique()
        x = np.arange(len(lang_pairs))
        for i, m in enumerate(MODELS):
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

    # ── Plot 2: Retrieval ─────────────────────────────────────────────────────
    ax = axes[1]
    if not ret_df.empty:
        metrics = ["MRR", "P@1", "P@5", "Hit@10"]
        metrics = [m for m in metrics if m in ret_df.columns]
        # Average across language pairs
        agg = ret_df.groupby("model")[metrics].mean()
        x   = np.arange(len(metrics))
        for i, m in enumerate(MODELS):
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

    # ── Plot 3: Clustering ────────────────────────────────────────────────────
    ax = axes[2]
    if not clu_df.empty:
        metrics = ["silhouette", "purity", "triplet_cohesion"]
        metrics = [m for m in metrics if m in clu_df.columns]
        x       = np.arange(len(metrics))
        for i, m in enumerate(MODELS):
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
    path = PLOTS_DIR / "final_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Comparison chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Determine winner and write verdict
# ─────────────────────────────────────────────────────────────────────────────
def compute_verdict(sim_df, ret_df, clu_df, sig_tests) -> str:
    """
    Tallies wins per model across all metrics and writes a plain-English verdict.
    """
    scores = {"mSBERT": 0, "LaBSE": 0}

    lines = []
    lines.append("=" * 65)
    lines.append("  MODEL COMPARISON VERDICT")
    lines.append("=" * 65)

    # Similarity
    if not sim_df.empty:
        lines.append("\n[1] SEMANTIC SIMILARITY (Mean Cosine Score)")
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

    # Retrieval
    if not ret_df.empty:
        lines.append("\n[2] RETRIEVAL METRICS")
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

    # Clustering
    if not clu_df.empty:
        lines.append("\n[3] CLUSTERING METRICS")
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

    # Final verdict
    lines.append("\n" + "=" * 65)
    lines.append(f"  SCORE TALLY:  mSBERT={scores['mSBERT']}  LaBSE={scores['LaBSE']}")
    overall_winner = max(scores, key=scores.get)
    lines.append(f"\n  OVERALL WINNER: {overall_winner}")
    lines.append("=" * 65)

    if overall_winner == "LaBSE":
        lines.append("""
  INTERPRETATION:
  LaBSE (Language-Agnostic BERT Sentence Embeddings) outperforms mSBERT
  on this cross-lingual idiom task. This is expected because LaBSE was
  specifically trained to align representations across 109 languages
  using a dual-encoder framework with cross-lingual transfer. Its training
  objective directly optimised for cross-lingual similarity, making it
  better suited for low-resource language pairs like Bengali–Hindi and
  Bengali–English in an idiomatic context.
""")
    else:
        lines.append("""
  INTERPRETATION:
  mSBERT (Multilingual Sentence-BERT) outperforms LaBSE on this task.
  This may be because mSBERT's training on paraphrase data across many
  languages gives it a stronger understanding of figurative/paraphrase
  equivalence, which is exactly what idiom matching requires. LaBSE is
  optimised for literal sentence alignment (e.g. translation pairs),
  which may not transfer as well to idiom semantics.
""")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 8: Final Model Comparison")
    print("=" * 60 + "\n")

    sim_df, ret_df, clu_df = load_results()

    # Statistical significance
    sig_tests = {}
    if not sim_df.empty:
        sig_tests = significance_test(sim_df)

    # Comparison bar charts
    plot_comparison_bars(sim_df, ret_df, clu_df)

    # Verdict
    verdict = compute_verdict(sim_df, ret_df, clu_df, sig_tests)
    print(verdict)

    # Save verdict to file
    verdict_path = RESULTS_DIR / "model_comparison_summary.txt"
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write(verdict)
    print(f"\n[✓] Verdict saved → {verdict_path}")
    print("\n✅  PROJECT PIPELINE COMPLETE\n")
    print("Final outputs:")
    print("  results/similarity_scores.csv")
    print("  results/retrieval_metrics.csv")
    print("  results/clustering_metrics.csv")
    print("  results/full_evaluation_report.csv")
    print("  results/model_comparison_summary.txt")
    print("  results/plots/  (all charts)\n")
