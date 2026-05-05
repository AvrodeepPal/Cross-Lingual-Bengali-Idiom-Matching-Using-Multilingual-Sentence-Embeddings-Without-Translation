"""
STEP 6: Cross-Lingual Clustering
==================================
Clusters all Bengali, Hindi, and English idiom embeddings together.
If a model truly understands cross-lingual semantics, same-meaning idioms
from different languages should fall in the same cluster.

Metrics:
  - Silhouette Score     — how well-separated are clusters overall?
  - Cluster Purity       — what % of each cluster is one language/topic?
  - Cross-lingual Cohesion — are triplet members in the same cluster?

Outputs:
  results/cluster_assignments.csv   — which cluster each idiom was assigned to
  results/clustering_metrics.csv    — Silhouette + Purity per model
  results/plots/clustering_*.png    — 2D UMAP/t-SNE visualisations

Run: python src/06_clustering.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: UMAP is better than t-SNE for large datasets
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

PROCESSED_DIR = Path("../data/processed")
RESULTS_DIR   = Path("../results")
PLOTS_DIR     = Path("../results/plots")
EMB_DIR       = Path("../results/embeddings")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS        = ["mSBERT", "LaBSE"]


# ─────────────────────────────────────────────────────────────────────────────
def load_embeddings(model_key):
    d = EMB_DIR / model_key
    return {lang: np.load(d / f"{lang}.npy") for lang in ["bn", "hi", "en"]}


def reduce_to_2d(matrix: np.ndarray, method: str = "auto") -> np.ndarray:
    """Reduce high-dim embeddings to 2D for visualisation."""
    if HAS_UMAP and method in ("auto", "umap"):
        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    else:
        reducer = TSNE(n_components=2, random_state=42, metric="cosine",
                       perplexity=min(30, len(matrix) - 1))
    return reducer.fit_transform(matrix)


# ─────────────────────────────────────────────────────────────────────────────
def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Cluster purity: for each cluster, find the most common true label.
    Purity = (sum of per-cluster majority counts) / total samples
    """
    from collections import Counter
    total, correct = 0, 0
    for cluster_id in np.unique(labels_pred):
        mask    = labels_pred == cluster_id
        counts  = Counter(labels_true[mask])
        correct += counts.most_common(1)[0][1]
        total   += mask.sum()
    return correct / total if total > 0 else 0.0


def triplet_cohesion(triplet_df: pd.DataFrame,
                     bn_idioms: list, hi_idioms: list, en_idioms: list,
                     all_idioms: list, cluster_labels: np.ndarray) -> float:
    """
    For each verified triplet, check if all three idioms landed in the same cluster.
    Returns the fraction of triplets that are fully co-clustered.
    """
    idiom_to_cluster = {idiom: cluster_labels[i] for i, idiom in enumerate(all_idioms)}
    hits, total = 0, 0

    valid = triplet_df[
        triplet_df["idiom_bn"].isin(bn_idioms) &
        triplet_df["idiom_hi"].isin(hi_idioms) &
        triplet_df["idiom_en"].isin(en_idioms)
    ]

    for _, row in valid.iterrows():
        c_bn = idiom_to_cluster.get(row["idiom_bn"])
        c_hi = idiom_to_cluster.get(row["idiom_hi"])
        c_en = idiom_to_cluster.get(row["idiom_en"])
        if c_bn is not None and c_hi is not None and c_en is not None:
            if c_bn == c_hi == c_en:
                hits += 1
            total += 1

    return hits / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def run_clustering(model_key: str,
                   emb: dict,
                   bn_idioms: list,
                   hi_idioms: list,
                   en_idioms: list,
                   triplet_df: pd.DataFrame) -> dict:
    """
    Runs K-Means clustering on the combined embedding space of all three languages.
    """
    # Stack all embeddings together
    all_embs   = np.vstack([emb["bn"], emb["hi"], emb["en"]])  # (N_total, D)
    all_idioms = bn_idioms + hi_idioms + en_idioms
    lang_labels = (
        ["Bengali"]  * len(bn_idioms) +
        ["Hindi"]    * len(hi_idioms) +
        ["English"]  * len(en_idioms)
    )

    # Number of clusters: heuristic = max(3, total/10) capped at 50
    n_clusters = min(50, max(3, len(all_idioms) // 10))
    print(f"  [{model_key}] Clustering {len(all_idioms)} idioms into {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_embs)

    # Encode language labels as integers for purity calculation
    le = LabelEncoder()
    lang_int = le.fit_transform(lang_labels)

    # Metrics
    sil_score = silhouette_score(all_embs, cluster_labels, metric="cosine",
                                 sample_size=min(2000, len(all_embs)))
    purity    = cluster_purity(lang_int, cluster_labels)
    cohesion  = triplet_cohesion(triplet_df, bn_idioms, hi_idioms, en_idioms,
                                  all_idioms, cluster_labels)

    print(f"    Silhouette:  {sil_score:.4f}")
    print(f"    Purity:      {purity:.4f}")
    print(f"    Triplet cohesion: {cohesion:.4f}")

    # Save cluster assignments
    assign_df = pd.DataFrame({
        "model":         model_key,
        "idiom":         all_idioms,
        "language":      lang_labels,
        "cluster":       cluster_labels,
    })

    # ── 2D visualisation ──────────────────────────────────────────────────────
    print(f"  [{model_key}] Reducing to 2D for visualisation...")
    coords_2d = reduce_to_2d(all_embs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Cross-Lingual Idiom Clustering — {model_key}", fontsize=14)

    # Plot 1: colour by language
    colours = {"Bengali": "#e63946", "Hindi": "#2a9d8f", "English": "#f4a261"}
    for lang in ["Bengali", "Hindi", "English"]:
        mask = np.array(lang_labels) == lang
        axes[0].scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                        c=colours[lang], label=lang, alpha=0.6, s=20)
    axes[0].set_title("Coloured by Language")
    axes[0].legend()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot 2: colour by cluster
    scatter = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1],
                               c=cluster_labels, cmap="tab20", alpha=0.6, s=20)
    axes[1].set_title(f"Coloured by Cluster (k={n_clusters})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"clustering_{model_key}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    Plot saved → {plot_path}")

    return {
        "model":          model_key,
        "n_idioms":       len(all_idioms),
        "n_clusters":     n_clusters,
        "silhouette":     round(sil_score, 4),
        "purity":         round(purity, 4),
        "triplet_cohesion": round(cohesion, 4),
        "assign_df":      assign_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 6: Cross-Lingual Clustering")
    print("=" * 60 + "\n")

    bn_df = pd.read_csv(PROCESSED_DIR / "clean_bengali.csv")
    hi_df = pd.read_csv(PROCESSED_DIR / "clean_hindi.csv")
    en_df = pd.read_csv(PROCESSED_DIR / "clean_english.csv")
    triplet_df = pd.read_csv(PROCESSED_DIR / "triplets.csv")

    bn_col = "idiom_bn" if "idiom_bn" in bn_df.columns else bn_df.columns[0]
    hi_col = "idiom_hi" if "idiom_hi" in hi_df.columns else hi_df.columns[0]
    en_col = "idiom_en" if "idiom_en" in en_df.columns else en_df.columns[0]

    bn_idioms = bn_df[bn_col].fillna("").tolist()
    hi_idioms = hi_df[hi_col].fillna("").tolist()
    en_idioms = en_df[en_col].fillna("").tolist()

    all_assign = []
    all_metrics = []

    for model_key in MODELS:
        print(f"\n── {model_key} ──────────────────────────────────────────────")
        emb    = load_embeddings(model_key)
        result = run_clustering(model_key, emb, bn_idioms, hi_idioms, en_idioms, triplet_df)
        all_assign.append(result.pop("assign_df"))
        all_metrics.append(result)

    # Save
    pd.concat(all_assign, ignore_index=True).to_csv(
        RESULTS_DIR / "cluster_assignments.csv", index=False)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "clustering_metrics.csv", index=False)

    print("\n[✓] Saved → results/cluster_assignments.csv")
    print("[✓] Saved → results/clustering_metrics.csv")
    print("\n── Clustering Metrics Summary ────────────────────────────────")
    print(metrics_df[["model","n_idioms","n_clusters","silhouette","purity","triplet_cohesion"]].to_string(index=False))
    print("\nNext: python src/07_evaluate.py\n")
