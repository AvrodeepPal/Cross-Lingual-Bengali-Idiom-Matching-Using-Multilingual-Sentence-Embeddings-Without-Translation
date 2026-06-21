"""
STEP 6: Cross-Lingual Clustering  +  Monolingual Bengali Clustering
======================================================================
Cross-lingual: Clusters Bengali, Hindi, and English embeddings together
  for mSBERT, LaBSE, XLM-R. Metrics: Silhouette, Purity, Triplet Cohesion.
Monolingual: Clusters only Bengali embeddings for all models (incl.
  BanglaBERT, XLM-R) and computes silhouette score.

Outputs:
  results/cluster_assignments.csv       (cross-lingual per model)
  results/clustering_metrics.csv        (cross-lingual metrics)
  results/monolingual_clustering.csv    (Bengali-only silhouette)
  results/plots/clustering_*.png        (visualisations for cross-lingual)

Run: python src/06_clustering.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Cross-lingual models (have bn, hi, en embeddings)
CROSS_MODELS = ["mSBERT", "LaBSE", "XLM-R"]
# All models that have Bengali embeddings
ALL_BENGALI_MODELS = ["mSBERT", "LaBSE", "XLM-R", "BanglaBERT"]  


# ─────────────────────────────────────────────────────────────────────────────
def load_embeddings(model_key):
    d = EMB_DIR / model_key
    if model_key == "BanglaBERT":
        return {"bn": np.load(d / "bn.npy")}
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
def run_crosslingual_clustering(model_key: str,
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
# NEW: Monolingual Bengali clustering (all models with Bengali embeddings)
def run_monolingual_clustering(model_key: str,
                               bn_embs: np.ndarray,
                               n_clusters: int = None) -> float:
    """
    Cluster only Bengali embeddings and return silhouette score.
    """
    n = len(bn_embs)
    if n < 2:
        print(f"  [{model_key}] Not enough Bengali idioms for clustering.")
        return np.nan

    if n_clusters is None:
        n_clusters = max(2, n // 10)
    print(f"  [{model_key}] Monolingual clustering {n} Bengali idioms into {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(bn_embs)
    sil = silhouette_score(bn_embs, labels, metric="cosine",
                           sample_size=min(1000, n))
    return sil


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

    # --- Cross-lingual clustering (only mSBERT and LaBSE) ---
    all_assign = []
    all_metrics = []

    for model_key in CROSS_MODELS:
        print(f"\n── {model_key} ──────────────────────────────────────────────")
        emb    = load_embeddings(model_key)
        result = run_crosslingual_clustering(model_key, emb,
                                             bn_idioms, hi_idioms, en_idioms,
                                             triplet_df)
        all_assign.append(result.pop("assign_df"))
        all_metrics.append(result)

    # Save cross-lingual results
    pd.concat(all_assign, ignore_index=True).to_csv(
        RESULTS_DIR / "cluster_assignments.csv", index=False)
    cross_metrics_df = pd.DataFrame(all_metrics)
    cross_metrics_df.to_csv(RESULTS_DIR / "clustering_metrics.csv", index=False)

    print("\n[✓] Saved → results/cluster_assignments.csv")
    print("[✓] Saved → results/clustering_metrics.csv")
    print("\n── Cross-Lingual Clustering Metrics Summary ───────────────────")
    print(cross_metrics_df[["model","n_idioms","n_clusters","silhouette","purity","triplet_cohesion"]].to_string(index=False))

    # --- NEW: Monolingual Bengali clustering (all models) ---
    print("\n" + "=" * 60)
    print("  Monolingual Bengali Clustering")
    print("=" * 60 + "\n")

    mono_metrics = []
    for model_key in ALL_BENGALI_MODELS:
        emb_path = EMB_DIR / model_key / "bn.npy"
        if not emb_path.exists():
            print(f"[!] {model_key}: Bengali embeddings not found, skipping.")
            continue
        bn_embs = np.load(emb_path)
        sil = run_monolingual_clustering(model_key, bn_embs)
        mono_metrics.append({
            "model":        model_key,
            "lang_pair":    "bn",
            "silhouette":   round(sil, 4) if not np.isnan(sil) else sil,
        })
        if not np.isnan(sil):
            print(f"    Silhouette: {sil:.4f}")
        else:
            print(f"    Silhouette: N/A")

    if mono_metrics:
        mono_df = pd.DataFrame(mono_metrics)
        mono_df.to_csv(RESULTS_DIR / "monolingual_clustering.csv", index=False)
        print(f"\n[✓] Saved → results/monolingual_clustering.csv")
        print("\n── Monolingual Clustering Metrics Summary ───────────────────")
        print(mono_df.to_string(index=False))
    else:
        print("[!] No monolingual clustering metrics generated.")

    print("\nNext: python src/07_evaluate.py\n")
