"""
STEP 4: Semantic Similarity Scoring
=====================================
For every (Bengali, English) and (Bengali, Hindi) pair in the triplet dataset,
compute cosine similarity using mSBERT, LaBSE and XLM-R embeddings.

Also computes monolingual Bengali self-similarity for ALL models
that have Bengali embeddings (mSBERT, LaBSE, BanglaBERT, XLM-R).

Output:
  results/similarity_scores.csv              (cross-lingual pairs)
  results/monolingual_similarity.csv         (Bengali-only self-similarity)

Run: python src/04_similarity.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("../data/processed")
RESULTS_DIR   = Path("../results")
EMB_DIR       = Path("../results/embeddings")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cross‑lingual models (have bn, hi, en embeddings)
CROSS_MODELS = ["mSBERT", "LaBSE", "XLM-R"]
# All models that have Bengali embeddings (incl. BanglaBERT)
ALL_BENGALI_MODELS = ["mSBERT", "LaBSE", "XLM-R", "BanglaBERT"]  


# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def load_embeddings(model_key: str) -> dict:
    d = EMB_DIR / model_key
    if model_key == "BanglaBERT":
        return {"bn": np.load(d / "bn.npy")}
    return {lang: np.load(d / f"{lang}.npy") for lang in ["bn", "hi", "en"]}


# ─────────────────────────────────────────────────────────────────────────────
def compute_bengali_self_similarity(bn_embeddings: np.ndarray,
                                    sample_size: int = 5000) -> float:
    n = bn_embeddings.shape[0]
    if n < 2:
        return np.nan
    if n * (n - 1) // 2 > sample_size:
        idx = np.random.choice(np.arange(n, dtype=np.int64),
                               size=sample_size, replace=True)
        half = sample_size // 2
        v1 = bn_embeddings[idx[:half]]
        v2 = bn_embeddings[idx[half:2*half]]
        similarities = (v1 * v2).sum(axis=1)
        return similarities.mean()
    else:
        dot = bn_embeddings @ bn_embeddings.T
        triu = np.triu_indices(n, k=1)
        return dot[triu].mean()


def build_index(idiom_list: list, embeddings: np.ndarray) -> dict:
    return {idiom: embeddings[i] for i, idiom in enumerate(idiom_list)}


def score_pairs(pairs_df: pd.DataFrame,
                bn_index: dict, tgt_index: dict,
                model_key: str) -> pd.Series:
    scores = []
    for _, row in pairs_df.iterrows():
        src = row["idiom_src"]
        tgt = row["idiom_tgt"]
        if src in bn_index and tgt in tgt_index:
            score = cosine_similarity(bn_index[src], tgt_index[tgt])
        else:
            score = np.nan
        scores.append(score)
    return pd.Series(scores, name=f"{model_key}_score")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 4: Semantic Similarity Scoring")
    print("=" * 60 + "\n")

    bn_en = pd.read_csv(PROCESSED_DIR / "pairs_bn_en.csv")
    bn_hi = pd.read_csv(PROCESSED_DIR / "pairs_bn_hi.csv")
    print(f"[✓] Loaded  bn-en: {len(bn_en)} pairs  |  bn-hi: {len(bn_hi)} pairs\n")

    bn_df = pd.read_csv(PROCESSED_DIR / "clean_bengali.csv")
    hi_df = pd.read_csv(PROCESSED_DIR / "clean_hindi.csv")
    en_df = pd.read_csv(PROCESSED_DIR / "clean_english.csv")

    bn_col = "idiom_bn" if "idiom_bn" in bn_df.columns else bn_df.columns[0]
    hi_col = "idiom_hi" if "idiom_hi" in hi_df.columns else hi_df.columns[0]
    en_col = "idiom_en" if "idiom_en" in en_df.columns else en_df.columns[0]

    bn_idioms = bn_df[bn_col].fillna("").tolist()
    hi_idioms = hi_df[hi_col].fillna("").tolist()
    en_idioms = en_df[en_col].fillna("").tolist()

    all_results = []

    # Cross-lingual similarity scoring for all multilingual models
    for model_key in CROSS_MODELS:
        print(f"[*] Scoring cross-lingual with {model_key}...")
        emb = load_embeddings(model_key)

        bn_index = build_index(bn_idioms, emb["bn"])
        hi_index = build_index(hi_idioms, emb["hi"])
        en_index = build_index(en_idioms, emb["en"])

        bn_en_scores = score_pairs(bn_en, bn_index, en_index, model_key)
        bn_en[f"{model_key}_score"] = bn_en_scores.values

        bn_hi_scores = score_pairs(bn_hi, bn_index, hi_index, model_key)
        bn_hi[f"{model_key}_score"] = bn_hi_scores.values

        print(f"    bn-en mean score ({model_key}): {bn_en_scores.mean():.4f}")
        print(f"    bn-hi mean score ({model_key}): {bn_hi_scores.mean():.4f}\n")

    combined = pd.concat([bn_en, bn_hi], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "similarity_scores.csv", index=False)
    print(f"[✓] Saved → results/similarity_scores.csv\n")

    # Monolingual Bengali self-similarity
    print("── Bengali Monolingual Self-Similarity ───────────────────────")
    mono_rows = []
    for model_key in ALL_BENGALI_MODELS:
        emb_path = EMB_DIR / model_key / "bn.npy"
        if not emb_path.exists():
            print(f"    {model_key}: No Bengali embeddings found, skipping.")
            continue
        print(f"    Computing for {model_key}...")
        bn_emb = np.load(emb_path)
        mean_sim = compute_bengali_self_similarity(bn_emb)
        mono_rows.append({
            "model": model_key,
            "lang_pair": "bn-bn",
            "metric_type": "monolingual_similarity",
            "mean_cosine": round(mean_sim, 4),
        })
        print(f"      mean self-similarity: {mean_sim:.4f}")

    mono_df = pd.DataFrame(mono_rows)
    mono_df.to_csv(RESULTS_DIR / "monolingual_similarity.csv", index=False)
    print(f"\n[✓] Saved → results/monolingual_similarity.csv\n")

    print("── Summary: Mean Cosine Similarity (cross-lingual) ──────────")
    for lp in ["bn-en", "bn-hi"]:
        subset = combined[combined["lang_pair"] == lp]
        for m in CROSS_MODELS:
            col = f"{m}_score"
            if col in subset.columns:
                print(f"  {lp} | {m:8s}: {subset[col].mean():.4f}  (n={subset[col].notna().sum()})")
    print("\nNext: python src/05_retrieval.py\n")