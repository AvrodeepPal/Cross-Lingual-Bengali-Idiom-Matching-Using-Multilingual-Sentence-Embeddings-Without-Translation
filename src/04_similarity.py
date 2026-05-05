"""
STEP 4: Semantic Similarity Scoring
=====================================
For every (Bengali, English) and (Bengali, Hindi) pair in the triplet dataset,
compute cosine similarity using both mSBERT and LaBSE embeddings.

Output: results/similarity_scores.csv
  columns: id | idiom_bn | idiom_tgt | lang_pair | mSBERT_score | LaBSE_score | confidence

Run: python src/04_similarity.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("../data/processed")
RESULTS_DIR   = Path("../results")
EMB_DIR       = Path("../results/embeddings")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["mSBERT", "LaBSE"]


# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since embeddings are already normalised in Step 3, this is just dot product.
    """
    return float(np.dot(a, b))


def load_embeddings(model_key: str) -> dict:
    d = EMB_DIR / model_key
    return {lang: np.load(d / f"{lang}.npy") for lang in ["bn", "hi", "en"]}


# ─────────────────────────────────────────────────────────────────────────────
def build_index(idiom_list: list, embeddings: np.ndarray) -> dict:
    """Map each idiom string → its embedding vector."""
    return {idiom: embeddings[i] for i, idiom in enumerate(idiom_list)}


# ─────────────────────────────────────────────────────────────────────────────
def score_pairs(pairs_df: pd.DataFrame,
                bn_index: dict, tgt_index: dict,
                model_key: str) -> pd.Series:
    """
    Compute similarity score for each (idiom_src, idiom_tgt) pair.
    Returns a Series of scores aligned to pairs_df index.
    """
    scores = []
    for _, row in pairs_df.iterrows():
        src = row["idiom_src"]
        tgt = row["idiom_tgt"]
        if src in bn_index and tgt in tgt_index:
            score = cosine_similarity(bn_index[src], tgt_index[tgt])
        else:
            score = np.nan   # idiom not found in embedding index
        scores.append(score)
    return pd.Series(scores, name=f"{model_key}_score")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 4: Semantic Similarity Scoring")
    print("=" * 60 + "\n")

    # Load pairwise datasets
    bn_en = pd.read_csv(PROCESSED_DIR / "pairs_bn_en.csv")
    bn_hi = pd.read_csv(PROCESSED_DIR / "pairs_bn_hi.csv")
    print(f"[✓] Loaded  bn-en: {len(bn_en)} pairs  |  bn-hi: {len(bn_hi)} pairs\n")

    # Load idiom lists (same order as embeddings)
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

    for model_key in MODELS:
        print(f"[*] Scoring with {model_key}...")
        emb = load_embeddings(model_key)

        bn_index = build_index(bn_idioms, emb["bn"])
        hi_index = build_index(hi_idioms, emb["hi"])
        en_index = build_index(en_idioms, emb["en"])

        # Score bn-en pairs
        bn_en_scores = score_pairs(bn_en, bn_index, en_index, model_key)
        bn_en[f"{model_key}_score"] = bn_en_scores.values

        # Score bn-hi pairs
        bn_hi_scores = score_pairs(bn_hi, bn_index, hi_index, model_key)
        bn_hi[f"{model_key}_score"] = bn_hi_scores.values

        print(f"    bn-en mean score ({model_key}): {bn_en_scores.mean():.4f}")
        print(f"    bn-hi mean score ({model_key}): {bn_hi_scores.mean():.4f}\n")

    # Combine into one results file
    combined = pd.concat([bn_en, bn_hi], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "similarity_scores.csv", index=False)
    print(f"[✓] Saved → results/similarity_scores.csv\n")

    # Summary table
    print("── Summary: Mean Cosine Similarity ──────────────────────────")
    for lp in ["bn-en", "bn-hi"]:
        subset = combined[combined["lang_pair"] == lp]
        for m in MODELS:
            col = f"{m}_score"
            if col in subset.columns:
                print(f"  {lp} | {m:8s}: {subset[col].mean():.4f}  (n={subset[col].notna().sum()})")
    print()
    print("Next: python src/05_retrieval.py\n")
