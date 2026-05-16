"""
STEP 3: Embedder
=================
Generates embeddings for all idioms using FOUR models:
  Model 1: mSBERT    → paraphrase-multilingual-mpnet-base-v2
  Model 2: LaBSE     → sentence-transformers/LaBSE
  Model 3: XLM-R     → FacebookAI/xlm-roberta-base  (multilingual)
  Model 4: BanglaBERT → csebuetnlp/banglabert  (Bengali only)

mSBERT, LaBSE & XLM-R embed Bengali, Hindi, and English into the SAME vector space.
BanglaBERT is monolingual (Bengali only) and is NOT used for cross-lingual tasks.

Embeddings are saved as .npy files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time
import torch

PROCESSED_DIR = Path("../data/processed")
RESULTS_DIR   = Path("../results/embeddings")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cross-lingual models (embed all three languages)
MULTILINGUAL_MODELS = {
    "mSBERT": "paraphrase-multilingual-mpnet-base-v2",
    "LaBSE":  "sentence-transformers/LaBSE",
    "XLM-R":  "FacebookAI/xlm-roberta-base",
}

# Monolingual Bengali model (only used for Bengali)
BANGLA_MODEL = "csebuetnlp/banglabert"


# ─────────────────────────────────────────────────────────────────────────────
class IdiomEmbedder:
    def __init__(self, model_key: str, model_name: str, is_banglabert: bool = False):
        print(f"[*] Loading {model_key} ({model_name})...")
        t = time.time()
        self.model      = SentenceTransformer(model_name)
        self.model_key  = model_key
        self.is_banglabert = is_banglabert
        print(f"[✓] {model_key} ready in {time.time()-t:.1f}s")

    def embed(self, texts: list) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_all(self, bn_texts, hi_texts=None, en_texts=None) -> dict:
        """
        For multilingual models: embed all three languages.
        For BanglaBERT: embed only Bengali, hi & en are ignored.
        """
        print(f"\n── {self.model_key}: Bengali ({len(bn_texts)}) ──")
        emb_bn = self.embed(bn_texts)

        if self.is_banglabert:
            print(f"── {self.model_key}: Hindi & English skipped (monolingual) ──")
            return {"bn": emb_bn}

        print(f"── {self.model_key}: Hindi ({len(hi_texts)}) ──")
        emb_hi = self.embed(hi_texts)
        print(f"── {self.model_key}: English ({len(en_texts)}) ──")
        emb_en = self.embed(en_texts)
        return {"bn": emb_bn, "hi": emb_hi, "en": emb_en}


# ─────────────────────────────────────────────────────────────────────────────
def save_embeddings(embeddings: dict, model_key: str):
    out = RESULTS_DIR / model_key
    out.mkdir(parents=True, exist_ok=True)
    for lang, arr in embeddings.items():
        np.save(out / f"{lang}.npy", arr)
    print(f"[✓] {model_key} embeddings saved → results/embeddings/{model_key}/")


def load_embeddings(model_key: str) -> dict:
    d = RESULTS_DIR / model_key
    if model_key == "BanglaBERT":
        # Only Bengali embedding exists
        return {"bn": np.load(d / "bn.npy")}
    return {lang: np.load(d / f"{lang}.npy") for lang in ["bn", "hi", "en"]}


def embeddings_exist(model_key: str) -> bool:
    d = RESULTS_DIR / model_key
    if model_key == "BanglaBERT":
        return (d / "bn.npy").exists()
    return all((d / f"{lang}.npy").exists() for lang in ["bn", "hi", "en"])


# ─────────────────────────────────────────────────────────────────────────────
def load_idiom_lists():
    bn_df = pd.read_csv(PROCESSED_DIR / "clean_bengali.csv")
    hi_df = pd.read_csv(PROCESSED_DIR / "clean_hindi.csv")
    en_df = pd.read_csv(PROCESSED_DIR / "clean_english.csv")

    bn_col = "idiom_bn" if "idiom_bn" in bn_df.columns else bn_df.columns[0]
    hi_col = "idiom_hi" if "idiom_hi" in hi_df.columns else hi_df.columns[0]
    en_col = "idiom_en" if "idiom_en" in en_df.columns else en_df.columns[0]

    bn = bn_df[bn_col].fillna("").tolist()
    hi = hi_df[hi_col].fillna("").tolist()
    en = en_df[en_col].fillna("").tolist()

    print(f"[✓] Idiom counts  →  Bengali: {len(bn)}  Hindi: {len(hi)}  English: {len(en)}\n")
    return bn, hi, en, bn_df, hi_df, en_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 3: Generating Embeddings  (mSBERT, LaBSE, XLM-R & BanglaBERT)")
    print("=" * 60 + "\n")

    bn_texts, hi_texts, en_texts, *_ = load_idiom_lists()

    # 1. Multilingual models (cross-lingual)
    for key, name in MULTILINGUAL_MODELS.items():
        if embeddings_exist(key):
            print(f"[→] {key} embeddings already exist — delete folder to recompute.\n")
            continue
        embedder   = IdiomEmbedder(key, name, is_banglabert=False)
        embeddings = embedder.embed_all(bn_texts, hi_texts, en_texts)
        save_embeddings(embeddings, key)
        for lang, arr in embeddings.items():
            print(f"    {key}[{lang}] shape: {arr.shape}")
        print()

    # 2. BanglaBERT (Bengali only)
    key = "BanglaBERT"
    if embeddings_exist(key):
        print(f"[→] {key} embeddings already exist — delete folder to recompute.\n")
    else:
        embedder   = IdiomEmbedder(key, BANGLA_MODEL, is_banglabert=True)
        embeddings = embedder.embed_all(bn_texts)   # no hi/en arguments needed
        save_embeddings(embeddings, key)
        print(f"    {key}[bn] shape: {embeddings['bn'].shape}")
        print()

    print("[✓] All embeddings done.\nNext: python src/04_similarity.py\n")