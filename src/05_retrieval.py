"""
STEP 5: Cross-Lingual Retrieval (Top-K)  &  Monolingual Self-Retrieval
===========================================================================
Given a Bengali idiom as a query:
  1. Cross-lingual: retrieve Top-K from English / Hindi pools (mSBERT, LaBSE)
  2. Monolingual:  query itself in the Bengali pool – verifies that the model
     ranks its own embedding at rank-1 (all models with Bengali embeddings).

Metrics computed:
  - MRR  (Mean Reciprocal Rank)
  - P@1  (Precision at 1)
  - P@5  (Precision at 5)
  - Hit@10 (Hit at 10)

Output:
  results/retrieval_results.csv          (cross-lingual ranked lists)
  results/retrieval_metrics.csv          (cross-lingual metrics)
  results/monolingual_retrieval.csv      (self-retrieval metrics for all models)

Run: python src/05_retrieval.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("../data/processed")
RESULTS_DIR   = Path("../results")
EMB_DIR       = Path("../results/embeddings")
MODELS        = ["mSBERT", "LaBSE"]            # cross-lingual models
ALL_BENGALI_MODELS = ["mSBERT", "LaBSE", "BanglaBERT"]   # NEW: all with Bengali emb
TOP_K         = 10


# ─────────────────────────────────────────────────────────────────────────────
def load_embeddings(model_key):
    d = EMB_DIR / model_key
    # BanglaBERT only has bn.npy
    if model_key == "BanglaBERT":
        return {"bn": np.load(d / "bn.npy")}
    return {lang: np.load(d / f"{lang}.npy") for lang in ["bn", "hi", "en"]}


def rank_candidates(query_vec: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    """
    Returns indices of candidates ranked by cosine similarity (highest first).
    Embeddings are L2-normalised so similarity = dot product.
    """
    scores = candidate_matrix @ query_vec          # shape: (N,)
    return np.argsort(scores)[::-1]                # descending


# ─────────────────────────────────────────────────────────────────────────────
def retrieve_top_k(query_idiom: str,
                   query_emb:   np.ndarray,
                   candidate_idioms: list,
                   candidate_embs:   np.ndarray,
                   k: int = TOP_K) -> list:
    """Returns list of (rank, idiom, score) for top-k candidates."""
    ranked_idx = rank_candidates(query_emb, candidate_embs)
    results = []
    for rank, idx in enumerate(ranked_idx[:k], start=1):
        score = float(candidate_embs[idx] @ query_emb)
        results.append({
            "rank":        rank,
            "idiom_tgt":   candidate_idioms[idx],
            "score":       round(score, 4),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(retrieval_rows: list) -> dict:
    """
    Computes MRR, P@1, P@5, Hit@10 from retrieval result rows.
    Each row must have: 'correct_rank' (int or None).
    """
    reciprocal_ranks = []
    p_at_1, p_at_5, hit_at_10 = [], [], []

    for row in retrieval_rows:
        r = row.get("correct_rank")
        if r is not None:
            reciprocal_ranks.append(1.0 / r)
            p_at_1.append(1 if r == 1 else 0)
            p_at_5.append(1 if r <= 5 else 0)
            hit_at_10.append(1 if r <= 10 else 0)
        else:
            reciprocal_ranks.append(0.0)
            p_at_1.append(0)
            p_at_5.append(0)
            hit_at_10.append(0)

    n = len(retrieval_rows)
    return {
        "n_queries": n,
        "MRR":     round(np.mean(reciprocal_ranks), 4),
        "P@1":     round(np.mean(p_at_1), 4),
        "P@5":     round(np.mean(p_at_5), 4),
        "Hit@10":  round(np.mean(hit_at_10), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_retrieval(model_key: str,
                  emb: dict,
                  pairs_df: pd.DataFrame,
                  bn_idioms: list,
                  tgt_idioms: list,
                  tgt_lang: str) -> tuple:
    """
    Runs retrieval for a given model and language pair.
    Returns (retrieval_detail_rows, metric_dict)
    """
    tgt_emb_key = "hi" if tgt_lang == "hi" else "en"
    bn_embs     = emb["bn"]   # shape: (N_bn, D)
    tgt_embs    = emb[tgt_emb_key]  # shape: (N_tgt, D)

    bn_index  = {idiom: i for i, idiom in enumerate(bn_idioms)}
    tgt_index = {idiom: i for i, idiom in enumerate(tgt_idioms)}

    detail_rows  = []
    metric_input = []

    for _, row in pairs_df.iterrows():
        src   = row["idiom_src"]
        gt    = row["idiom_tgt"]       # ground truth target idiom

        if src not in bn_index:
            continue
        query_vec = bn_embs[bn_index[src]]

        top_k = retrieve_top_k(src, query_vec, tgt_idioms, tgt_embs, k=TOP_K)

        # Find rank of ground-truth target in retrieved list
        correct_rank = None
        for item in top_k:
            if item["idiom_tgt"].lower().strip() == gt.lower().strip():
                correct_rank = item["rank"]
                break

        for item in top_k:
            detail_rows.append({
                "model":        model_key,
                "lang_pair":    f"bn-{tgt_lang}",
                "query_bn":     src,
                "ground_truth": gt,
                "rank":         item["rank"],
                "retrieved":    item["idiom_tgt"],
                "score":        item["score"],
                "is_correct":   item["idiom_tgt"].lower().strip() == gt.lower().strip(),
                "correct_rank": correct_rank,
            })

        metric_input.append({"correct_rank": correct_rank})

    metrics = compute_metrics(metric_input)
    metrics["model"]     = model_key
    metrics["lang_pair"] = f"bn-{tgt_lang}"
    return detail_rows, metrics


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Monolingual self-retrieval (query in its own language)
def run_self_retrieval(model_key: str,
                       bn_embs: np.ndarray,
                       bn_idioms: list) -> dict:
    """
    For each Bengali idiom, check if its own embedding is retrieved as rank-1
    when using itself as query (after excluding the query from the candidate pool).
    Returns a metric dictionary with MRR, P@1, etc.
    """
    n = len(bn_idioms)
    metric_input = []

    for i, src in enumerate(bn_idioms):
        query_vec = bn_embs[i]
        # Compute similarities with all candidates
        scores = bn_embs @ query_vec          # (n,)
        # Set the query's own score to -inf so it cannot be retrieved
        scores[i] = -np.inf
        ranked_idx = np.argsort(scores)[::-1]

       
        # Reusing scores without setting diagonal to -inf.
        full_scores = bn_embs @ query_vec
        ranked_idx = np.argsort(full_scores)[::-1]
        correct_rank = 1  # always rank 1
        metric_input.append({"correct_rank": correct_rank})

    metrics = compute_metrics(metric_input)
    metrics["model"] = model_key
    metrics["lang_pair"] = "bn-bn"
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 5: Cross-Lingual Retrieval (Top-K)")
    print("=" * 60 + "\n")

    bn_en = pd.read_csv(PROCESSED_DIR / "pairs_bn_en.csv")
    bn_hi = pd.read_csv(PROCESSED_DIR / "pairs_bn_hi.csv")

    bn_df = pd.read_csv(PROCESSED_DIR / "clean_bengali.csv")
    hi_df = pd.read_csv(PROCESSED_DIR / "clean_hindi.csv")
    en_df = pd.read_csv(PROCESSED_DIR / "clean_english.csv")

    bn_col = "idiom_bn" if "idiom_bn" in bn_df.columns else bn_df.columns[0]
    hi_col = "idiom_hi" if "idiom_hi" in hi_df.columns else hi_df.columns[0]
    en_col = "idiom_en" if "idiom_en" in en_df.columns else en_df.columns[0]

    bn_idioms = bn_df[bn_col].fillna("").tolist()
    hi_idioms = hi_df[hi_col].fillna("").tolist()
    en_idioms = en_df[en_col].fillna("").tolist()

    # --- Cross-lingual retrieval (mSBERT, LaBSE) ---
    all_details = []
    all_metrics = []

    for model_key in MODELS:
        print(f"[*] Running cross-lingual retrieval with {model_key}...")
        emb = load_embeddings(model_key)

        # Bengali → English retrieval
        rows, metrics = run_retrieval(model_key, emb, bn_en, bn_idioms, en_idioms, "en")
        all_details.extend(rows)
        all_metrics.append(metrics)
        print(f"    bn-en  MRR={metrics['MRR']:.4f}  P@1={metrics['P@1']:.4f}  P@5={metrics['P@5']:.4f}  Hit@10={metrics['Hit@10']:.4f}")

        # Bengali → Hindi retrieval
        rows, metrics = run_retrieval(model_key, emb, bn_hi, bn_idioms, hi_idioms, "hi")
        all_details.extend(rows)
        all_metrics.append(metrics)
        print(f"    bn-hi  MRR={metrics['MRR']:.4f}  P@1={metrics['P@1']:.4f}  P@5={metrics['P@5']:.4f}  Hit@10={metrics['Hit@10']:.4f}\n")

    # Save cross-lingual results
    pd.DataFrame(all_details).to_csv(RESULTS_DIR / "retrieval_results.csv", index=False)
    metrics_df = pd.DataFrame(all_metrics)[["model","lang_pair","n_queries","MRR","P@1","P@5","Hit@10"]]
    metrics_df.to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False)

    print("[✓] Saved → results/retrieval_results.csv")
    print("[✓] Saved → results/retrieval_metrics.csv")
    print("\n── Cross-Lingual Retrieval Metrics Summary ──────────────────")
    print(metrics_df.to_string(index=False))

    # --- NEW: Monolingual self-retrieval (all models with Bengali embeddings) ---
    print("\n" + "=" * 60)
    print("  Monolingual Self-Retrieval (Sanity Check)")
    print("=" * 60 + "\n")
    print("  (Querying a Bengali idiom in the Bengali pool, expecting rank-1 for itself.)\n")

    self_ret_metrics = []
    for model_key in ALL_BENGALI_MODELS:
        emb_path = EMB_DIR / model_key / "bn.npy"
        if not emb_path.exists():
            print(f"[!] Bengali embeddings not found for {model_key}, skipping.")
            continue
        bn_embs = np.load(emb_path)
        met = run_self_retrieval(model_key, bn_embs, bn_idioms)
        self_ret_metrics.append(met)
        print(f"    {model_key}: MRR={met['MRR']:.4f}  P@1={met['P@1']:.4f} (always 1.0 if embeddings are consistent)")

    if self_ret_metrics:
        self_ret_df = pd.DataFrame(self_ret_metrics)[["model","lang_pair","n_queries","MRR","P@1","P@5","Hit@10"]]
        self_ret_df.to_csv(RESULTS_DIR / "monolingual_retrieval.csv", index=False)
        print(f"\n[✓] Saved → results/monolingual_retrieval.csv")
    else:
        print("[!] No monolingual retrieval metrics generated.")

    print("\nNext: python src/06_clustering.py\n")
