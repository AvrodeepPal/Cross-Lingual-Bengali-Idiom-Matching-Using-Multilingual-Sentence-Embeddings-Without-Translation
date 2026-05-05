"""
STEP 2: Data Preprocessor & Triplet Builder
=============================================
Builds the gold triplet dataset:
    Bengali idiom ←→ English idiom ←→ Hindi idiom
    (all linked by shared figurative meaning, NO translation)

KEY INSIGHT from your data:
  The Bagdhara dataset already has "similar_in_english": ["Learning the ABCs"]
  This field IS your Bengali→English ground truth alignment.
  We use it to directly link Bengali to English idioms.

  For Hindi: we match via the shared English meaning as a conceptual bridge.

Triplet dataset schema:
  id | idiom_bn | meaning_bn | idiom_en | meaning_en | idiom_hi | meaning_hi | verified

Run: python src/02_data_preprocessor.py
"""

import csv
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text


def clean_list(val) -> list:
    """Parse a field that might be a list, string repr of list, or plain string."""
    if isinstance(val, list):
        return [clean(x) for x in val if clean(x)]
    if isinstance(val, str):
        # Try to parse as JSON list
        val = val.strip()
        if val.startswith("["):
            try:
                parsed = json.loads(val)
                return [clean(x) for x in parsed if clean(x)]
            except Exception:
                pass
        # Semicolon or comma separated
        if ";" in val:
            return [clean(x) for x in val.split(";") if clean(x)]
        return [clean(val)] if clean(val) else []
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Normalize Bengali merged CSV
# ─────────────────────────────────────────────────────────────────────────────
def normalize_bengali(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: merged Bengali CSV from Step 1
    Output: clean DataFrame with consistent columns
    """
    df = df.copy()
    df["idiom_bn"]              = df["idiom_bn"].apply(clean)
    df["figurative_meaning_en"] = df["figurative_meaning_en"].apply(clean)
    df["figurative_meaning_bn"] = df["figurative_meaning_bn"].apply(clean)
    df["similar_in_english"]    = df["similar_in_english"].apply(clean_list)
    df["example_sentences_bn"]  = df["example_sentences_bn"].apply(clean_list)

    # Drop rows without idiom text
    df = df[df["idiom_bn"].str.len() > 2].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Detect columns flexibly for Hindi dataset
# ─────────────────────────────────────────────────────────────────────────────
def detect_hindi_cols(df: pd.DataFrame) -> dict:
    """
    Try to auto-detect idiom, meaning, and English-meaning columns.
    Returns dict: {'idiom': col, 'meaning': col, 'english_meaning': col}
    """
    mapping = {}
    for col in df.columns:
        cl = col.lower()
        if not mapping.get("idiom") and any(k in cl for k in ["idiom", "phrase", "muhavara", "expression", "proverb"]):
            mapping["idiom"] = col
        elif not mapping.get("english_meaning") and any(k in cl for k in ["english", "translation", "eng_mean", "english_meaning"]):
            mapping["english_meaning"] = col
        elif not mapping.get("meaning") and any(k in cl for k in ["meaning", "arth", "अर्थ", "definition"]):
            mapping["meaning"] = col
        elif not mapping.get("language") and "lang" in cl:
            mapping["language"] = col

    # Fallback to positional
    cols = list(df.columns)
    if "idiom" not in mapping and len(cols) > 0:
        mapping["idiom"] = cols[0]
    if "meaning" not in mapping and len(cols) > 1:
        mapping["meaning"] = cols[1]

    return mapping


def normalize_hindi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes multilingual Indian dataset to standard Hindi schema.
    Filters to Hindi language rows only.
    Output columns: idiom_hi | meaning_hi | english_meaning
    """
    if df.empty:
        print("[!] Hindi dataset is empty — skipping")
        return pd.DataFrame(columns=["idiom_hi", "meaning_hi", "english_meaning"])

    col_map = detect_hindi_cols(df)
    print(f"    Detected Hindi column mapping: {col_map}")

    out = pd.DataFrame()
    out["idiom_hi"]       = df[col_map["idiom"]].apply(clean)
    out["meaning_hi"]     = df[col_map.get("meaning", col_map["idiom"])].apply(clean)
    out["english_meaning"]= df[col_map["english_meaning"]].apply(clean) if "english_meaning" in col_map else ""

    # Filter to Hindi only if language column exists
    if "language" in col_map:
        lang_col = col_map["language"]
        mask = df[lang_col].str.lower().str.contains("hindi", na=False)
        out = out[mask.values].reset_index(drop=True)

    out = out[out["idiom_hi"].str.len() > 2].reset_index(drop=True)
    print(f"[✓] Hindi idioms after normalization: {len(out)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Normalize English (MAGPIE) dataset
# ─────────────────────────────────────────────────────────────────────────────
def normalize_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes MAGPIE to: idiom_en | meaning_en | sentence_en
    """
    if df.empty:
        print("[!] English dataset is empty — skipping")
        return pd.DataFrame(columns=["idiom_en", "meaning_en"])

    out = pd.DataFrame()

    # Detect idiom column
    idiom_col = next((c for c in df.columns if any(k in c.lower() for k in ["idiom_en", "expression", "idiom", "phrase"])), df.columns[0])
    meaning_col = next((c for c in df.columns if any(k in c.lower() for k in ["meaning", "definition", "gloss"])), None)
    sentence_col = next((c for c in df.columns if any(k in c.lower() for k in ["sentence", "context", "text"])), None)

    out["idiom_en"]   = df[idiom_col].apply(clean)
    out["meaning_en"] = df[meaning_col].apply(clean) if meaning_col else ""
    out["sentence_en"]= df[sentence_col].apply(clean) if sentence_col else ""

    out = out[out["idiom_en"].str.len() > 2].drop_duplicates(subset=["idiom_en"]).reset_index(drop=True)
    print(f"[✓] English idioms after normalization: {len(out)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRIPLETS — exploiting similar_in_english field
# ─────────────────────────────────────────────────────────────────────────────
def build_triplets(bengali_df: pd.DataFrame,
                   hindi_df:   pd.DataFrame,
                   english_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds Bengali–English–Hindi triplets.

    Strategy A (HIGH CONFIDENCE — Bagdhara rows):
      The Bagdhara JSON already gives us "similar_in_english" as a list.
      e.g. Bengali: "অ আ ক খ" → similar_in_english: ["Learning the ABCs"]
      We match this directly against MAGPIE's idiom_en column.
      Then we find the matching Hindi idiom via English meaning bridge.

    Strategy B (MEDIUM CONFIDENCE — Bengali-Bangla rows):
      figurative_meaning_en field is plain English meaning text.
      We use keyword overlap to find similar MAGPIE and Hindi idioms.

    The 'confidence' column indicates which strategy was used.
    You should manually verify low-confidence triplets.
    """

    triplets = []

    def keyword_overlap(text_a: str, text_b: str) -> int:
        """Count shared meaningful words between two strings."""
        stopwords = {"a", "an", "the", "to", "in", "of", "on", "at", "by",
                     "is", "be", "as", "it", "its", "or", "and", "for",
                     "one", "ones", "someone", "something", "very", "much"}
        words_a = set(text_a.lower().split()) - stopwords
        words_b = set(text_b.lower().split()) - stopwords
        return len(words_a & words_b)

    # Pre-build Hindi lookup dict: english_meaning → row (for speed)
    hindi_lookup = {}
    if not hindi_df.empty and "english_meaning" in hindi_df.columns:
        for _, hr in hindi_df.iterrows():
            key = clean(str(hr.get("english_meaning", ""))).lower()
            if key:
                hindi_lookup[key] = hr

    def find_best_hindi(english_meaning: str):
        """Find best matching Hindi idiom for a given English meaning."""
        if not english_meaning or hindi_df.empty:
            return None, 0
        best_row, best_score = None, 0
        for key, row in hindi_lookup.items():
            score = keyword_overlap(english_meaning, key)
            if score > best_score:
                best_score, best_row = score, row
        return best_row, best_score

    def find_best_english(query: str):
        """Find best matching English idiom from MAGPIE for a query string."""
        if english_df.empty or not query:
            return None, 0
        best_row, best_score = None, 0
        for _, er in english_df.iterrows():
            en_text = f"{er['idiom_en']} {er.get('meaning_en', '')}".lower()
            score = keyword_overlap(query, en_text)
            if score > best_score:
                best_score, best_row = score, er
        return best_row, best_score

    print("[*] Building triplets...")

    for _, bn_row in bengali_df.iterrows():
        idiom_bn = bn_row["idiom_bn"]
        meaning_bn = bn_row.get("figurative_meaning_bn", "")
        meaning_en_bridge = clean(bn_row.get("figurative_meaning_en", ""))
        similar_en_list = clean_list(bn_row.get("similar_in_english", []))

        # ── Strategy A: Use similar_in_english if available ───────────────────
        # if similar_en_list:
        #     for similar_en in similar_en_list:
        #         # Find in MAGPIE or just use as-is
        #         # magpie_match, magpie_score = find_best_english(similar_en)
        #         en_idiom = similar_en
        #         en_meaning = ""
        #         magpie_score = 1
        #         # en_idiom  = magpie_match["idiom_en"]  if magpie_match is not None else similar_en
        #         # To this:
        #         en_idiom = magpie_match["idiom_en"] if magpie_match is not None else (
        #             similar_en[0] if isinstance(similar_en, list) else str(similar_en).strip("[]'\"")
        #         )
        #         en_meaning= magpie_match.get("meaning_en", "") if magpie_match is not None else ""

        #         hi_row, hi_score = find_best_hindi(meaning_en_bridge or similar_en)

        #         triplets.append({
        #             "id":          len(triplets) + 1,
        #             "idiom_bn":    idiom_bn,
        #             "meaning_bn":  meaning_bn,
        #             "idiom_en":    en_idiom,
        #             "meaning_en":  en_meaning,
        #             "idiom_hi":    hi_row["idiom_hi"]  if hi_row is not None else "",
        #             "meaning_hi":  hi_row.get("meaning_hi","") if hi_row is not None else "",
        #             "bridge_text": meaning_en_bridge,
        #             "confidence":  "HIGH" if magpie_score > 0 else "MEDIUM",
        #             "source_bn":   bn_row.get("source", ""),
        #             "verified":    False,
        #         })
        #     continue   # done with this Bengali row
        if similar_en_list:
            for similar_en in similar_en_list:
                # ✅ Direct use (NO English search)
                en_idiom = str(similar_en).strip("[]'\" ")
                en_meaning = ""   # optional: keep empty or use Bengali meaning
                magpie_score = 1  # mark as high confidence

        # Hindi matching (still needed)
                hi_row, hi_score = find_best_hindi(meaning_en_bridge or en_idiom)

                triplets.append({
                    "id":          len(triplets) + 1,
                    "idiom_bn":    idiom_bn,
                    "meaning_bn":  meaning_bn,
                    "idiom_en":    en_idiom,
                    "meaning_en":  en_meaning,
                    "idiom_hi":    hi_row["idiom_hi"] if hi_row is not None else "",
                    "meaning_hi":  hi_row.get("meaning_hi", "") if hi_row is not None else "",
                    "bridge_text": meaning_en_bridge,
                    "confidence":  "HIGH",   # always high (direct mapping)
                    "source_bn":   bn_row.get("source", ""),
                    "verified":    False,
                })
            continue

        # ── Strategy B: Keyword match via figurative_meaning_en ───────────────
        if meaning_en_bridge:
            en_row, en_score  = find_best_english(meaning_en_bridge)
            hi_row, hi_score  = find_best_hindi(meaning_en_bridge)

            if en_row is not None and en_score >= 2:
                triplets.append({
                    "id":          len(triplets) + 1,
                    "idiom_bn":    idiom_bn,
                    "meaning_bn":  meaning_bn,
                    "idiom_en":    en_row["idiom_en"],
                    "meaning_en":  en_row.get("meaning_en", ""),
                    "idiom_hi":    hi_row["idiom_hi"]   if hi_row is not None and hi_score >= 2 else "",
                    "meaning_hi":  hi_row.get("meaning_hi","") if hi_row is not None else "",
                    "bridge_text": meaning_en_bridge,
                    "confidence":  "MEDIUM",
                    "source_bn":   bn_row.get("source", ""),
                    "verified":    False,
                })

    triplet_df = pd.DataFrame(triplets)
    print(f"[✓] Built {len(triplet_df)} triplets")
    if len(triplet_df) > 0:
        print(f"    HIGH confidence : {(triplet_df['confidence']=='HIGH').sum()}")
        print(f"    MEDIUM confidence: {(triplet_df['confidence']=='MEDIUM').sum()}")
    return triplet_df


# ─────────────────────────────────────────────────────────────────────────────
# Build pairwise eval datasets from triplets
# ─────────────────────────────────────────────────────────────────────────────
def build_pairwise(triplet_df: pd.DataFrame):
    """
    Creates two pairwise datasets for model evaluation:
      - bn_en_pairs.csv : Bengali ↔ English pairs
      - bn_hi_pairs.csv : Bengali ↔ Hindi pairs
    """
    # Bengali–English pairs (drop rows without English idiom)
    bn_en = triplet_df[triplet_df["idiom_en"].str.len() > 0][
        ["id", "idiom_bn", "meaning_bn", "idiom_en", "meaning_en", "confidence", "verified"]
    ].copy()
    bn_en["lang_pair"] = "bn-en"
    bn_en.columns = ["id", "idiom_src", "meaning_src", "idiom_tgt",
                     "meaning_tgt", "confidence", "verified", "lang_pair"]

    # Bengali–Hindi pairs (drop rows without Hindi idiom)
    bn_hi = triplet_df[triplet_df["idiom_hi"].str.len() > 0][
        ["id", "idiom_bn", "meaning_bn", "idiom_hi", "meaning_hi", "confidence", "verified"]
    ].copy()
    bn_hi["lang_pair"] = "bn-hi"
    bn_hi.columns = ["id", "idiom_src", "meaning_src", "idiom_tgt",
                     "meaning_tgt", "confidence", "verified", "lang_pair"]

    print(f"[✓] bn-en pairs: {len(bn_en)}  |  bn-hi pairs: {len(bn_hi)}")
    return bn_en, bn_hi


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 2: Preprocessing & Building Triplet Dataset")
    print("=" * 60 + "\n")

    # Load data produced by Step 1 (already clean — skip normalize functions)
    bengali_raw = pd.read_csv(PROCESSED_DIR / "bengali_merged.csv")
    hindi_df    = pd.read_csv(PROCESSED_DIR / "clean_hindi.csv")
    english_df  = pd.read_csv(PROCESSED_DIR / "clean_english.csv")

    # Only Bengali still needs normalizing (it came from mixed sources)
    print("[*] Normalizing Bengali...")
    bengali_df = normalize_bengali(bengali_raw)
    bengali_df.to_csv(PROCESSED_DIR / "clean_bengali.csv", index=False)

    print(f"    Bengali : {len(bengali_df)} idioms")
    print(f"    Hindi   : {len(hindi_df)} idioms")
    print(f"    English : {len(english_df)} idioms\n")

    # Build triplets
    # triplet_df = build_triplets(bengali_df, hindi_df, english_df)
    # triplet_df.to_csv(PROCESSED_DIR / "triplets.csv", index=False)
    triplet_df = build_triplets(bengali_df, hindi_df, english_df)

# Clean up idiom_en — remove list brackets if present
    triplet_df["idiom_en"] = triplet_df["idiom_en"].astype(str).str.strip().str.strip("[]'\"")
    triplet_df["idiom_hi"] = triplet_df["idiom_hi"].astype(str).str.strip()
    triplet_df["idiom_bn"] = triplet_df["idiom_bn"].astype(str).str.strip()
    triplet_df.columns     = triplet_df.columns.str.strip()   # strip column name spaces


    triplet_df.to_csv(PROCESSED_DIR / "triplets.csv", index=False, quoting=csv.QUOTE_ALL)
    print(f"\n[✓] Saved → data/processed/triplets.csv")

    # Build pairwise eval sets
    bn_en, bn_hi = build_pairwise(triplet_df)
    bn_en.to_csv(PROCESSED_DIR / "pairs_bn_en.csv", index=False)
    bn_hi.to_csv(PROCESSED_DIR / "pairs_bn_hi.csv", index=False)
    print("[✓] Saved → data/processed/pairs_bn_en.csv")
    print("[✓] Saved → data/processed/pairs_bn_hi.csv")

    print("\n[!] ACTION REQUIRED:")
    print("    Open data/processed/triplets.csv and manually set 'verified=True'")
    print("    for triplets you confirm are semantically equivalent.")
    print("\nNext: python src/03_embedder.py\n")