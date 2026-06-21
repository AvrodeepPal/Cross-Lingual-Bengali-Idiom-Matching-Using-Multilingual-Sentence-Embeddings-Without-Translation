"""
STEP 1: Data Loader
====================
Loads all datasets.
Now uses ONLY the new hindi.json at data/raw/hindi.json
(ignores Gemma/hindi.json).
"""

import json
import pandas as pd
from pathlib import Path
import glob

RAW_DIR       = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOADER 1: Bagdhara — one JSON object per file (1.json, 2.json, ...)
# ─────────────────────────────────────────────────────────────────────────────
def load_bagdhara_folder(folder: Path = RAW_DIR) -> pd.DataFrame:
    records = []
    json_files = sorted(
        glob.glob(str(folder / "bagdhara/**/*.json"), recursive=True),
        key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else 9999
    )
    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                continue
            if "idiom" not in data:
                continue
            record = {
                "id":                    data.get("id", Path(filepath).stem),
                "idiom_bn":              data.get("idiom", "").strip(),
                "alternative_idioms":    data.get("alternative_idioms", []),
                "literal_meaning":       data.get("literal_meaning", "").strip(),
                "figurative_meaning_bn": data.get("figurative_meaning_bn", "").strip(),
                "figurative_meaning_en": data.get("figurative_meaning_en", "").strip(),
                "similar_in_english":    data.get("similar_in_english", []),
                "similar_in_english_str": "; ".join(data.get("similar_in_english", [])),
                "example_sentences_bn":  data.get("example_sentences_in_bangla", []),
                "example_sentences_en":  data.get("example_sentences_in_english", []),
                "usage_domain":          data.get("usage_domain", []),
                "tags":                  data.get("tags", []),
                "frequency":             data.get("frequency", ""),
                "sentiment":             data.get("sentiment", "neutral"),
                "cultural_significance": data.get("cultural_significance", False),
                "note":                  data.get("note", ""),
                "source":                "bagdhara",
            }
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [!] Skipped {filepath}: {e}")
    df = pd.DataFrame(records)
    print(f"[✓] Bagdhara: {len(df)} idioms loaded from {len(json_files)} JSON file(s)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOADER 2: bengali_bangla.json (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def load_bengali_bangla(filepath: Path = RAW_DIR / "Gemma" / "bengali(bangla).json") -> pd.DataFrame:
    if not Path(filepath).exists():
        print(f"[!] Not found: {filepath}")
        return pd.DataFrame()
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, entry in enumerate(data):
        records.append({
            "id":                     f"bb_{i+1}",
            "idiom_bn":               entry.get("idiom", "").strip(),
            "literal_meaning":        entry.get("literal_meaning", "").strip(),
            "figurative_meaning_en":  entry.get("figurative_meaning", "").strip(),
            "figurative_meaning_bn":  "",
            "similar_in_english":     [],
            "similar_in_english_str": "",
            "example_sentences_bn":   [entry.get("example", "")],
            "example_sentences_en":   [],
            "source":                 "bengali_bangla",
        })
    df = pd.DataFrame(records)
    print(f"[✓] Bengali-Bangla: {len(df)} idioms loaded")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOADER 3: English JSON (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def load_english_json(filepath: Path = RAW_DIR / "Gemma" / "english.json") -> pd.DataFrame:
    if not Path(filepath).exists():
        print(f"[!] Not found: {filepath}")
        return pd.DataFrame()
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, entry in enumerate(data):
        records.append({
            "idiom_en":    entry.get("idiom", "").strip(),
            "meaning_en":  entry.get("figurative_meaning", "").strip(),
            "literal_en":  entry.get("literal_meaning", "").strip(),
            "sentence_en": entry.get("example", "").strip(),
        })
    df = pd.DataFrame(records)
    print(f"[✓] English JSON: {len(df)} idioms loaded")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOADER 4: Hindi JSON – ONLY the new file at data/raw/hindi.json
# ─────────────────────────────────────────────────────────────────────────────
def load_hindi_json() -> pd.DataFrame:
    """Load ONLY the new hindi.json placed directly in data/raw/.
       Ignores Gemma/hindi.json entirely.
    """
    filepath = RAW_DIR / "hindi.json"          # <-- the new file
    if not filepath.exists():
        print(f"[!] Not found: {filepath}  (expecting your new hindi.json in data/raw/)")
        return pd.DataFrame(columns=["idiom_hi", "meaning_hi", "literal_hi", "sentence_hi"])

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expect a list of objects like the sample you provided
    records = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        records.append({
            "idiom_hi":    entry.get("idiom", "").strip(),
            "meaning_hi":  entry.get("figurative_meaning", "").strip(),
            "literal_hi":  entry.get("literal_meaning", "").strip(),
            "sentence_hi": entry.get("example", "").strip(),
        })
    df = pd.DataFrame(records).drop_duplicates(subset=["idiom_hi"]).reset_index(drop=True)
    print(f"[✓] Hindi JSON: {len(df)} idioms loaded from {filepath}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merge both Bengali sources (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def merge_bengali_sources(bagdhara_df: pd.DataFrame, bangla_df: pd.DataFrame) -> pd.DataFrame:
    common = ["id", "idiom_bn", "literal_meaning", "figurative_meaning_bn",
              "figurative_meaning_en", "similar_in_english",
              "similar_in_english_str", "example_sentences_bn", "source"]
    def ensure_cols(df):
        for col in common:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))] if col == "similar_in_english" else ""
        return df[common]
    merged = pd.concat([ensure_cols(bagdhara_df), ensure_cols(bangla_df)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["idiom_bn"]).reset_index(drop=True)
    print(f"[✓] Merged Bengali: {len(merged)} unique idioms")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 1: Loading All Datasets")
    print("=" * 60 + "\n")

    bagdhara_df = load_bagdhara_folder(RAW_DIR / "bagdhara")
    bangla_df   = load_bengali_bangla()
    hindi_df    = load_hindi_json()      # now only the new file
    english_df  = load_english_json()

    bengali_df  = merge_bengali_sources(bagdhara_df, bangla_df)

    english_df.to_csv(PROCESSED_DIR / "clean_english.csv", index=False)
    hindi_df.to_csv(  PROCESSED_DIR / "clean_hindi.csv",   index=False)
    bengali_df.to_csv(PROCESSED_DIR / "bengali_merged.csv", index=False)

    print(f"Hindi rows   : {len(hindi_df)}")
    print(f"English rows : {len(english_df)}")

    print("\n── Bengali sample (first 3) ──────────────────────────────")
    for _, row in bengali_df.head(3).iterrows():
        print(f"  idiom_bn   : {row['idiom_bn']}")
        print(f"  meaning_en : {row['figurative_meaning_en']}")
        print(f"  similar_en : {row['similar_in_english_str']}")
        print()

    print("[✓] Saved → data/processed/bengali_merged.csv")
    print("[✓] Saved → data/processed/clean_hindi.csv")
    print("[✓] Saved → data/processed/clean_english.csv")
    print("\nNext: python src/02_data_preprocessor.py\n")