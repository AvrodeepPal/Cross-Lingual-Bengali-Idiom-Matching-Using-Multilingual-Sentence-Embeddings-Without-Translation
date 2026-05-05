"""
STEP 1: Data Loader
====================
Loads all datasets matching the EXACT schemas found in your JSON files.

Schema 1 — Bagdhara (one JSON object per file, e.g. 1.json, 2.json ...):
{
  "id", "idiom", "alternative_idioms", "literal_meaning",
  "figurative_meaning_bn", "figurative_meaning_en",
  "similar_in_english": [...],          <-- LIST of English equivalents ← GOLD
  "example_sentences_in_bangla": [...],
  "example_sentences_in_english": [...],
  "usage_domain", "tags", "frequency", "sentiment", ...
}

Schema 2 — bengali_bangla.json (array of objects):
{
  "idiom", "literal_meaning", "figurative_meaning", "example", "language"
}

Run: python src/01_data_loader.py
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
    """
    Scans the folder for all single-object JSON files (1.json, 2.json ...).
    Each file is ONE idiom entry.

    Key fields:
      - idiom                    → Bengali idiom text
      - figurative_meaning_bn    → meaning in Bengali
      - figurative_meaning_en    → meaning in English (bridge for matching)
      - similar_in_english       → list of English idioms ← THIS IS GOLD FOR YOUR PROJECT
      - example_sentences_in_bangla → usage examples
    """
    records = []

    json_files = sorted(
        glob.glob(str(folder / "**/*.json"), recursive=True),
        key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else 9999
    )

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Skip array-style files (bengali_bangla.json)
            if isinstance(data, list):
                continue

            # Skip if no idiom field
            if "idiom" not in data:
                continue

            record = {
                "id":                    data.get("id", Path(filepath).stem),
                "idiom_bn":              data.get("idiom", "").strip(),
                "alternative_idioms":    data.get("alternative_idioms", []),
                "literal_meaning":       data.get("literal_meaning", "").strip(),
                "figurative_meaning_bn": data.get("figurative_meaning_bn", "").strip(),
                "figurative_meaning_en": data.get("figurative_meaning_en", "").strip(),
                # similar_in_english is a LIST — very useful for automatic matching
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
# LOADER 2: bengali_bangla.json — array of objects
# ─────────────────────────────────────────────────────────────────────────────
def load_bengali_bangla(filepath: Path = RAW_DIR / "bengali_bangla.json") -> pd.DataFrame:
    """
    Loads the array-style Bengali idiom JSON.

    Schema per entry:
      { "idiom", "literal_meaning", "figurative_meaning", "example", "language" }

    NOTE: figurative_meaning here is in ENGLISH — useful as matching bridge.
    """
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
            "figurative_meaning_bn":  "",   # not in this dataset
            "similar_in_english":     [],   # not in this dataset
            "similar_in_english_str": "",
            "example_sentences_bn":   [entry.get("example", "")],
            "example_sentences_en":   [],
            "source":                 "bengali_bangla",
        })

    df = pd.DataFrame(records)
    print(f"[✓] Bengali-Bangla: {len(df)} idioms loaded")
    return df

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


def load_hindi_json(filepath: Path = RAW_DIR / "Gemma" / "hindi.json") -> pd.DataFrame:
    if not Path(filepath).exists():
        print(f"[!] Not found: {filepath}")
        return pd.DataFrame()
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, entry in enumerate(data):
        records.append({
            "idiom_hi":    entry.get("idiom", "").strip(),
            "meaning_hi":  entry.get("figurative_meaning", "").strip(),
            "literal_hi":  entry.get("literal_meaning", "").strip(),
            "sentence_hi": entry.get("example", "").strip(),
        })
    df = pd.DataFrame(records)
    print(f"[✓] Hindi JSON: {len(df)} idioms loaded")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# LOADER 3: Multilingual Indian idioms (Kaggle CSV)
# ─────────────────────────────────────────────────────────────────────────────
def load_multilingual_indian(filepath: Path = RAW_DIR / "multilingual_idioms_indian.csv") -> pd.DataFrame:
    """
    Loads the Kaggle multilingual Indian idioms dataset.
    Download: https://www.kaggle.com/datasets/aryanrahultandon/multilingual-idioms-indian
    Save as:  data/raw/multilingual_idioms_indian.csv
    """
    fp = Path(filepath)
    if not fp.exists():
        # Try JSON fallback
        json_fp = RAW_DIR / "multilingual_idioms_indian.json"
        if json_fp.exists():
            with open(json_fp) as f:
                data = json.load(f)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            print(f"[✓] Multilingual Indian (JSON): {len(df)} rows")
            return df
        else:
            print(f"[!] Not found: {fp}")
            print("    Download from Kaggle and save to data/raw/multilingual_idioms_indian.csv")
            return pd.DataFrame()

    df = pd.read_csv(fp)
    print(f"[✓] Multilingual Indian (CSV): {len(df)} rows | columns: {list(df.columns)}")
    if "language" in df.columns:
        print(f"    Language counts: {df['language'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOADER 4: MAGPIE English Idioms (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────
def load_magpie(use_huggingface: bool = True) -> pd.DataFrame:
    """
    Loads MAGPIE English idioms dataset.
    Source: https://huggingface.co/datasets/gsarti/magpie
    Cached locally after first download.
    """
    cache_path = RAW_DIR / "magpie_english.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"[✓] MAGPIE (cache): {len(df)} English idioms")
        return df

    if use_huggingface:
        try:
            from datasets import load_dataset
            print("[*] Downloading MAGPIE from HuggingFace (first time, may take a few minutes)...")
            dataset = load_dataset("gsarti/magpie", "magpie", trust_remote_code=True)
            df = dataset["train"].to_pandas()

            # Keep only figurative instances (label=1 means idiomatic usage)
            if "label" in df.columns:
                df = df[df["label"] == 1].copy()

            # Deduplicate on expression (idiom)
            idiom_col = "expression" if "expression" in df.columns else df.columns[0]
            df = df.drop_duplicates(subset=[idiom_col]).reset_index(drop=True)
            df = df.rename(columns={idiom_col: "idiom_en"})

            df.to_csv(cache_path, index=False)
            print(f"[✓] MAGPIE: {len(df)} unique English idioms (saved to cache)")
            return df

        except Exception as e:
            print(f"[!] HuggingFace load failed: {e}")

    print("[!] MAGPIE unavailable. Save magpie_english.csv to data/raw/")
    return pd.DataFrame(columns=["idiom_en"])


# ─────────────────────────────────────────────────────────────────────────────
# Merge both Bengali sources
# ─────────────────────────────────────────────────────────────────────────────
def merge_bengali_sources(bagdhara_df: pd.DataFrame, bangla_df: pd.DataFrame) -> pd.DataFrame:
    """Combine Bagdhara + Bengali-Bangla, deduplicate on idiom text."""
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
    bangla_df   = load_bengali_bangla(RAW_DIR / "Gemma" / "bengali(bangla).json")
    # # data\raw\Gemma\bengali(bangla).json
    # data\raw\Gemma\english.json
    # hindi_df    = load_multilingual_indian() 
    # english_df  = load_magpie()
    hindi_df   = load_hindi_json()
    english_df = load_english_json()

    bengali_df  = merge_bengali_sources(bagdhara_df, bangla_df)
    # print("Bagdhara rows:", len(bagdhara_df))
    # print("Bangla rows:", len(bangla_df))

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