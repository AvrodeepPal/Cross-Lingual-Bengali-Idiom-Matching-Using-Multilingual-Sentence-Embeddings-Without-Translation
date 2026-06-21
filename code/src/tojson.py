import json
from pathlib import Path

# 🔴 UPDATE THIS PATH EXACTLY TO YOUR FOLDER
BAGDHARA_DIR = Path("../data/raw/bagdhara")

OUTPUT_FILE = Path("../data/raw/english_from_bengali.json")

english_set = set()
english_data = []

# 🔥 recursive read (important)
json_files = list(BAGDHARA_DIR.glob("**/*.json"))

print(f"[INFO] Found {len(json_files)} files")

for file in json_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure correct format
        if not isinstance(data, dict):
            continue

        similar_list = data.get("similar_in_english", [])

        if not similar_list:
            continue

        for idiom in similar_list:
            idiom_clean = str(idiom).strip()

            if not idiom_clean:
                continue

            # remove duplicates
            if idiom_clean.lower() in english_set:
                continue

            english_set.add(idiom_clean.lower())

            english_data.append({
                "idiom": idiom_clean,
                "literal_meaning": "",
                "figurative_meaning": data.get("figurative_meaning_en", ""),
                "example": "",
                "language": "English"
            })

    except Exception as e:
        print(f"[ERROR] {file}: {e}")

# Save file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(english_data, f, indent=2, ensure_ascii=False)

print(f"[✓] Extracted {len(english_data)} English idioms")