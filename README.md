# Bengali Cross-Lingual Idiom Matching System
## Cross-lingual Idiom Semantic Similarity

This project finds semantically similar idioms between **Bengali ↔ English** and **Bengali ↔ Hindi**
using multilingual sentence embeddings — **without any translation**.

## Models Compared
- **mSBERT** — `paraphrase-multilingual-mpnet-base-v2`
- **LaBSE** — `sentence-transformers/LaBSE`

## Tasks
1. Semantic Similarity Scoring
2. Cross-lingual Idiom Retrieval (Top-K)
3. Clustering across 3 languages

## Project Structure
```
bengali_idiom_project/
├── data/
│   ├── raw/               # Downloaded datasets (place CSV files here)
│   └── processed/         # Cleaned triplet dataset
├── src/
│   ├── 01_data_loader.py      # Load & inspect all datasets
│   ├── 02_data_preprocessor.py # Clean & build triplet dataset
│   ├── 03_embedder.py          # mSBERT & LaBSE embedding pipelines
│   ├── 04_similarity.py        # Semantic similarity scoring
│   ├── 05_retrieval.py         # Cross-lingual retrieval (Top-K)
│   ├── 06_clustering.py        # Cross-lingual clustering
│   ├── 07_evaluate.py          # All evaluation metrics
|   |── run_all.py
│   └── 08_compare_models.py    # Final model comparison & plots
├── results/                   # Output CSVs and plots
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Run Full Pipeline
```bash
# Step 1: Place dataset CSVs in data/raw/ (see README for filenames)
# Step 2: Run each script in order
python src/01_data_loader.py
python src/02_data_preprocessor.py
python src/03_embedder.py
python src/04_similarity.py
python src/05_retrieval.py
python src/06_clustering.py
python src/07_evaluate.py
python src/08_compare_models.py
python src/run_all.py
```

## Expected Dataset Files in data/raw/
- `bagdhara_bengali_idioms.csv`   — from Kaggle: sakhadib/bagdhara-bangla-idioms-dataset
- `multilingual_idioms_indian.csv` — from Kaggle: aryanrahultandon/multilingual-idioms-indian
- `magpie_english_idioms.csv`      — from HuggingFace: gsarti/magpie (export to CSV)


# After all the dependencies are installed, run the following command
``` python run_all.py```