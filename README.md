
# Cross-Lingual Bengali Idiom Matching Using Multilingual Sentence Embeddings Without Translation

## Overview
This project addresses the challenge of matching Bengali idioms with their semantically similar English counterparts, bypassing the need for explicit translation. It leverages various multilingual sentence embedding models to evaluate their effectiveness across three core tasks:
1.  **Idiom Similarity:** Assessing how well models capture semantic similarity between Bengali and English idioms.
2.  **Idiom Retrieval:** Evaluating the ability of models to retrieve the correct English equivalent for a given Bengali idiom from a set of candidates.
3.  **Cross-Lingual Clustering:** Grouping idioms from both languages into semantically coherent clusters.

## Methodology
The pipeline involved:
*   **Data Loading and Preprocessing:** Acquiring and cleaning idiom datasets in Bengali, Hindi, and English.
*   **Embedding Generation:** Generating numerical representations for idioms using various multilingual sentence embedding models.
*   **Evaluation:** Performing similarity, retrieval, and clustering tasks to compare the performance of different models.
*   **Visualization:** Generating plots to illustrate model performance and clustering results.

## Models Used
The following multilingual sentence embedding models were employed for embedding generation and evaluation:
*   **mSBERT (Multilingual Sentence BERT):** A multilingual version of Sentence-BERT, suitable for various cross-lingual tasks.
*   **LaBSE (Language-Agnostic BERT Sentence Embeddings):** Specifically trained for cross-lingual semantic search and text similarity, known for its strong cross-lingual alignment.
*   **XLM-R (XLM-RoBERTa):** A large multilingual language model, fine-tuned for sentence embedding tasks.
*   **BanglaBERT:** A BERT-based model specifically trained for the Bengali language, used here to assess monolingual Bengali embedding capabilities for cross-lingual comparisons.

## Key Findings
Based on the comprehensive evaluation across all tasks, **LaBSE (Language-Agnostic BERT Sentence Embeddings)** emerged as the overall winner for cross-lingual idiom matching.

**Interpretation:**
LaBSE performs better for cross-lingual idiom matching, which aligns with its dual-encoder training specifically designed for cross-lingual alignment. mSBERT, while strong on paraphrases, may not align languages as tightly as LaBSE.

## Project Structure
The repository is organized as follows:
*   `data/`: Contains raw and processed idiom datasets.
    *   `data/raw/`: Original datasets.
    *   `data/processed/`: Cleaned and merged datasets (e.g., `bengali_merged.csv`, `clean_hindi.csv`, `clean_english.csv`).
*   `src/`: Contains the Python scripts for data loading, preprocessing, model evaluation, and visualization.
*   `results/`: Stores all evaluation outputs, metrics, and generated plots.
    *   `results/embeddings/`: Stores generated sentence embeddings for various models.
    *   `results/plots/`: Contains various visualizations of model performance and clustering (e.g., `evaluation_radar.png`, `final_comparison_crosslingual.png`, `clustering_XLM-R.png`).
    *   `results/model_comparison_summary.txt`: A summary of model performance and the overall winner.
    *   Other CSV files: `similarity_scores.csv`, `retrieval_metrics.csv`, `clustering_metrics.csv`, `full_evaluation_report.csv`, etc.
