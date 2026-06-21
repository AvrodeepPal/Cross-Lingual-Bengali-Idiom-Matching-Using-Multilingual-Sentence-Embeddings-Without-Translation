"""
run_all.py
===========
Runs the entire pipeline end-to-end in order.
Each step is only run if its output does not already exist.

Usage:
    python run_all.py           # full pipeline
    python run_all.py --from 4  # resume from Step 4
"""

import subprocess
import sys
from pathlib import Path

STEPS = [
    (1, "data_loader.py",       "../data/processed/bengali_merged.csv"),
    (2, "data_preprocessor.py", "../data/processed/triplets.csv"),
    (3, "03_embedder.py",          "../results/embeddings/LaBSE/bn.npy"),
    (4, "04_similarity.py",        "../results/similarity_scores.csv"),
    (5, "05_retrieval.py",         "../results/retrieval_metrics.csv"),
    (6, "06_clustering.py",        "../results/clustering_metrics.csv"),
    (7, "07_evaluate.py",          "../results/full_evaluation_report.csv"),
    (8, "08_compare_models.py",    "../results/model_comparison_summary.txt"),
]

def main():
    start_from = 1
    if "--from" in sys.argv:
        idx = sys.argv.index("--from")
        start_from = int(sys.argv[idx + 1])

    print("\n" + "=" * 60)
    print("  Bengali Cross-Lingual Idiom Matching — Full Pipeline")
    print("=" * 60 + "\n")

    for step_num, script, output_check in STEPS:
        if step_num < start_from:
            print(f"[SKIP] Step {step_num} (--from {start_from})")
            continue

        output_exists = Path(output_check).exists()
        if output_exists and step_num not in (3,):  # always re-run embedding if forced
            print(f"[DONE] Step {step_num}: output exists → {output_check}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Running Step {step_num}: {script}")
        print(f"{'─'*60}\n")

        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"\n[ERROR] Step {step_num} failed (exit code {result.returncode})")
            print(f"        Fix the error above, then re-run: python run_all.py --from {step_num}")
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 60)
    print("\nKey output files:")
    print("  data/processed/triplets.csv              ← your dataset")
    print("  results/similarity_scores.csv            ← Task 1 results")
    print("  results/retrieval_metrics.csv            ← Task 2 results")
    print("  results/clustering_metrics.csv           ← Task 3 results")
    print("  results/full_evaluation_report.csv       ← all metrics")
    print("  results/model_comparison_summary.txt     ← winner verdict")
    print("  results/plots/final_comparison.png       ← comparison chart\n")


if __name__ == "__main__":
    main()
