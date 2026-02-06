"""
Compare human labels to LLM judge labels per failure mode.
Computes TP, TN, FP, FN, accuracy (and precision, recall, F1) and saves to human_vs_llm_comparison.json.

Run from project root: python scripts/human_vs_llm_comparison.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path for shared imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diy_repair.config import DEFAULT_OUTPUT_DIR, FILENAMES, HUMAN_LABELS_FILE, HUMAN_VS_LLM_COMPARISON_FILE
from diy_repair.failure_labeling import FAILURE_MODE_NAMES


def _project_root() -> Path:
    return _PROJECT_ROOT


def evaluate(
    failure_modes=None,
    llm_file=None,
    human_file=None,
    output_file=None,
):
    """
    Compare LLM-labeled dataset to human-labeled dataset per failure mode.
    Saves TP, TN, FP, FN, accuracy, precision, recall, F1 to output_file.
    """
    if failure_modes is None:
        failure_modes = FAILURE_MODE_NAMES
    out_dir = _project_root() / DEFAULT_OUTPUT_DIR
    llm_path = out_dir / (llm_file or FILENAMES["failure_labeled_json"])
    human_path = out_dir / (human_file or HUMAN_LABELS_FILE)
    out_path = out_dir / (output_file or HUMAN_VS_LLM_COMPARISON_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(llm_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
    with open(human_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    llm_df = pd.DataFrame(llm_data)
    human_df = pd.DataFrame(human_data)

    if "trace_id" not in llm_df.columns or "trace_id" not in human_df.columns:
        print("Both files must contain trace_id. Exiting.")
        return

    merged = llm_df[["trace_id"] + failure_modes].merge(
        human_df[["trace_id"] + failure_modes],
        on="trace_id",
        how="inner",
        suffixes=("_llm", "_human"),
    )
    if merged.empty:
        print("No samples with both LLM and human labels. Label more samples in human_labels.json.")
        return

    results = {}
    for mode in failure_modes:
        llm_col = f"{mode}_llm"
        human_col = f"{mode}_human"
        llm_labels = merged[llm_col].astype(int)
        human_labels = merged[human_col].astype(int)

        tp = int(((llm_labels == 1) & (human_labels == 1)).sum())
        tn = int(((llm_labels == 0) & (human_labels == 0)).sum())
        fp = int(((llm_labels == 1) & (human_labels == 0)).sum())
        fn = int(((llm_labels == 0) & (human_labels == 1)).sum())
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[mode] = {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Compared {len(merged)} samples. Results saved to {out_path}")
    for mode, r in results.items():
        print(f"  {mode}: accuracy={r['accuracy']:.2%}  TP={r['true_positives']} TN={r['true_negatives']} FP={r['false_positives']} FN={r['false_negatives']}")


if __name__ == "__main__":
    evaluate()
