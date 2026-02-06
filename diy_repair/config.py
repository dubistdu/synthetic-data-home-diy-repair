"""
Project configuration and shared constants.
Single source for default paths, target metrics, and file names.
"""

from pathlib import Path

# Success criterion: >80% reduction from baseline (~31.6%) → final failure rate < 6.32%
TARGET_FAILURE_RATE = 0.0632

# All pipeline outputs go here by default (JSON, CSV, PNG, reports)
DEFAULT_OUTPUT_DIR = Path("output")

# Default filenames (relative to output dir); used by main.py and quick_stats
FILENAMES = {
    "generation_results": "generation_results.json",
    "structurally_valid_qa": "structurally_valid_qa_pairs.json",
    "validation_summary": "validation_summary.json",
    "failure_labeled_csv": "failure_labeled_data.csv",
    "failure_labeled_json": "failure_labeled_data.json",
    "failure_analysis_report": "failure_analysis_report.json",
    "corrected_qa": "corrected_qa_pairs.json",
}

# Human-labeling / comparison (scripts use project root)
HUMAN_LABELS_FILE = "human_labels.json"
HUMAN_VS_LLM_COMPARISON_FILE = "human_vs_llm_comparison.json"
