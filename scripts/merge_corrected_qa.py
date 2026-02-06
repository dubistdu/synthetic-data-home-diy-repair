"""
Merge corrected Q&A pairs back into the dataset for re-labeling.
Reads: structurally_valid_qa_pairs.json, failure_labeled_data.json, corrected_qa_pairs.json.
Builds: one record per trace_id; failed items replaced by valid corrected content.
Saves: qa_after_correction.json (same structure as structurally_valid_qa_pairs).

Run from project root: python scripts/merge_corrected_qa.py
Then: python main.py --labeling-only --input-qa output/qa_after_correction.json
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diy_repair.config import DEFAULT_OUTPUT_DIR, FILENAMES


def main():
    out_dir = _PROJECT_ROOT / DEFAULT_OUTPUT_DIR
    valid_path = out_dir / FILENAMES["structurally_valid_qa"]
    failure_path = out_dir / FILENAMES["failure_labeled_json"]
    corrected_path = out_dir / FILENAMES["corrected_qa"]
    output_path = out_dir / "qa_after_correction.json"

    for p, name in [(valid_path, "structurally_valid_qa_pairs"), (failure_path, "failure_labeled_data"), (corrected_path, "corrected_qa")]:
        if not p.exists():
            print(f"Missing {name}: {p}")
            return 1

    with open(valid_path, "r", encoding="utf-8") as f:
        valid_qa = json.load(f)
    with open(failure_path, "r", encoding="utf-8") as f:
        failure_labeled = json.load(f)
    with open(corrected_path, "r", encoding="utf-8") as f:
        corrected_list = json.load(f)

    failed_ids = {r["trace_id"] for r in failure_labeled if r.get("overall_failure") == 1}
    corrected_by_id = {}
    for r in corrected_list:
        tid = r.get("trace_id")
        if tid and r.get("is_valid") and r.get("qa_pair"):
            corrected_by_id[tid] = {**r["qa_pair"], "trace_id": tid}

    merged = []
    for rec in valid_qa:
        trace_id = rec.get("trace_id")
        if trace_id in failed_ids and trace_id in corrected_by_id:
            merged.append(corrected_by_id[trace_id])
        else:
            merged.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    replaced = sum(1 for rec in valid_qa if rec.get("trace_id") in failed_ids and rec.get("trace_id") in corrected_by_id)
    print(f"Merged {len(merged)} Q&A pairs ({replaced} replaced with corrected versions).")
    print(f"Saved to {output_path}")
    print("Re-label with: python main.py --labeling-only --input-qa output/qa_after_correction.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
