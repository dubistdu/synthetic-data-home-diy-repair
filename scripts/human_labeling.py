"""
Human labeling for Home DIY Repair Q&A.
Interactive CLI: show one sample at a time, collect failure-mode labels + optional comment.
Saves to human_labels.json for comparison with LLM judge and for improving judge prompts.

Run from project root: python scripts/human_labeling.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is on path for shared imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diy_repair.config import DEFAULT_OUTPUT_DIR, FILENAMES, HUMAN_LABELS_FILE
from diy_repair.failure_labeling import FAILURE_MODE_NAMES, FAILURE_MODES_CLI

CODE_TO_NAME = {code: name for name, code, _ in FAILURE_MODES_CLI}


def _project_root() -> Path:
    """Project root (parent of scripts/)."""
    return _PROJECT_ROOT


def _output_dir() -> Path:
    """Directory for pipeline outputs (and human labels)."""
    return _project_root() / DEFAULT_OUTPUT_DIR


def load_samples():
    """Load samples to label. Prefer failure_labeled_data (same set as LLM), else structurally_valid_qa."""
    out = _output_dir()
    for name in [FILENAMES["failure_labeled_json"], FILENAMES["structurally_valid_qa"]]:
        path = out / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return [data] if isinstance(data, dict) else []
    return []


def load_existing_labels():
    """Load existing human labels (trace_id -> label dict)."""
    path = _output_dir() / HUMAN_LABELS_FILE
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["trace_id"]: e for e in entries}


def save_labels(labels_list):
    """Write full list of label entries to output/human_labels.json."""
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / HUMAN_LABELS_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")


def parse_codes(raw: str):
    """Parse comma-separated codes (e.g. 'ia,sv' or 'ia sv') into list of failure mode names. Empty => none."""
    if not raw or not raw.strip():
        return []
    names = []
    for part in raw.replace(",", " ").split():
        code = part.strip().lower()
        if code in CODE_TO_NAME:
            names.append(CODE_TO_NAME[code])
    return names


def prompt_for_labels(sample, existing):
    """Show one sample and prompt for failure codes + comment. Return label dict."""
    trace_id = sample.get("trace_id", "")
    print("\n" + "=" * 60)
    print("QUESTION:", sample.get("question", ""))
    print("-" * 60)
    print("ANSWER:", sample.get("answer", ""))
    print("-" * 60)
    print("EQUIPMENT:", sample.get("equipment_problem", ""))
    print("STEPS:", sample.get("steps", []))
    print("SAFETY:", sample.get("safety_info", "") or "")
    print("TIPS:", sample.get("tips", "") or "")
    print("=" * 60)
    # Show LLM judge result if this sample came from failure-labeled data
    llm_overall = sample.get("overall_failure")
    if llm_overall is not None:
        if llm_overall == 1:
            failed_modes = [name for name in FAILURE_MODE_NAMES if sample.get(name) == 1]
            print("LLM JUDGE: FAILED on:", ", ".join(failed_modes) if failed_modes else "(unknown)")
        else:
            print("LLM JUDGE: PASSED (no failure modes)")
    else:
        print("LLM JUDGE: (no prior label – from structurally_valid only)")
    print("-" * 60)
    print("Codes: " + ", ".join(f"{code}={name}" for name, code, _ in FAILURE_MODES_CLI))
    print("Enter comma-separated codes (or Enter for NONE, q to quit): ", end="")
    raw = input().strip()
    if raw.lower() == "q":
        raise KeyboardInterrupt
    names = parse_codes(raw)
    comment = None
    if names:
        print("Comment (why it failed, optional): ", end="")
        comment = input().strip() or None

    out = {
        "trace_id": trace_id,
        **{name: 1 if name in names else 0 for name in FAILURE_MODE_NAMES},
        "comment": comment or None,
    }
    out["overall_failure"] = 1 if any(out[k] for k in FAILURE_MODE_NAMES) else 0
    return out


def main():
    samples = load_samples()
    if not samples:
        print("No samples found. Run the pipeline first so output/ has structurally_valid_qa_pairs.json or failure_labeled_data.json")
        return

    existing = load_existing_labels()
    labeled_ids = set(existing.keys())
    to_label = [s for s in samples if s.get("trace_id") not in labeled_ids]
    if not to_label:
        print(f"All {len(samples)} samples already labeled in {HUMAN_LABELS_FILE}.")
        return

    print(f"Samples to label: {len(to_label)} (already labeled: {len(labeled_ids)})")
    print("Commands: Enter=codes, then optional comment. q=quit and save.")
    labels_list = list(existing.values())

    for i, sample in enumerate(to_label, 1):
        print(f"\n--- Sample {i}/{len(to_label)} ---")
        try:
            label = prompt_for_labels(sample, existing)
            labels_list.append(label)
            existing[label["trace_id"]] = label
            save_labels(labels_list)
        except (EOFError, KeyboardInterrupt):
            print("\nQuit. Progress saved.")
            break

    print(f"Done. Total labels in {HUMAN_LABELS_FILE}: {len(labels_list)}")


if __name__ == "__main__":
    main()
