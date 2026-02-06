"""Correction phase: fix failed Q&A pairs using judge criteria."""

import json
import time
from datetime import datetime
from typing import List, Dict, Any

from .config import DEFAULT_OUTPUT_DIR, FILENAMES
from .failure_labeling import FailureModeDefinitions, FAILURE_MODE_NAMES
from .models import DIYRepairQA, validate_json_structure
from .openai_client import get_openai_client


FIX_INSTRUCTIONS = {
    "incomplete_answer": "Expand the answer and steps so they are comprehensive, step-by-step, and sufficient for someone to complete the repair successfully. Add any missing critical steps.",
    "safety_violations": "Add clear safety warnings, precautions (e.g. turn off power/water), and explicitly state when to call a professional. Ensure nothing dangerous is implied.",
    "unrealistic_tools": "Use only tools commonly available to homeowners or easily found at hardware stores. Replace any specialized or expensive tools with realistic alternatives.",
    "overcomplicated_solution": "Simplify the solution: fewer steps where possible, appropriate to a DIY skill level. Remove unnecessary complexity while keeping it effective.",
    "missing_context": "Add when to use this solution, any prerequisites (e.g. parts, time), and situational notes (e.g. when it might not apply, when to call a pro).",
    "poor_quality_tips": "Rewrite tips to be specific and actionable (e.g. exact techniques, order of operations, what to avoid). Do not repeat the main answer; add real extra value.",
}


class DIYRepairCorrector:
    """Correction Phase for Home DIY Repair Q&A Synthetic Data."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = get_openai_client()
        self.model = model

    def _create_correction_prompt(self, failed_qa_pair: Dict[str, Any]) -> str:
        """Build a correction prompt that uses the same success criteria as the judge."""
        modes_by_name = {m.name: m for m in FailureModeDefinitions.get_failure_modes()}
        failed_modes = [name for name in FAILURE_MODE_NAMES if failed_qa_pair.get(name) == 1]
        if not failed_modes:
            failed_modes = ["quality"]
        fix_section_lines = []
        for name in failed_modes:
            if name == "quality":
                fix_section_lines.append("- Improve overall quality so it would pass all 6 quality checks.")
                continue
            mode = modes_by_name.get(name)
            success = mode.success_criteria if mode else "Meet quality bar for this aspect."
            how = FIX_INSTRUCTIONS.get(name) or f"Revise so the content satisfies: {success}"
            fix_section_lines.append(f"- **{name}**: Judge expects: \"{success}\" → To fix: {how}")
        fix_section = "\n".join(fix_section_lines)
        prompt = f"""You are an expert home DIY repair technician. This Q&A pair failed quality checks. Your corrected version will be re-evaluated with the same criteria; you must fix it so it passes.

FAILURES TO FIX (do exactly what is needed so the judge scores 0 / success):
{fix_section}

ORIGINAL Q&A PAIR:
Question: {failed_qa_pair.get('question', '')}
Answer: {failed_qa_pair.get('answer', '')}
Equipment Problem: {failed_qa_pair.get('equipment_problem', '')}
Tools Required: {failed_qa_pair.get('tools_required', [])}
Steps: {failed_qa_pair.get('steps', [])}
Safety Info: {failed_qa_pair.get('safety_info', '')}
Tips: {failed_qa_pair.get('tips', '')}

TASK: Produce a CORRECTED Q&A that addresses every failure above. Keep the same topic and equipment problem. Each field must be substantive (no placeholders).

Return ONLY a valid JSON object with this exact structure:
{{
  "question": "A specific question about DIY repair",
  "answer": "Detailed step-by-step answer with technical details",
  "equipment_problem": "Specific equipment or problem being addressed",
  "tools_required": ["list", "of", "specific", "tools", "needed"],
  "steps": ["step 1", "step 2", "step 3", "etc"],
  "safety_info": "Important safety warnings and precautions",
  "tips": "Professional tips and best practices"
}}"""
        return prompt

    def correct_single_qa(self, failed_qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Correct a single failed Q&A pair. Returns a result dict with corrected_qa, is_valid, etc."""
        correction_prompt = self._create_correction_prompt(failed_qa_pair)
        system_msg = (
            "You correct failed DIY repair Q&A so they pass quality checks. "
            "Your output will be re-evaluated on the same 6 criteria; fix each reported failure so it would score success (0). "
            "Return only valid JSON with the required fields."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": correction_prompt},
            ],
            temperature=0.5,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        is_valid, qa_pair, errors = validate_json_structure(raw)
        return {
            "trace_id": failed_qa_pair.get("trace_id", ""),
            "qa_pair": qa_pair.model_dump() if qa_pair else None,
            "raw_response": raw,
            "is_valid": is_valid,
            "validation_errors": errors or [],
            "generation_timestamp": datetime.now().isoformat(),
        }

    def correct_batch(self, failed_qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correct a batch of failed Q&A pairs. Returns list of result dicts."""
        results = []
        for i, failed_qa_pair in enumerate(failed_qa_pairs):
            result = self.correct_single_qa(failed_qa_pair)
            results.append(result)
            if i < len(failed_qa_pairs) - 1:
                time.sleep(0.5)
        return results

    def save_results(self, results: List[Dict[str, Any]], filename: str = None) -> None:
        """Save correction results to JSON file."""
        if filename is None:
            filename = str(DEFAULT_OUTPUT_DIR / FILENAMES["corrected_qa"])
        serializable_results = []
        for result in results:
            result_dict = dict(result)
            if result_dict.get("qa_pair"):
                result_dict["qa_pair"] = result["qa_pair"]
            serializable_results.append(result_dict)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


def run_correction_phase(
    model: str = "gpt-3.5-turbo",
    input_file: str = None,
    output_file: str = None,
) -> List[Dict[str, Any]]:
    """Run the correction phase on failed Q&A pairs from failure_labeled_data."""
    if input_file is None:
        input_file = str(DEFAULT_OUTPUT_DIR / FILENAMES["failure_labeled_json"])
    if output_file is None:
        output_file = str(DEFAULT_OUTPUT_DIR / FILENAMES["corrected_qa"])
    corrector = DIYRepairCorrector(model=model)
    with open(input_file, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)
    failed_qa_pairs = [p for p in all_pairs if p.get("overall_failure") == 1]
    if not failed_qa_pairs:
        print("No failed Q&A pairs to correct.")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []
    print(f"Correcting {len(failed_qa_pairs)} failed Q&A pair(s)...")
    results = corrector.correct_batch(failed_qa_pairs)
    corrector.save_results(results, output_file)
    valid = sum(1 for r in results if r["is_valid"])
    print(f"Corrected: {len(results)} valid: {valid} invalid: {len(results) - valid}")
    return results


if __name__ == "__main__":
    run_correction_phase()
