"""
Validation Phase for Home DIY Repair Q&A Synthetic Data
Uses Pydantic to validate that outputs are structurally correct and filters invalid entries
"""

import json
from collections import Counter
from typing import List, Tuple

from .models import DIYRepairQA, GenerationResult, ValidationSummary, validate_json_structure


class DIYRepairValidator:
    """
    Validator for Home DIY Repair Q&A pairs - structural validation only
    Filters invalid entries before moving to error analysis
    """

    def __init__(self):
        pass

    def validate_structure(self, result: GenerationResult) -> Tuple[bool, List[str]]:
        """Validate only the structural correctness of a generation result."""
        if not result.is_valid:
            return False, result.validation_errors

        if result.qa_pair is None:
            return False, ["No valid Q&A pair generated"]

        errors = []
        qa_pair = result.qa_pair

        if not qa_pair.question.strip():
            errors.append("Question is empty or whitespace only")
        if not qa_pair.answer.strip():
            errors.append("Answer is empty or whitespace only")
        if not qa_pair.equipment_problem.strip():
            errors.append("Equipment problem is empty or whitespace only")
        if not qa_pair.safety_info.strip():
            errors.append("Safety info is empty or whitespace only")
        if not qa_pair.tips.strip():
            errors.append("Tips is empty or whitespace only")
        if not qa_pair.tools_required:
            errors.append("Tools required list is empty")
        if not qa_pair.steps:
            errors.append("Steps list is empty")
        if any(not tool.strip() for tool in qa_pair.tools_required):
            errors.append("Tools required contains empty items")
        if any(not step.strip() for step in qa_pair.steps):
            errors.append("Steps contains empty items")

        return len(errors) == 0, errors

    def validate_batch(self, results: List[GenerationResult]) -> Tuple[List[GenerationResult], ValidationSummary]:
        """Validate a batch of generation results - structural validation only."""
        valid_results = []
        all_errors = []

        for result in results:
            is_structurally_valid, structural_errors = self.validate_structure(result)

            if is_structurally_valid:
                valid_results.append(result)
            else:
                result.is_valid = False
                result.validation_errors.extend(structural_errors)
                all_errors.extend(structural_errors)

        error_counter = Counter(all_errors)
        common_errors = [error for error, count in error_counter.most_common(5)]

        summary = ValidationSummary(
            total_generated=len(results),
            valid_samples=len(valid_results),
            invalid_samples=len(results) - len(valid_results),
            validation_rate=len(valid_results) / len(results) * 100 if results else 0,
            common_errors=common_errors
        )

        return valid_results, summary


def run_validation_phase(results: List[GenerationResult]) -> Tuple[List[GenerationResult], ValidationSummary]:
    """Main function to run the validation phase - structural validation only."""
    print("Starting validation phase (structural validation only)...")

    validator = DIYRepairValidator()
    valid_results, summary = validator.validate_batch(results)

    print(f"\nValidation Phase Complete:")
    print(f"Total samples: {summary.total_generated}")
    print(f"Structurally valid samples: {summary.valid_samples}")
    print(f"Structurally invalid samples: {summary.invalid_samples}")
    print(f"Structural validation rate: {summary.validation_rate:.1f}%")

    if summary.common_errors:
        print(f"\nMost common structural errors:")
        for i, error in enumerate(summary.common_errors, 1):
            print(f"{i}. {error}")

    return valid_results, summary


def save_valid_data(valid_results: List[GenerationResult], filename: str = "structurally_valid_qa_pairs.json"):
    """Save only the structurally valid Q&A pairs to a clean JSON file."""
    clean_data = []
    for result in valid_results:
        if result.qa_pair:
            qa_dict = result.qa_pair.model_dump()
            qa_dict['trace_id'] = result.trace_id
            clean_data.append(qa_dict)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

    print(f"Structurally valid Q&A pairs saved to {filename}")


if __name__ == "__main__":
    from .config import DEFAULT_OUTPUT_DIR, FILENAMES
    path = DEFAULT_OUTPUT_DIR / FILENAMES["generation_results"]
    with open(path, 'r') as f:
        data = json.load(f)
    results = []
    for item in data:
        if item.get('qa_pair'):
            item['qa_pair'] = DIYRepairQA(**item['qa_pair'])
        results.append(GenerationResult(**item))
    valid_results, summary = run_validation_phase(results)
    save_valid_data(valid_results, str(DEFAULT_OUTPUT_DIR / FILENAMES["structurally_valid_qa"]))
