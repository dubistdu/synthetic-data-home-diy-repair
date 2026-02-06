"""
Phase 3: Failure Labeling for Home DIY Repair Q&A Synthetic Data
Uses LLM-as-Judge to evaluate Q&A pairs against 6 failure modes
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, FILENAMES
from .models import DIYRepairQA, GenerationResult
from .openai_client import get_openai_client


@dataclass
class FailureMode:
    """Definition of a failure mode with evaluation criteria"""
    name: str
    description: str
    success_criteria: str
    failure_criteria: str
    evaluation_prompt: str


class FailureModeDefinitions:
    """Defines the 6 failure modes for DIY repair Q&A evaluation."""

    @staticmethod
    def get_failure_modes() -> List[FailureMode]:
        """Returns the 6 failure modes with clear criteria for LLM evaluation."""
        return [
            FailureMode(
                name="incomplete_answer",
                description="Answer lacks sufficient detail or steps to complete the repair",
                success_criteria="Answer provides comprehensive, step-by-step instructions that would allow someone to complete the repair successfully",
                failure_criteria="Answer is vague, missing critical steps, or lacks sufficient detail for successful completion",
                evaluation_prompt="""
Evaluate if this DIY repair answer is COMPLETE and SUFFICIENT:

SUCCESS (0): Answer and steps together give clear, step-by-step instructions with enough detail for someone to complete the repair (key steps present, no critical gaps).
FAILURE (1): Answer is vague, missing critical steps, or lacks sufficient detail for successful completion. If steps or answer skip important actions or leave the reader unsure how to proceed, score 1.

Question: {question}
Answer: {answer}
Steps: {steps}

Rate as 0 (complete) or 1 (incomplete). Respond with only the number.
""",
            ),
            FailureMode(
                name="safety_violations",
                description="Missing critical safety warnings or dangerous advice",
                success_criteria="Includes appropriate safety warnings, precautions, and mentions when to call professionals",
                failure_criteria="Missing safety warnings, provides dangerous advice, or fails to mention professional help when needed",
                evaluation_prompt="""
Evaluate if this DIY repair guidance has ADEQUATE SAFETY information:

SUCCESS (0): Includes appropriate safety warnings and precautions for the task (e.g. turn off power/water, wear protection, ventilation). For risky tasks, mentions when to call a professional.
FAILURE (1): Missing important safety warnings for the task, implies dangerous practice, or omits when a pro should be called for clearly risky work.

Question: {question}
Safety Info: {safety_info}
Answer: {answer}

Rate as 0 (safe) or 1 (safety violation). Respond with only the number.
""",
            ),
            FailureMode(
                name="unrealistic_tools",
                description="Requires tools that are unrealistic for typical homeowners",
                success_criteria="Tools are commonly available to homeowners or easily obtainable from hardware stores",
                failure_criteria="Requires specialized professional tools, overly expensive equipment, or unrealistic tool combinations",
                evaluation_prompt="""
Evaluate if the required tools are REALISTIC for typical homeowners:

SUCCESS (0): Tools are commonly available at hardware stores or typical home toolkits (screwdrivers, wrenches, pliers, basic hand tools, common supplies). Standard DIY tools are OK.
FAILURE (1): Requires specialized professional-only tools, very expensive equipment, or an unrealistic combination. Do NOT fail for normal hardware-store or common DIY tools.

Question: {question}
Tools Required: {tools_required}
Equipment Problem: {equipment_problem}

Rate as 0 (realistic tools) or 1 (unrealistic tools). Respond with only the number.
""",
            ),
            FailureMode(
                name="overcomplicated_solution",
                description="Solution is unnecessarily complex for the problem described",
                success_criteria="Solution is appropriately scaled to the problem complexity and homeowner skill level",
                failure_criteria="Solution is overly complex, requires excessive steps, or is disproportionate to the problem",
                evaluation_prompt="""
Evaluate if this DIY repair solution is APPROPRIATELY COMPLEX:

SUCCESS (0): Solution has a reasonable number of steps for the problem (e.g. 4–10 steps for a typical repair is fine). Step-by-step instructions that match the task are NOT overcomplicated. Only fail if truly excessive.
FAILURE (1): Solution is clearly overkill: far too many steps for a simple fix, or requires professional-level complexity for a basic DIY task. Normal detailed steps do NOT count as overcomplicated.

Question: {question}
Equipment Problem: {equipment_problem}
Steps: {steps}
Tools Required: {tools_required}

Rate as 0 (appropriate complexity) or 1 (overcomplicated). Respond with only the number.
""",
            ),
            FailureMode(
                name="missing_context",
                description="Lacks important context about when, why, or how to apply the solution",
                success_criteria="Provides context about when to use this solution, prerequisites, and situational considerations",
                failure_criteria="Missing context about applicability, prerequisites, or situational factors",
                evaluation_prompt="""
Evaluate if this DIY repair guidance provides ADEQUATE CONTEXT:

SUCCESS (0): Gives some sense of when to use this approach, what’s needed (e.g. time, parts), or when to call a pro. Brief context is enough.
FAILURE (1): No context at all: reader cannot tell when this solution applies, what to have ready, or when it’s beyond DIY. Clearly missing prerequisites or situational guidance.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tips: {tips}

Rate as 0 (adequate context) or 1 (missing context). Respond with only the number.
""",
            ),
            FailureMode(
                name="poor_quality_tips",
                description="Tips are generic, unhelpful, or don't add value beyond the main answer",
                success_criteria="Tips provide specific, actionable advice that enhances the repair process",
                failure_criteria="Tips are generic, obvious, unhelpful, or simply repeat information from the answer",
                evaluation_prompt="""
Evaluate if the provided tips are HIGH QUALITY and VALUABLE:

SUCCESS (0): Tips add some useful value (specific advice, order of operations, or what to avoid). They need not be perfect; slightly generic but helpful is OK.
FAILURE (1): Tips are purely generic, only repeat the answer, or add no real value. Only fail when tips are clearly unhelpful or redundant.

Question: {question}
Answer: {answer}
Tips: {tips}

Rate as 0 (quality tips) or 1 (poor quality tips). Respond with only the number.
""",
            ),
        ]


_FAILURE_MODES_LIST = FailureModeDefinitions.get_failure_modes()
FAILURE_MODE_NAMES = [m.name for m in _FAILURE_MODES_LIST]
FAILURE_MODE_SHORT_CODES = {
    "incomplete_answer": "ia",
    "safety_violations": "sv",
    "unrealistic_tools": "ut",
    "overcomplicated_solution": "os",
    "missing_context": "mc",
    "poor_quality_tips": "pt",
}
FAILURE_MODES_CLI = [
    (name, FAILURE_MODE_SHORT_CODES[name], name.replace("_", " ").title())
    for name in FAILURE_MODE_NAMES
]


class LLMJudge:
    """LLM-as-Judge system for evaluating Q&A pairs against failure modes."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = get_openai_client()
        self.model = model
        self.failure_modes = FailureModeDefinitions.get_failure_modes()

    def evaluate_single_failure_mode(self, qa_pair: DIYRepairQA, failure_mode: FailureMode) -> Tuple[int, str]:
        """Evaluate a Q&A pair against a single failure mode. Returns (failure_score, raw_response)."""
        try:
            prompt = failure_mode.evaluation_prompt.format(
                question=qa_pair.question,
                answer=qa_pair.answer,
                equipment_problem=qa_pair.equipment_problem,
                tools_required=qa_pair.tools_required,
                steps=qa_pair.steps,
                safety_info=qa_pair.safety_info,
                tips=qa_pair.tips
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert DIY repair evaluator. Respond with only 0 or 1."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            raw_response = response.choices[0].message.content.strip()
            try:
                failure_score = int(raw_response)
                if failure_score not in [0, 1]:
                    failure_score = 1
            except ValueError:
                failure_score = 1
            return failure_score, raw_response
        except Exception as e:
            print(f"Error evaluating {failure_mode.name}: {str(e)}")
            return 1, f"Error: {str(e)}"

    def evaluate_qa_pair(self, qa_pair: DIYRepairQA, trace_id: str) -> Dict:
        """Evaluate a Q&A pair against all failure modes."""
        results = {
            'trace_id': trace_id,
            'question': qa_pair.question,
            'answer': qa_pair.answer,
            'equipment_problem': qa_pair.equipment_problem,
            'tools_required': qa_pair.tools_required,
            'steps': qa_pair.steps,
            'safety_info': qa_pair.safety_info,
            'tips': qa_pair.tips
        }
        failure_count = 0
        for failure_mode in self.failure_modes:
            failure_score, raw_response = self.evaluate_single_failure_mode(qa_pair, failure_mode)
            results[failure_mode.name] = failure_score
            results[f"{failure_mode.name}_response"] = raw_response
            failure_count += failure_score
        results['overall_failure'] = 1 if failure_count > 0 else 0
        results['failure_count'] = failure_count
        return results


def load_structurally_valid_data(filename: str = None) -> List[Dict]:
    """Load the structurally valid Q&A pairs from validation phase."""
    if filename is None:
        filename = str(DEFAULT_OUTPUT_DIR / FILENAMES["structurally_valid_qa"])
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run the validation phase first.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {str(e)}")
        return []


def create_failure_labeled_dataframe(qa_data: List[Dict], model: str = "gpt-3.5-turbo") -> pd.DataFrame:
    """Create a Pandas DataFrame with failure labels for all Q&A pairs."""
    print(f"Starting failure labeling for {len(qa_data)} Q&A pairs using {model}...")
    judge = LLMJudge(model=model)
    labeled_data = []
    for i, qa_dict in enumerate(qa_data, 1):
        print(f"Evaluating sample {i}/{len(qa_data)}...")
        qa_pair = DIYRepairQA(**{k: v for k, v in qa_dict.items() if k != 'trace_id'})
        trace_id = qa_dict.get('trace_id', str(uuid.uuid4()))
        results = judge.evaluate_qa_pair(qa_pair, trace_id)
        labeled_data.append(results)
        time.sleep(0.5)
    df = pd.DataFrame(labeled_data)
    print(f"\nFailure labeling complete!")
    print(f"Total samples: {len(df)}")
    print(f"Overall failures: {df['overall_failure'].sum()}")
    print(f"Overall success rate: {(1 - df['overall_failure'].mean()) * 100:.1f}%")
    return df


if __name__ == "__main__":
    qa_data = load_structurally_valid_data()
    if qa_data:
        df = create_failure_labeled_dataframe(qa_data)
        out = DEFAULT_OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / FILENAMES["failure_labeled_csv"], index=False)
        df.to_json(out / FILENAMES["failure_labeled_json"], orient="records", indent=2)
        print(f"\nResults saved to {out / FILENAMES['failure_labeled_csv']}, {out / FILENAMES['failure_labeled_json']}")
    else:
        print("No data to process. Please run the generation and validation phases first.")
