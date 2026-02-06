"""
Generation Phase for Home DIY Repair Q&A Synthetic Data
Uses OpenAI client with multiple prompt templates to generate diverse Q&A pairs
"""

import json
import random
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

from .openai_client import get_openai_client
from .models import DIYRepairQA, GenerationResult, validate_json_structure

# Shared JSON schema for all prompts; single place to update structure.
_JSON_SCHEMA_BLOCK = """
Return ONLY a valid JSON object with this exact structure:
{
    "question": "A specific question about the repair",
    "answer": "Detailed step-by-step answer with technical details",
    "equipment_problem": "Specific equipment or problem description",
    "tools_required": ["list", "of", "specific", "tools", "needed"],
    "steps": ["step 1", "step 2", "step 3", "etc"],
    "safety_info": "Important safety warnings and precautions",
    "tips": "Professional tips and best practices"
}
"""


class DIYRepairGenerator:
    """
    Generator for Home DIY Repair Q&A pairs using multiple prompt templates
    """

    def __init__(self, model="gpt-3.5-turbo"):
        self.client = get_openai_client()
        self.model = model
        self.prompt_templates = self._create_prompt_templates()

    def _create_prompt_templates(self) -> List[Dict[str, str]]:
        """
        Create diverse prompt templates for different types of DIY repair scenarios.
        Shared JSON schema is defined once; each template adds domain-specific instructions.
        """
        # (name, system, task_focus, closing_instruction)
        domains = [
            (
                "appliance_repair",
                "You are an expert home appliance repair technician with 20+ years of experience.",
                "Focus on common household appliances like refrigerators, washing machines, dryers, dishwashers, or ovens.",
                "Make it realistic and practical for a homeowner.",
            ),
            (
                "plumbing_repair",
                "You are a professional plumber with extensive experience in residential plumbing repairs.",
                "Focus on common issues like leaks, clogs, fixture repairs, or pipe problems.",
                "Make it realistic and safe for a homeowner to attempt.",
            ),
            (
                "electrical_repair",
                "You are a licensed electrician specializing in safe home electrical repairs.",
                "Focus on SAFE homeowner-level electrical work like outlet replacement, switch repair, or light fixture installation.",
                "Emphasize safety and when to call a professional. Only include repairs safe for homeowners.",
            ),
            (
                "hvac_maintenance",
                "You are an HVAC technician specializing in homeowner maintenance and basic repairs.",
                "Focus on filter changes, thermostat issues, vent cleaning, or basic troubleshooting.",
                "Focus on maintenance and basic repairs homeowners can safely perform.",
            ),
            (
                "general_home_repair",
                "You are a skilled handyperson with expertise in general home repairs and maintenance.",
                "Focus on common issues like drywall repair, door/window problems, flooring issues, or basic carpentry.",
                "Make it practical for a DIY homeowner with basic skills.",
            ),
        ]
        templates = []
        for name, system, task_focus, closing in domains:
            user = f"""Generate a realistic {name.replace('_', ' ')} Q&A pair. {task_focus}
{_JSON_SCHEMA_BLOCK}
{closing}"""
            templates.append({"name": name, "system": system, "user": user})
        return templates

    def generate_single_qa(self, template: Dict[str, str]) -> GenerationResult:
        """
        Generate a single Q&A pair using the specified template

        Args:
            template: Prompt template dictionary

        Returns:
            GenerationResult with validation status and data
        """
        trace_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": template["system"]},
                    {"role": "user", "content": template["user"]}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            raw_response = response.choices[0].message.content.strip()

            # Validate the response
            is_valid, qa_pair, errors = validate_json_structure(raw_response)

            return GenerationResult(
                trace_id=trace_id,
                qa_pair=qa_pair,
                raw_response=raw_response,
                is_valid=is_valid,
                validation_errors=errors,
                generation_timestamp=timestamp
            )

        except Exception as e:
            return GenerationResult(
                trace_id=trace_id,
                qa_pair=None,
                raw_response="",
                is_valid=False,
                validation_errors=[f"Generation error: {str(e)}"],
                generation_timestamp=timestamp
            )

    def generate_batch(self, num_samples: int = 20) -> List[GenerationResult]:
        """
        Generate a batch of Q&A pairs using diverse templates

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of GenerationResult objects
        """
        results = []

        for i in range(num_samples):
            template = random.choice(self.prompt_templates)
            print(f"Generating sample {i+1}/{num_samples} using template: {template['name']}")

            result = self.generate_single_qa(template)
            results.append(result)

            time.sleep(0.5)

        return results

    def save_results(
        self, results: List[GenerationResult], filename: str = "generation_results.json"
    ) -> None:
        """Save generation results to JSON file."""
        serializable_results = []
        for result in results:
            result_dict = result.model_dump()
            if result_dict['qa_pair']:
                result_dict['qa_pair'] = result.qa_pair.model_dump()
            serializable_results.append(result_dict)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {filename}")


def run_generation_phase(num_samples: int = 20, model: str = "gpt-3.5-turbo", seed: int = None) -> List[GenerationResult]:
    """Main function to run the generation phase. Use seed for reproducible template order."""
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed {seed} for reproducible template selection.")
    print(f"Starting generation phase for {num_samples} samples using {model}...")

    generator = DIYRepairGenerator(model=model)
    results = generator.generate_batch(num_samples)

    generator.save_results(results)

    valid_count = sum(1 for r in results if r.is_valid)
    print(f"\nGeneration Phase Complete:")
    print(f"Total generated: {len(results)}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {len(results) - valid_count}")
    print(f"Success rate: {valid_count/len(results)*100:.1f}%")

    return results


if __name__ == "__main__":
    run_generation_phase()
