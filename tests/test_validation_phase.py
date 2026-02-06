from diy_repair.models import DIYRepairQA, GenerationResult
from diy_repair.validation_phase import run_validation_phase


def _make_valid_qa() -> DIYRepairQA:
    return DIYRepairQA(
        question="How can I fix a leaky kitchen faucet?",
        answer="Turn off the water, disassemble the handle, replace the worn parts, and reassemble.",
        equipment_problem="Leaky kitchen faucet",
        tools_required=["adjustable wrench", "screwdriver"],
        steps=["Turn off water", "Disassemble handle", "Replace parts", "Reassemble and test"],
        safety_info="Turn off the water supply before starting any work.",
        tips="Take a photo before disassembly so you can reassemble correctly.",
    )


def test_run_validation_phase_filters_invalid_samples():
    """Validation phase should keep only structurally valid GenerationResult entries."""

    valid_qa = _make_valid_qa()

    valid_result = GenerationResult(
        trace_id="valid-1",
        qa_pair=valid_qa,
        raw_response="{}",
        is_valid=True,
        validation_errors=[],
        generation_timestamp="2026-01-01T00:00:00",
    )

    invalid_result = GenerationResult(
        trace_id="invalid-1",
        qa_pair=None,
        raw_response="{}",
        is_valid=False,
        validation_errors=["some error"],
        generation_timestamp="2026-01-01T00:00:00",
    )

    valid_results, summary = run_validation_phase([valid_result, invalid_result])

    assert len(valid_results) == 1
    assert valid_results[0].trace_id == "valid-1"
    assert summary.total_generated == 2
    assert summary.valid_samples == 1
    assert summary.invalid_samples == 1
    assert summary.validation_rate == 50.0

