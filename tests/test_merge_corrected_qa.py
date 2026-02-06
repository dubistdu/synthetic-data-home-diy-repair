from scripts.merge_corrected_qa import merge_corrected_qa_records


def test_merge_corrected_qa_records_replaces_failed_with_corrected():
    """merge_corrected_qa_records should swap in corrected Q&A for failed trace_ids."""

    valid_qa = [
        {"trace_id": "t1", "question": "Q1"},
        {"trace_id": "t2", "question": "Q2"},
    ]

    failure_labeled = [
        {"trace_id": "t1", "overall_failure": 0},
        {"trace_id": "t2", "overall_failure": 1},
    ]

    corrected_list = [
        {
            "trace_id": "t2",
            "qa_pair": {"question": "Q2-corrected"},
            "is_valid": True,
        }
    ]

    merged = merge_corrected_qa_records(valid_qa, failure_labeled, corrected_list)

    assert len(merged) == 2
    # t1 unchanged
    assert merged[0]["trace_id"] == "t1"
    assert merged[0]["question"] == "Q1"
    # t2 replaced
    assert merged[1]["trace_id"] == "t2"
    assert merged[1]["question"] == "Q2-corrected"

