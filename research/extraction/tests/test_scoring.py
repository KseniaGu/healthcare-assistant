"""Exact scoring remains order-, duplicate-, and warning-sensitive."""

from research.extraction.evaluation.scoring import compare, summarize


def test_exact_scoring_keeps_duplicates_order_and_warning_contract():
    a = {"name": "A", "value": "1", "unit": None, "reference_range": None}
    b = {"name": "B", "value": "2", "unit": None, "reference_range": None}
    target = {"date": None, "results": [a, a, b], "errors": ["A warning"]}
    prediction = {"date": None, "results": [b, a, a], "errors": ["A warning"]}
    score = compare(target, prediction)
    assert score["exact_record_matches"] == 3
    assert score["order_only_difference"] and not score["ordered_results_exact"]
    assert not score["strict_pass"] and score["warning_status"] == "pending_review"
    assert summarize([score])["exact_record_matches"] == 3
