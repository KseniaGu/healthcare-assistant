"""Exact offline scoring shared by extraction experiments; no string normalization."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from research.extraction.datasets.packet import CA1Result, CA1Target

ResultField = Literal["name", "value", "unit", "reference_range"]
FIELDS: tuple[ResultField, ...] = ("name", "value", "unit", "reference_range")


class Score(TypedDict):
    schema_valid: bool
    date_exact: bool
    expected_rows: int
    actual_rows: int
    expected_fields: int
    field_matches: int
    ordered_record_matches: int
    ordered_results_exact: bool
    exact_record_matches: int
    missing_exact_records: int
    extra_exact_records: int
    extra_duplicate_records: int
    record_precision: float
    record_recall: float
    order_only_difference: bool
    warning_status: Literal["failed", "pass", "pending_review"]
    differences: list[dict[str, Any]]
    strict_pass: bool


def compare(target: CA1Target, prediction: CA1Target | None) -> Score:
    """Scores exact rows independently of order; strict pass requires empty warnings.

    Returns:
        Exact-record counts use a multiset, while ordered metrics retain row
        position. Nonempty warning lists require later source review rather
        than an automatic warning pass.
    """
    expected = target["results"]
    actual = prediction["results"] if prediction is not None else []

    def encode(row: CA1Result) -> tuple[str | None, ...]:
        return row["name"], row["value"], row["unit"], row["reference_range"]

    wanted = Counter(map(encode, expected))
    got = Counter(map(encode, actual))
    matches = sum((wanted & got).values())
    differences: list[dict[str, Any]] = []
    if prediction is None or prediction["date"] != target["date"]:
        differences.append(
            {"field": "date", "expected": target["date"], "actual": prediction["date"] if prediction else None}
        )
    field_matches, ordered_matches = 0, 0
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else None
        right = actual[index] if index < len(actual) else None
        if left is not None and right is not None:
            field_matches += sum(left[field] == right[field] for field in FIELDS)
            ordered_matches += int(left == right)
        if left != right:
            differences.append({"row": index + 1, "expected": left, "actual": right})
    warning_status: Literal["failed", "pass", "pending_review"]
    if prediction is None:
        warning_status = "failed"
    elif prediction["errors"] == target["errors"] == []:
        warning_status = "pass"
    else:
        warning_status = "pending_review"
    if prediction is not None and prediction["errors"] != target["errors"]:
        differences.append({"field": "errors", "expected": target["errors"], "actual": prediction["errors"]})

    return {
        "schema_valid": prediction is not None,
        "date_exact": prediction is not None and prediction["date"] == target["date"],
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "expected_fields": 4 * len(expected),
        "field_matches": field_matches,
        "ordered_record_matches": ordered_matches,
        "ordered_results_exact": prediction is not None and actual == expected,
        "exact_record_matches": matches,
        "missing_exact_records": sum((wanted - got).values()),
        "extra_exact_records": sum((got - wanted).values()),
        "extra_duplicate_records": sum(max(0, count - max(1, wanted[row])) for row, count in got.items()),
        "record_precision": matches / len(actual) if actual else 0,
        "record_recall": matches / len(expected) if expected else int(prediction is not None),
        "order_only_difference": prediction is not None and wanted == got and actual != expected,
        "warning_status": warning_status,
        "differences": differences,
        "strict_pass": prediction is not None
        and prediction["date"] == target["date"]
        and actual == expected
        and warning_status == "pass",
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    keys = (
        "schema_valid",
        "date_exact",
        "expected_rows",
        "actual_rows",
        "expected_fields",
        "field_matches",
        "ordered_record_matches",
        "exact_record_matches",
        "missing_exact_records",
        "extra_exact_records",
        "strict_pass",
    )
    result: dict[str, int | float] = {key: sum(row[key] for row in rows) for key in keys}
    result["reports"] = len(rows)
    result["record_precision"] = result["exact_record_matches"] / result["actual_rows"] if result["actual_rows"] else 0
    result["record_recall"] = result["exact_record_matches"] / result["expected_rows"] if result["expected_rows"] else 0
    return result
