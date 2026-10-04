"""Visible-cell extraction and provenance checks with invented report text."""

import pytest

from research.extraction.datasets import packet
from research.extraction.datasets import synthetic_reports as reports
from research.extraction.tests.test_ca1_packet import FakeTokenizer


def test_visible_value_keeps_numbers_and_qualitative_polarity():
    assert reports.visible_value("value", "4,2 *") == "4,2"
    assert reports.visible_value("value", "Отрищательный") == "Отрицательный"
    assert reports.visible_value("unit", "10°/π") is None
    assert reports.visible_value("value", "—") is None
    assert reports.select_date(["Дата печати: 03.01.2023", "Дата выполнения: 02.01.2023",
                                "Дата взятия образца: 01.01.2023"]) == "2023-01-01"


def test_visible_target_records_missing_cells_and_evidence():
    text = "Глюкоза | 4,2 | — | —"
    row = {"name": "Глюкоза", "value": "4,2", "unit": "—", "reference_range": "—", "state": "current"}
    spans = {}
    cursor = 0
    for field in ("name", "value", "unit", "reference_range"):
        value = row[field]
        start = text.find(value, cursor)
        spans[field] = {"start": start, "end": start + len(value), "text": value}
        cursor = start + len(value)
    target, evidence = reports.target_from_visible(text, [row], [spans], [])
    assert target == {"date": None, "results": [{"name": "Глюкоза", "value": "4,2",
                       "unit": None, "reference_range": None}],
                      "errors": ["No report collection, performance or order date is visible."]}
    assert {item["field"] for item in evidence} == {"unit", "reference_range"}
    schema = {"type": "object", "required": ["date", "results", "errors"]}
    reports.check_pair(text, [row], [spans], target, evidence, schema)


def test_family_and_reserved_source_rejection():
    base = {"id": "F1", "family": "family", "source_group": "group", "split": "train",
            "development_sources": ["A1"], "input": "Invented 1", "input_sha256": packet.sha("Invented 1")}
    second = dict(base, id="F2", split="validation", input="Invented 2",
                  input_sha256=packet.sha("Invented 2"))
    sources = {"A1": {"role": "development"}, "B1": {"role": "reserved"}}
    with pytest.raises(ValueError, match="leakage"):
        reports.ensure_split_and_provenance([base, second], sources)
    with pytest.raises(ValueError, match="reserved"):
        reports.ensure_split_and_provenance([dict(base, development_sources=["B1"])], sources)


def test_assistant_labels_stop_at_eos_and_mask_newline():
    messages = [{"role": "system", "content": "Fiction"}, {"role": "user", "content": "Invented result"},
                {"role": "assistant", "content": "{}"}]
    ids, labels, prefix, supervised = packet.tokenize(FakeTokenizer(), messages)
    assert labels[:prefix] == [-100] * prefix
    assert labels[prefix:prefix + supervised] == ids[prefix:prefix + supervised]
    assert labels[-1] == -100 and ids[-2] == FakeTokenizer.eos_token_id
    assert packet.validate_labels({"id": "F1", "input_ids": ids, "labels": labels},
                                  FakeTokenizer.eos_token_id, [10]) == prefix
