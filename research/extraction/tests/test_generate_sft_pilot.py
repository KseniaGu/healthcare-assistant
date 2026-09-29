"""Fictitious fixtures for CA1 pilot rendering, boundaries and split checks."""

import json

import pytest

from research.extraction import generate_sft_pilot as pilot


SCHEMA = json.loads((pilot.CORRECTION / "schema.json").read_text())


def family(style=0):
    return dict(family="fixture", domain="chemistry", style=style, split="train")


def row(name="Тест", value="1,00", unit="г/π", reference="0–2"):
    return dict(name=name, value=value, unit=unit, reference_range=reference, state="current")


def make(rows, dates=None, style=0):
    dates = dates or ["Дата печати: 03.01.2024", "Дата выполнения: 02.01.2024"]
    text, spans = pilot.render(family(style), 0, rows, dates)
    target, evidence = pilot.target_from_visible(text, rows, spans, dates)
    pilot.check_pair(text, rows, spans, target, evidence, SCHEMA)
    return text, target, evidence


def test_deleted_visible_cells_are_not_restored():
    text, target, evidence = make([row(value="—"), row(name="—", value="7,30", reference="—")])
    assert target["results"] == [dict(name="Тест", value=None, unit="г/л", reference_range="0–2")]
    assert "7,30" in target["errors"][0]
    assert "reference cell(s) incomplete" in target["errors"][0]
    assert "Missing result value" in target["errors"][1]
    assert any(x["kind"] == "omission" for x in evidence)
    assert "7,30" in text


def test_repair_abstention_and_polarity():
    _, target, evidence = make([row(name="Глюкoза", value="Отрищательный", unit="10°/π")])
    assert target["results"][0] == dict(name="Глюкоза", value="Отрицательный", unit=None,
                                        reference_range="0–2")
    assert {x["field"] for x in evidence} == {"name", "value", "unit"}
    assert pilot.visible_value("value", "+") == "+"
    assert pilot.visible_value("value", "−") == "−"


def test_paired_cells_and_date_priority():
    rows = [row(name="Лимфоциты, %", value="43.00", unit="%"),
            row(name="Лимфоциты, #", value="2,40", unit="10^9/л")]
    dates = ["Дата печати: 03.01.2024", "Дата направления: 31.12.2023",
             "Дата выполнения: 02.01.2024", "Дата взятия образца: 01.01.2024"]
    _, target, _ = make(rows, dates, 2)
    assert target["date"] == "2024-01-01"
    assert [x["name"] for x in target["results"]] == ["Лимфоциты, %", "Лимфоциты, #"]
    assert [x["value"] for x in target["results"]] == ["43.00", "2,40"]


def test_missing_date_and_routine_exclusions():
    rows = [row(), dict(row(name="Прошлый результат", value="9"), state="history"),
            dict(row(name="Повтор", value="ожидается"), state="pending")]
    _, target, _ = make(rows, ["Дата печати: 03.01.2024"])
    assert target["date"] is None
    assert len(target["results"]) == 1
    assert target["errors"] == ["No report collection, performance or order date is visible."]


def test_split_duplicate_and_excluded_provenance():
    base = dict(id="one", split="train", family="f", source_group="g", development_sources=[],
                input="text", input_sha256=pilot.sha("text"))
    with pytest.raises(ValueError, match="split leakage"):
        pilot.ensure_split_and_provenance([base, dict(base, id="two", split="validation", input="other",
                                                        input_sha256=pilot.sha("other"))], {})
    with pytest.raises(ValueError, match="Duplicate input"):
        pilot.ensure_split_and_provenance([base, dict(base, id="two", family="other", source_group="other")], {})
    with pytest.raises(ValueError, match="Excluded"):
        pilot.ensure_split_and_provenance([dict(base, development_sources=["D0001"])],
                                          {"D0001": {"role": "development"}})


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2] if add_generation_prompt else [1, 2, 3, 9, 4]


def test_eos_and_post_eos_mask():
    ids, labels, prefix, supervised = pilot.tokenize(Tokenizer(), [{}, {}, {}])
    assert ids == [1, 2, 3, 9, 4]
    assert labels == [-100, -100, 3, 9, -100]
    assert (prefix, supervised) == (2, 2)


def test_total_token_budget_rejects_overflow():
    class LongTokenizer(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1] if add_generation_prompt else [1] * 8192 + [9]
    with pytest.raises(Exception, match="8192"):
        pilot.tokenize(LongTokenizer(), [{}, {}, {}])


def test_detection_limit_stays_positive():
    import random
    value = pilot.value_for("urine", "Белок мочи", "<0,15", random.Random(7), 11)
    assert value.startswith("<0,") and value not in ("<0,0", "<0,00")
