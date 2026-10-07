"""Fictitious checks for the revised CA1 document layouts."""

import json
import random

import pytest

from research.extraction.datasets import packet
from research.extraction.datasets.synthetic import reports

SCHEMA = {"type": "object", "required": ["date", "results", "errors"]}


def family(domain="chemistry", style=0):
    return dict(family="fixture", domain=domain, style=style, split="train")


def row(name="Глюкоза", value="4,70", unit="ммоль/л", reference_range="3,1–6,2", state="current"):
    return dict(name=name, value=value, unit=unit, reference_range=reference_range, state=state)


def test_catalog_uses_only_eligible_layout_evidence():
    groups = reports.catalog()
    assert len(groups) == 32
    assert all(set(group["development_sources"]) <= reports.ELIGIBLE for group in groups)
    assert all(not group["development_sources"] for group in groups if group["split"] == "validation")
    assert len({group["family"] for group in groups}) == 32


@pytest.mark.parametrize("style", [-1, 14, 20, 2.5, "6", True])
def test_invalid_report_style_is_rejected(style):
    group = family(style=style)
    with pytest.raises(ValueError, match="report style"):
        reports.panel_for(group)
    with pytest.raises(ValueError, match="report style"):
        reports.render_report_header(group, 0, [])
    with pytest.raises(ValueError, match="report style"):
        reports.row_line(row(), style, 1, 0)
    with pytest.raises(ValueError, match="report style"):
        reports.footer(group, 0)


def test_qualitative_scenarios_use_related_assays_and_no_lettered_markers():
    for style in range(14):
        panel = reports.panel_for(family("qualitative", style))
        assert len(panel) >= 12
        assert len({name for name, _, _ in panel}) == len(panel)
        assert not any("маркер A" in name or "маркер B" in name for name, _, _ in panel)
    urogenital = [name for name, _, _ in reports.panel_for(family("qualitative", 2))]
    assert any("Chlamydia" in name for name in urogenital)
    assert not any("HBsAg" in name for name in urogenital)
    respiratory = [name for name, _, _ in reports.panel_for(family("qualitative", 0))]
    assert not any("ротавирус" in name.lower() for name in respiratory)
    assert len(respiratory) == len(set(respiratory))
    validation_respiratory = [name for name, _, _ in reports.panel_for(family("qualitative", 12))]
    validation_hepatitis = [name for name, _, _ in reports.panel_for(family("qualitative", 13))]
    assert set(validation_respiratory) != set(respiratory)
    assert set(validation_hepatitis) != set(name for name, _, _ in reports.panel_for(family("qualitative", 1)))


def test_visible_cells_survive_wrapping_and_missing_values():
    rows = [
        row(name="Длинное название лабораторного показателя глюкозы"),
        row(name="Креатинин", value="—", unit="мкмоль/л", reference_range="45–101"),
        row(name="—", value="7,20"),
    ]
    dates = ["Дата печати: 04.01.2024", "Дата направления: 01.01.2024"]
    text, spans = reports.render(family("chemistry", 0), 0, rows, dates)
    target, evidence = reports.target_from_visible(rows, spans, dates)
    reports.check_pair(text, rows, target, evidence, SCHEMA)
    assert target["date"] == "2024-01-01"
    assert len(target["results"]) == 2
    assert target["results"][1]["value"] is None
    assert "7,20" in target["errors"][0]
    assert "Страница 2 / продолжение результатов" in text
    assert "стр. 2 из 2" in text
    assert "стр. 1 из 1" not in text


def test_paired_columns_keep_both_observations():
    rows = [
        row(name="Нейтрофилы, %", value="45,0", unit="%", reference_range="40–70"),
        row(name="Нейтрофилы, абс.", value="2,51", unit="10^9/л", reference_range="1,5–7,5"),
    ]
    dates = ["Дата печати: 04.01.2024", "Дата взятия образца: 02.01.2024"]
    text, spans = reports.render(family("hematology", 7), 1, rows, dates)
    target, evidence = reports.target_from_visible(rows, spans, dates)
    reports.check_pair(text, rows, target, evidence, SCHEMA)
    assert "       |       " in text
    assert [result["value"] for result in target["results"]] == ["45,0", "2,51"]


def test_dense_rows_are_measurements_not_padding():
    group = family("urine", 4)
    rows, _ = reports.rows_for(group, 1, random.Random(8), "ordinary", True)
    assert 22 <= len(rows) <= 30
    assert len({item["name"] for item in rows}) == len(rows)
    assert reports.value_for("urine", "Плотность мочи", "1,005–1,030", random.Random(8), 1).startswith("1,")
    assert reports.value_for("urine", "Цилиндры зернистые", "0–1", random.Random(8), 1).isdigit()


def test_distribution_profile_counts_layout_artifacts():
    result = reports.profile(["IMAGE\n<PERSON>\nwww.example.org", "Второй бланк\nИсследование"])
    assert result["count"] == 2
    assert (result["with_image"], result["with_person"], result["with_website"]) == (1, 1, 1)
    assert result["unique_openings"] == 2


def test_seeded_cases_keep_exact_rendered_targets_and_evidence():
    families = reports.catalog()
    rng = random.Random(reports.SEED)
    output = []
    for family, variant, bucket in (
        (families[0], 0, "ordinary"),
        (families[7], 1, "layout"),
        (families[-1], 2, "difficult"),
    ):
        rows, changes = reports.rows_for(family, variant, rng, bucket, False)
        dates, tag = reports.dates_for(family, variant, bucket)
        text, spans = reports.render(family, variant, rows, dates)
        target, evidence = reports.target_from_visible(rows, spans, dates)
        output.append([family["family"], text, target, changes, tag, evidence])
    assert packet.sha(json.dumps(output, ensure_ascii=False, sort_keys=True)) == (
        "500d7355dbd830e08f89acc8169f0cea1423ba3dfe6a00ccbd97533c72857b94"
    )
