"""Visible target derivation, evidence checks, and synthetic report audits."""

import difflib
import json
import re
from collections import Counter
from datetime import date
from statistics import median
from typing import Any

from research.extraction.datasets.packet import CA1Result, CA1Target, sha, validate_target

# Expose generation and case contracts through the report module used by the builder.
# isort: off
from research.extraction.datasets.synthetic.layouts import (
    ASSIGNMENT_SEED as ASSIGNMENT_SEED,
    ASSAYS as ASSAYS,
    ASSAY_ADDITIONS as ASSAY_ADDITIONS,
    BUCKET_ORDER as BUCKET_ORDER,
    BUCKETS as BUCKETS,
    DOMAINS as DOMAINS,
    ELIGIBLE as ELIGIBLE,
    EXTRA as EXTRA,
    LABELS as LABELS,
    MORE as MORE,
    PANELS as PANELS,
    REPORT_STYLE_COUNT as REPORT_STYLE_COUNT,
    SEED as SEED,
    SOURCE_HINTS as SOURCE_HINTS,
    SOURCE_LAYOUTS as SOURCE_LAYOUTS,
    VALIDATION_ASSAYS as VALIDATION_ASSAYS,
    VALIDATION_LAYOUT_DETAILS as VALIDATION_LAYOUT_DETAILS,
    VOCAB_SOURCES as VOCAB_SOURCES,
    assignments as assignments,
    base_value_for as base_value_for,
    catalog as catalog,
    date_lines as date_lines,
    dates_for as dates_for,
    family_catalog as family_catalog,
    footer as footer,
    locate as locate,
    panel_for as panel_for,
    render as render,
    render_report_header as render_report_header,
    row_line as row_line,
    rows_for as rows_for,
    smoke_bucket as smoke_bucket,
    value_for as value_for,
)
from research.extraction.datasets.synthetic.types import (
    Assay as Assay,
    Bucket as Bucket,
    CaseDomain as CaseDomain,
    CellChangeEvidence as CellChangeEvidence,
    CellSpan as CellSpan,
    DatasetCase as DatasetCase,
    Domain as Domain,
    Evidence as Evidence,
    FamilySpec as FamilySpec,
    LengthSummary as LengthSummary,
    OmissionEvidence as OmissionEvidence,
    ReportProfile as ReportProfile,
    ReportStyle as ReportStyle,
    Split as Split,
    VISIBLE_FIELDS as VISIBLE_FIELDS,
    VisibleField as VisibleField,
    VisibleRow as VisibleRow,
)
# isort: on

MAX_WARNING_LENGTH = 160
NEAR_DUPLICATE_THRESHOLD = 0.85

REPAIRS = {"Глюкoза": "Глюкоза", "Отрищательный": "Отрицательный", "г/π": "г/л", "φπ": "фл"}


def visible_value(field: str, raw: str | None) -> str | None:
    """Interprets only a rendered cell; hidden source-record fields are unavailable."""
    if raw is None or raw.strip() in ("", "—", "–", "-"):
        return None
    raw = raw.strip()
    if field == "unit" and raw == "10°/π":
        return None
    if field == "value" and raw.endswith(" *"):
        raw = raw[:-2]
    for before, after in REPAIRS.items():
        raw = raw.replace(before, after)
    return raw


def select_date(lines: list[str]) -> str | None:
    for label in ("Дата взятия образца", "Дата выполнения", "Дата направления"):
        for line in lines:
            if line.startswith(label + ": "):
                return date.fromisoformat("-".join(reversed(line.split(": ", 1)[1].split(".")))).isoformat()
    return None


def _repair_rule_and_reason(field: VisibleField, before: str, after: str | None) -> tuple[int, str]:
    """Keeps evidence rule numbers aligned with the retained CA1 review policy."""
    rule = {"name": 2, "unit": 3, "value": 4, "reference_range": 5}[field]
    if after is None:
        return rule, "Visible cell is absent or ambiguous"
    if field == "name":
        return rule, "Identifiable mixed-script name glyph"
    if field == "unit":
        return rule, "Surviving unit glyphs identify the unit"
    if before.endswith(" *"):
        return rule, "Separately printed abnormality flag"
    return rule, "Identifiable qualitative spelling with polarity preserved"


def target_from_visible(
    rows: list[VisibleRow], spans: list[dict[str, CellSpan]], dates: list[str]
) -> tuple[CA1Target, list[Evidence]]:
    """Derives the CA1 target solely from rendered visible cells.

    Args:
        spans: Cell offsets into the final rendered report, aligned with rows.
        dates: Visible date lines; collection, performance, then order takes priority.

    Returns:
        The target and omission, repair, or abstention evidence. Historical and
        pending rows are excluded; missing result values produce warnings.

    Raises:
        ValueError: Rows and spans are misaligned, or a warning exceeds the
            160-character policy limit.
    """
    result: list[CA1Result] = []
    evidence: list[Evidence] = []
    unnamed: list[str] = []
    incomplete_unnamed = 0
    missing: list[str] = []

    for row, cell_spans in zip(rows, spans, strict=True):
        if row["state"] != "current":
            continue

        name = visible_value("name", row["name"])
        if name is None:
            unnamed.append(row["value"])
            incomplete_unnamed += visible_value("reference_range", row["reference_range"]) is None
            evidence.append(OmissionEvidence(kind="omission", reason="Visible name absent", spans=cell_spans))
            continue

        cells = CA1Result(
            name=name,
            value=visible_value("value", row["value"]),
            unit=visible_value("unit", row["unit"]),
            reference_range=visible_value("reference_range", row["reference_range"]),
        )
        if cells["value"] is None:
            missing.append(name)
        result.append(cells)

        for field in VISIBLE_FIELDS:
            before = row[field]
            after = cells[field]
            if after == before:
                continue
            rule, reason = _repair_rule_and_reason(field, before, after)
            evidence.append(
                CellChangeEvidence(
                    kind="repair" if after is not None else "abstention",
                    result_index=len(result) - 1,
                    field=field,
                    before=before,
                    after=after,
                    rule=rule,
                    reason=reason,
                    span=cell_spans[field],
                )
            )

    warnings = []
    if unnamed:
        quote = ", ".join(f"'{value}'" for value in unnamed[:3])
        detail = f" {incomplete_unnamed} reference cell(s) incomplete." if incomplete_unnamed else ""
        warnings.append(f"Omitted {len(unnamed)} unnamed row(s); visible value(s): {quote}.{detail}")
    if missing:
        quote = ", ".join(f"'{name}'" for name in missing[:2])
        warnings.append(f"Missing result value for {len(missing)} named observation(s): {quote}.")

    chosen_date = select_date(dates)
    if chosen_date is None:
        warnings.append("No report collection, performance or order date is visible.")
    target = CA1Target(date=chosen_date, results=result, errors=warnings)
    if any(len(warning) > MAX_WARNING_LENGTH for warning in warnings):
        raise ValueError("Warning exceeds 160 characters")
    return target, evidence


def _numeric_tokens(value: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*|[<>≤≥+−]", value)


def check_pair(
    text: str,
    rows: list[VisibleRow],
    target: CA1Target,
    evidence: list[Evidence],
    schema: dict[str, Any],
) -> None:
    """Validates a target and its evidence against the rendered report.

    Raises:
        ValueError: Coverage, source spans, visible-cell values, numeric tokens,
            or warning language violate the CA1 policy.
        jsonschema.ValidationError: The target fails the packet schema.
    """
    validate_target(json.dumps(target, ensure_ascii=False), schema)

    expected_count = sum(r["state"] == "current" and visible_value("name", r["name"]) is not None for r in rows)
    if len(target["results"]) != expected_count:
        raise ValueError("Visible observation coverage mismatch")
    for item in evidence:
        source_spans = item["spans"].values() if item["kind"] == "omission" else [item["span"]]
        if any(text[s["start"] : s["end"]] != s["text"] for s in source_spans):
            raise ValueError("Evidence span mismatch")

    current = [r for r in rows if r["state"] == "current" and visible_value("name", r["name"]) is not None]
    for source, result in zip(current, target["results"], strict=True):
        for field in VISIBLE_FIELDS:
            before, after = source[field], result[field]
            if after != visible_value(field, before):
                raise ValueError("Target differs from visible-cell CA1 reading")
            if field in ("value", "reference_range") and after is not None:
                if _numeric_tokens(before.removesuffix(" *")) != _numeric_tokens(after):
                    raise ValueError("Measured number, bound or polarity changed")

    if any(re.search(r"[А-Яа-я]", re.sub(r"'[^']*'", "", warning)) for warning in target["errors"]):
        raise ValueError("Non-English warning text")


def ensure_split_and_provenance(items: list[DatasetCase], sources: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Enforces source roles and family isolation across packet splits.

    Returns:
        Each family mapped to its single train or validation split.

    Raises:
        ValueError: An ID or normalized input repeats, a family/source group
            crosses splits, or a reserved or excluded source informs generation.
    """
    families: dict[str, str] = {}
    groups: dict[str, str] = {}
    ids: set[str] = set()
    inputs: set[str] = set()
    normalized: set[str] = set()
    for item in items:
        if item["id"] in ids:
            raise ValueError("Duplicate ID")
        ids.add(item["id"])
        family_split = families.setdefault(item["family"], item["split"])
        group_split = groups.setdefault(item["source_group"], item["split"])
        if family_split != item["split"] or group_split != item["split"]:
            raise ValueError("Family/source-group split leakage")
        for source in item["development_sources"]:
            if source not in ELIGIBLE or sources[source]["role"] != "development":
                raise ValueError("Excluded or reserved provenance")
        digest = item["input_sha256"]
        compact = sha("".join(item["input"].split()))
        if digest in inputs or compact in normalized:
            raise ValueError("Duplicate input")
        inputs.add(digest)
        normalized.add(compact)
    return families


def near_duplicate_candidates(items: list[DatasetCase]) -> list[dict[str, Any]]:
    """Screens representative parser layouts across splits for manual adjudication."""
    representatives: dict[str, DatasetCase] = {}
    for item in items:
        representatives.setdefault(item["family"], item)

    def skeleton(item: DatasetCase) -> str:
        lines = []
        for line in item["input"].splitlines():
            if line.startswith(("Дата ", "Номер бланка", "Раздел ", "ЛАБОРАТОРНЫЙ")):
                continue
            if line in ("Результаты:", "Исследование завершено"):
                continue
            line = re.sub(r"[\wА-Яа-я]+", "X", line)
            lines.append(line)
        return "\n".join(lines[:22])

    train = [item for item in representatives.values() if item["split"] == "train"]
    validation = [item for item in representatives.values() if item["split"] == "validation"]
    candidates = []
    for left in train:
        for right in validation:
            score = difflib.SequenceMatcher(None, skeleton(left), skeleton(right)).ratio()
            if score >= NEAR_DUPLICATE_THRESHOLD:
                candidates.append(
                    dict(
                        train_family=left["family"],
                        validation_family=right["family"],
                        structural_similarity=round(score, 3),
                    )
                )
    return sorted(candidates, key=lambda x: -x["structural_similarity"])


def quantiles(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {"p50": median(ordered), "p95": ordered[(len(ordered) * 95 + 99) // 100 - 1], "max": ordered[-1]}


def profile(texts: list[str]) -> ReportProfile:
    lengths = [len(text) for text in texts]
    lines = [len(text.splitlines()) for text in texts]
    first = [
        next((line for line in text.splitlines() if line.strip() and line.strip() != "IMAGE"), "") for text in texts
    ]

    return ReportProfile(
        count=len(texts),
        characters=LengthSummary(min=min(lengths), median=median(lengths), max=max(lengths)),
        lines=LengthSummary(min=min(lines), median=median(lines), max=max(lines)),
        with_image=sum("IMAGE" in text for text in texts),
        with_person=sum("<PERSON>" in text for text in texts),
        with_organization=sum("<ORGANIZATION>" in text for text in texts),
        with_website=sum("www." in text.lower() for text in texts),
        unique_openings=len(set(first)),
        max_opening_count=max(Counter(first).values()),
    )
