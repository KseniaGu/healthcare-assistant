"""Shared data contracts for deterministic synthetic report generation."""

from typing import Literal, NotRequired, TypedDict

from research.extraction.datasets.packet import ChatMessage

Assay = tuple[str, str | None, str]
Split = Literal["train", "validation"]
Domain = Literal["chemistry", "hematology", "urine", "qualitative"]
CaseDomain = Domain | Literal["smoke"]
Bucket = Literal["ordinary", "layout", "difficult"]
ReportStyle = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
VisibleField = Literal["name", "value", "unit", "reference_range"]
VISIBLE_FIELDS: tuple[VisibleField, ...] = ("name", "value", "unit", "reference_range")


class FamilySpec(TypedDict):
    family: str
    split: Split
    domain: Domain
    style: ReportStyle
    count: int
    development_sources: list[str]


class VisibleRow(TypedDict):
    name: str
    value: str
    unit: str
    reference_range: str
    state: Literal["current", "history", "pending"]


class CellSpan(TypedDict):
    start: int
    end: int
    text: str


class OmissionEvidence(TypedDict):
    kind: Literal["omission"]
    reason: str
    spans: dict[str, CellSpan]


class CellChangeEvidence(TypedDict):
    kind: Literal["repair", "abstention"]
    result_index: int
    field: VisibleField
    before: str
    after: str | None
    rule: int
    reason: str
    span: CellSpan


Evidence = OmissionEvidence | CellChangeEvidence


class LengthSummary(TypedDict):
    min: int
    median: int | float
    max: int


class ReportProfile(TypedDict):
    count: int
    characters: LengthSummary
    lines: LengthSummary
    with_image: int
    with_person: int
    with_organization: int
    with_website: int
    unique_openings: int
    max_opening_count: int


class DatasetCase(TypedDict):
    id: str
    split: Split
    family: str
    source_group: str
    domain: CaseDomain
    bucket: Bucket
    clean: bool
    smoke: bool
    tags: list[str]
    transformations: list[str]
    development_sources: list[str]
    input: str
    input_sha256: str
    target_sha256: str
    evidence: list[Evidence]
    messages: list[ChatMessage]
    review_status: str
    review_notes: NotRequired[list[str]]
