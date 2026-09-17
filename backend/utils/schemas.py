import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from configs.enums import ProviderNameEnum, StepTypeEnum, ToolNameEnum


class PipelineRunContext(BaseModel):
    """Data structure passed between the Anonymization and Analysis phases."""

    pipeline_run_id: int
    patient_id: int
    anonymized_path: Path
    source_path: Path

    report_data: dict | None = None  # In case the document was already processed and report is done
    pipeline_run_uuid: uuid.UUID | None = None
    source_artifact_id: int | None = None
    anonymized_artifact_id: int | None = None
    processor_model_name: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class PipelineSchema(BaseModel):
    """Schema for pipeline metadata.

    This class is used to filter pipelines based on specific criteria.
    """

    task: str
    status: str


class PipelineStepSchema(BaseModel):
    """Schema for pipeline step information.

    This class is used to describe pipeline processing step.
    """

    step_type: StepTypeEnum | None = None
    tool_name: ToolNameEnum | None = None
    model_version: str | None = None
    prompt_path: str | None = None
    provider: ProviderNameEnum | None = None
    config: dict[str, Any] | None = None
    input_artifact_id: int | None = None
    output_artifact_id: int | None = None
    finished_at: datetime | None = None
    extra_data: dict[str, Any] | None = None


class ArtifactSchema(BaseModel):
    """Schema for artifact metadata.

    This class represents metadata for files or data artifacts in the system.
    """

    storage_type: str | None = None
    storage_path: str | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None
    mime_type: str | None = None
    size: int | None = None


class ReportCreateSchema(BaseModel):
    """Schema for creating a new report.

    This class is used to validate and structure report data before it's stored.
    """

    source_artifact_id: int
    report_hash: str
    raw_json: dict[str, Any]
    processed_at: datetime = Field(default_factory=datetime.now)

    report_date: datetime | None = None
    source_path: str | None = None


class ReportFilterSchema(BaseModel):
    """Schema for filtering reports.

    This class is used to filter reports based on specific criteria.
    """

    source_artifact_id: int | None = None


class TestCatalogSchema(BaseModel):
    """Schema for test catalog entries.

    This class represents standard test information in the catalog.
    """

    canonical_name: str

    preferred_unit: str | None = None


class TestObservationSchema(BaseModel):
    """Schema for test observations.

    This class represents the results of individual laboratory tests.
    """

    report_id: int
    patient_id: int

    test_catalog_id: int | None = None
    test_name: str | None = None
    observed_value: float | None = None
    observed_value_inequality: str | None = None
    unit: str | None = None
    flag: str | None = None
    reference_range: str | None = None
    observation_date: datetime | None = None
    created_at: datetime | None = None
