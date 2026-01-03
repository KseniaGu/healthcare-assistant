import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from configs.enums import StepTypeEnum, ToolNameEnum, ProviderNameEnum


class PipelineRunContext(BaseModel):
    """Data structure passed between the Anonymization and Analysis phases."""
    pipeline_run_id: int
    patient_id: int
    anonymized_path: Path
    source_path: Path

    report_data: Optional[dict] = None  # In case the document was already processed and report is done
    pipeline_run_uuid: Optional[uuid.UUID] = None
    source_artifact_id: Optional[int] = None
    anonymized_artifact_id: Optional[int] = None
    processor_model_name: Optional[str] = None

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
    step_type: Optional[StepTypeEnum] = None
    tool_name: Optional[ToolNameEnum] = None
    model_version: Optional[str] = None
    prompt_path: Optional[str] = None
    provider: Optional[ProviderNameEnum] = None
    config: Optional[Dict[str, Any]] = None
    input_artifact_id: Optional[int] = None
    output_artifact_id: Optional[int] = None
    finished_at: Optional[datetime] = None
    extra_data: Optional[Dict[str, Any]] = None


class ArtifactSchema(BaseModel):
    """Schema for artifact metadata.

    This class represents metadata for files or data artifacts in the system.
    """
    storage_type: Optional[str] = None
    storage_path: Optional[str] = None
    checksum: Optional[str] = None
    checksum_algorithm: Optional[str] = None
    mime_type: Optional[str] = None
    size: Optional[int] = None


class ReportCreateSchema(BaseModel):
    """Schema for creating a new report.

    This class is used to validate and structure report data before it's stored.
    """
    source_artifact_id: int
    report_hash: str
    raw_json: Dict[str, Any]
    processed_at: datetime = Field(default_factory=datetime.now)

    report_date: Optional[datetime] = None
    source_path: Optional[str] = None


class ReportFilterSchema(BaseModel):
    """Schema for filtering reports.

    This class is used to filter reports based on specific criteria.
    """
    source_artifact_id: Optional[int] = None


class TestCatalogSchema(BaseModel):
    """Schema for test catalog entries.

    This class represents standard test information in the catalog.
    """
    canonical_name: str

    preferred_unit: Optional[str] = None


class TestObservationSchema(BaseModel):
    """Schema for test observations.

    This class represents the results of individual laboratory tests.
    """
    report_id: int
    patient_id: int

    test_catalog_id: Optional[int] = None
    test_name: Optional[str] = None
    observed_value: Optional[float] = None
    observed_value_inequality: Optional[str] = None
    unit: Optional[str] = None
    flag: Optional[str] = None
    reference_range: Optional[str] = None
    observation_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
