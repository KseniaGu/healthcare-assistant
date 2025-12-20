import uuid as _uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Index, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, BIGINT, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func, text

from configs.enums import StepTypeEnum, TestFlagTypeEnum, ToolNameEnum, ProviderNameEnum, TaskEnum, PipelineStatusEnum


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class Patient(Base):
    """A class representing a patient in the healthcare system.

    Attributes:
        id: Unique identifier for the patient.
        name: Full name of the patient.
        dob: Date of birth of the patient.
        reports: List of medical reports associated with the patient.
        observations: List of test observations for the patient.
        pipeline_runs: List of pipeline runs associated with the patient.
    """
    __tablename__ = "patient"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    dob: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    reports: Mapped[list["Report"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    observations: Mapped[list["TestObservation"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    pipeline_runs: Mapped[list["PipelineRun"]] = relationship(back_populates="patient", cascade="none")


class Report(Base):
    """A class representing a medical report generated from a pipeline run.
    
    Attributes:
        id: Unique identifier for the report.
        uuid: Universally unique identifier for the report.
        pipeline_run_id: Reference to the pipeline run that generated this report.
        patient_id: Reference to the patient this report belongs to.
        report_date: Date when the medical test was conducted.
        source_path: Path to the original report file.
        source_artifact_id: Reference to the artifact containing the source data.
        report_hash: Hash of the report content for deduplication.
        processed_at: Timestamp when the report was processed.
        raw_json: Raw JSON data of the report.
    """
    __tablename__ = "report"

    id: Mapped[int] = mapped_column(primary_key=True)
    uuid: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, server_default=text("uuid_generate_v4()"), unique=True, index=True
    )
    pipeline_run_id: Mapped[int] = mapped_column(ForeignKey("pipeline_run.id"), nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patient.id", ondelete="CASCADE"), index=True, nullable=False)
    report_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=False)
    source_artifact_id: Mapped[int] = mapped_column(ForeignKey("artifact.id"), nullable=False)
    report_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    raw_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="reports")
    observations: Mapped[list["TestObservation"]] = relationship(back_populates="report", cascade="all, delete-orphan")
    pipeline_run: Mapped["PipelineRun"] = relationship(back_populates="report", cascade="none")


class TestCatalog(Base):
    """A class for catalog of standard laboratory tests.
    
    Attributes:
        id: Unique identifier for the test catalog entry.
        canonical_name: Standardized name of the test.
        preferred_unit: Preferred unit of measurement for the test results.
    """
    __tablename__ = "test_catalog"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(256), unique=True, index=True, nullable=False)
    preferred_unit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class TestObservation(Base):
    """A class representing a single test observation or result.
    
    Attributes:
        id: Unique identifier for the observation.
        test_catalog_id: Reference to the standard test in the catalog.
        report_id: Reference to the report containing this observation.
        patient_id: Reference to the patient this observation belongs to.
        test_name: Name of the test as it appears in the report.
        observed_value: Numeric result of the test.
        unit: Unit of measurement for the observed value.
        flag: Indicates if the result is high, low or normal.
        reference_range: Normal range for the test result.
        observation_date: Date when the test was conducted.
        created_at: Timestamp when the observation was recorded.
    """
    __tablename__ = "test_observation"

    id: Mapped[int] = mapped_column(primary_key=True)
    test_catalog_id: Mapped[Optional[int]] = mapped_column(ForeignKey("test_catalog.id"), nullable=True, index=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("report.id", ondelete="CASCADE"), index=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patient.id", ondelete="CASCADE"), index=True, nullable=False)

    test_name: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)
    observed_value: Mapped[Optional[float]] = mapped_column(nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    flag: Mapped[Optional[TestFlagTypeEnum]] = mapped_column(
        Enum(TestFlagTypeEnum, name="test_flag_type_enum"), nullable=True,
    )
    reference_range: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    observation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    catalog: Mapped["TestCatalog"] = relationship(backref="observations")
    report: Mapped["Report"] = relationship(back_populates="observations")
    patient: Mapped["Patient"] = relationship(back_populates="observations")


class Artifact(Base):
    """A class representing a stored file or data artifact in the system.
    
    Attributes:
        id: Unique identifier for the artifact.
        uuid: Universally unique identifier for the artifact.
        storage_type: Type of storage (e.g., 's3', 'local').
        storage_path: Path to the artifact in storage.
        checksum: Checksum of the artifact content.
        checksum_algorithm: Algorithm used to generate the checksum.
        mime_type: MIME type of the artifact.
        size: Size of the artifact in bytes.
        created_at: Timestamp when the artifact was created.
    """
    __tablename__ = "artifact"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    uuid: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, server_default=text("uuid_generate_v4()"), unique=True, index=True
    )
    storage_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)  # E.g., "s3", "hetzner"
    storage_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True,
                                                        index=True)  # E.g., "s3://bucket/key"
    checksum: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=True)
    checksum_algorithm: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    size: Mapped[Optional[int]] = mapped_column(BIGINT, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    input_steps: Mapped[list["PipelineStep"]] = relationship(
        back_populates="input_artifact",
        foreign_keys="PipelineStep.input_artifact_id",
        cascade="none",
    )
    output_steps: Mapped[list["PipelineStep"]] = relationship(
        back_populates="output_artifact",
        foreign_keys="PipelineStep.output_artifact_id",
        cascade="none",
    )
    __table_args__ = (Index("ix_artifact_storage_type_and_path", "storage_type", "storage_path"),)


class Pipeline(Base):
    """A class representing a processing pipeline configuration.
    
    Attributes:
        id: Unique identifier for the pipeline.
        name: Name of the pipeline.
        version: Version of the pipeline.
        description: Description of the pipeline's general steps.
        task: Type of task the pipeline performs.
        status: Current status of the pipeline.
        created_at: Timestamp when the pipeline was created.
    """
    __tablename__ = "pipeline"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    task: Mapped[TaskEnum] = mapped_column(Enum(TaskEnum, name="task_enum"), nullable=False, index=True)

    status: Mapped[PipelineStatusEnum] = mapped_column(
        Enum(PipelineStatusEnum, name="pipeline_status_enum"), nullable=False, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PipelineRun(Base):
    """A class representing an execution instance of a pipeline.

    Attributes:
        id: Unique identifier for the pipeline run.
        uuid: Universally unique identifier for the run.
        pipeline_id: Reference to the pipeline configuration.
        patient_id: Reference to the patient being processed.
        started_at: Timestamp when the run started.
        finished_at: Timestamp when the run completed.
    """
    __tablename__ = "pipeline_run"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    uuid: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, server_default=text("uuid_generate_v4()"), unique=True, index=True
    )
    pipeline_id: Mapped[int] = mapped_column(ForeignKey("pipeline.id", ondelete="CASCADE"), index=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patient.id", ondelete="CASCADE"), index=True, nullable=False)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list["PipelineStep"]] = relationship(back_populates="pipeline_run", cascade="all, delete-orphan")
    patient: Mapped["Patient"] = relationship(back_populates="pipeline_runs")
    report: Mapped["Report"] = relationship(back_populates="pipeline_run", uselist=False)


class PipelineStep(Base):
    """A class representing a single step in a pipeline execution.

    Attributes:
        id: Unique identifier for the step.
        pipeline_run_id: Reference to the parent pipeline run.
        step_type: Type of processing step.
        tool_name: Name of the tool used in this step.
        model_version: Version of the model used (if applicable).
        prompt_path: Path to the prompt template (if applicable).
        provider: Service provider for this step.
        config: Configuration parameters for the step.
        input_artifact_id: Reference to the input artifact.
        output_artifact_id: Reference to the output artifact.
        started_at: Timestamp when the step started.
        finished_at: Timestamp when the step completed.
        extra_data: Additional data or metadata for the step.
    """
    __tablename__ = "pipeline_step"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    pipeline_run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True, nullable=False
    )
    step_type: Mapped[Optional[StepTypeEnum]] = mapped_column(Enum(StepTypeEnum, name="step_type_enum"), nullable=True)
    tool_name: Mapped[Optional[ToolNameEnum]] = mapped_column(Enum(ToolNameEnum, name="tool_name_enum"), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prompt_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    provider: Mapped[Optional[ProviderNameEnum]] = mapped_column(
        Enum(ProviderNameEnum, name="provider_name_enum"), nullable=True,
    )

    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    input_artifact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("artifact.id"), nullable=True)
    output_artifact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("artifact.id"), nullable=True)

    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Extra data
    extra_data: Mapped[Optional[str]] = mapped_column(JSONB, nullable=True)  # E.g. raw text generated by LLM

    pipeline_run: Mapped["PipelineRun"] = relationship(back_populates="steps")
    input_artifact: Mapped[list["Artifact"]] = relationship(foreign_keys=[input_artifact_id])
    output_artifact: Mapped[list["Artifact"]] = relationship(foreign_keys=[output_artifact_id])
