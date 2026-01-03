from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Sequence

from alembic import command
from alembic.config import Config
from psycopg import errors
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.utils.exceptions import CriticalDatabaseSideError
from backend.utils.schemas import *
from configs.constants import ROOT_DIR
from configs.database import PostgreSQLSettings
from src.databases.postgres.models import *
from src.databases.postgres.utils import connection


class PostgreSQLDB:
    """SQLAlchemy wrapper class for PostgreSQL database."""

    def __init__(self, database_settings: PostgreSQLSettings):
        """Initializes database connection."""
        self.settings = database_settings

        self._init_database()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async context manager for database sessions."""
        async with self.async_session() as session:
            yield session

    def _init_database(self):
        """Initializes database session.

        Creates tables and run migrations if AUTO_MIGRATE is on (DEV environment only).
        """
        if os.getenv("ENV", "DEV") == "DEV":
            sync_engine = create_engine(self.settings.get_connection_url(), echo=False)
            try:
                with sync_engine.begin() as conn:
                    # To use uuid_generate_v4() function
                    conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
                    Base.metadata.create_all(bind=conn, checkfirst=True)
            except OperationalError as e:
                raise CriticalDatabaseSideError(f"Database initialization failed: {str(e)}")

            if self.settings.auto_migrate:
                self.run_migrations()

        self.engine = create_async_engine(self.settings.get_connection_url(), echo=False)
        self.async_session = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    def run_migrations(self):
        """Runs alembic migrations."""
        alembic_cfg = Config(str(ROOT_DIR / "alembic.ini"))
        alembic_cfg.set_main_option('sqlalchemy.url', self.settings.get_connection_url())
        alembic_cfg.set_main_option('script_location', str(ROOT_DIR / 'src/databases/postgres/migrations/alembic'))
        command.upgrade(alembic_cfg, 'head')

    @connection
    async def select(self, table_obj: type[Base], filters: BaseModel = None, session: AsyncSession = None):
        """Executes select query with filters (if provided)."""
        if filters is not None:
            filters = filters.model_dump(exclude_unset=True)
            filters = (getattr(table_obj, key, value) == value for key, value in filters.items())
            result = await session.execute(select(table_obj).where(*filters))
        else:
            result = await session.execute(select(table_obj))
        return result.scalars()

    @connection
    async def get_pipeline(self, pipeline_data: PipelineSchema, session: AsyncSession = None) -> Pipeline.id | None:
        """Gets pipeline by the given attributes."""
        result = await self.select(Pipeline, pipeline_data)
        result = result.first()
        if result:
            return result.id
        return None

    @connection
    async def add_pipeline_run(
            self,
            patient_id: Patient.id,
            task: TaskEnum,
            session: AsyncSession = None
    ) -> (PipelineRun.id, PipelineRun.uuid):
        """Adds started pipeline run."""
        pipeline_data = PipelineSchema(task=task.name, status=PipelineStatusEnum.active.name)
        pipeline = await self.get_pipeline(pipeline_data)
        pipeline_run = PipelineRun(pipeline_id=pipeline, patient_id=patient_id)
        session.add(pipeline_run)
        await session.flush()

        return pipeline_run.id, pipeline_run.uuid

    @connection
    async def add_pipeline_step(
            self,
            pipeline_run_id: Pipeline.id,
            pipeline_step_data: PipelineStepSchema,
            session: AsyncSession = None
    ) -> PipelineStep.id:
        """Adds pipeline step."""
        step_data = pipeline_step_data.model_dump(exclude_unset=True)
        pipeline_step = PipelineStep(pipeline_run_id=pipeline_run_id, **step_data)
        session.add(pipeline_step)
        await session.flush()

        return pipeline_step.id

    @connection
    async def add_artifact(self, artifact_data: ArtifactSchema, session: AsyncSession = None) -> Artifact.id:
        """Adds artifact."""
        artifact = Artifact(**artifact_data.model_dump(exclude_unset=True))
        session.add(artifact)
        try:
            await session.flush()
        except IntegrityError as e:
            # Artifact is duplicated
            if isinstance(e.orig, errors.UniqueViolation):
                await session.rollback()
                return None

        return artifact.id

    @connection
    async def get_artifact(self, artifact_data: ArtifactSchema, session: AsyncSession = None) -> Artifact.id | None:
        """Gets artifact by the given attributes."""
        result = await self.select(Artifact, artifact_data)
        result = result.first()
        if result:
            return result.id
        return None

    @connection
    async def finish_pipeline_run(
            self,
            pipeline_run_id: int,
            finished_at: datetime,
            session: AsyncSession = None
    ) -> bool:
        """Finishes pipeline run.

        Returns True if the run exists and its <finished_at> has been successfully updated and False otherwise.
        """
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == pipeline_run_id))
        result = result.scalars().first()
        if result:
            result.finished_at = finished_at
            return True
        return False

    @connection
    async def add_report(
            self,
            pipeline_run_id: int,
            patient_id: int,
            report_data: ReportCreateSchema,
            session: AsyncSession = None
    ) -> Report:
        """Adds generated report to the database."""
        report_data = report_data.model_dump(exclude_unset=True)
        report = Report(pipeline_run_id=pipeline_run_id, patient_id=patient_id, **report_data)
        session.add(report)
        await session.flush()
        return report

    @connection
    async def get_report(
            self,
            report_data: ReportFilterSchema,
            session: AsyncSession = None
    ) -> (Report.raw_json | None, Report.pipeline_run_id | None):
        """Gets report and its pipeline run id given report data."""
        result = await self.select(Report, report_data)
        result = result.first()
        if result:
            return result.raw_json, result.pipeline_run_id
        return None, None

    @connection
    async def get_test_catalog_id(
            self,
            test_data: TestCatalogSchema,
            to_add: bool = True,
            session: AsyncSession = None
    ) -> TestCatalog.id | None:
        """Gets test catalog id. If no test found and <to_add> is True, adds new test."""
        # TODO: Make test name unification
        result = await self.select(TestCatalog, test_data)
        result = result.first()
        if result:
            return result.id

        if to_add:
            test_catalog = TestCatalog(**test_data.model_dump(exclude_unset=True))
            session.add(test_catalog)
            await session.flush()

            return test_catalog.id

        return None

    @connection
    async def add_test_observations(self, test_observations: list[TestObservationSchema], session: AsyncSession = None):
        """Adds test observations extracted from generated report."""
        observations = [TestObservation(**test_obs.model_dump(exclude_unset=True)) for test_obs in test_observations]
        session.add_all(observations)

    @connection
    async def get_patient(self, name: str, session: AsyncSession = None) -> Patient.id | None:
        """Gets patient database identifier given its name."""
        result = await session.execute(select(Patient).where(Patient.name == name))
        result = result.scalars().first()
        if result:
            return result.id
        return None

    @connection
    async def get_patient_observations(
            self,
            patient_id: int,
            session: AsyncSession = None
    ) -> Sequence[TestObservation]:
        """Gets all test observations for a given patient ordered by date."""
        query = (
            select(TestObservation)
            .where(TestObservation.patient_id == patient_id)
            .order_by(TestObservation.observation_date.asc(), TestObservation.created_at.asc())
        )
        result = await session.execute(query)
        return result.scalars().all()

    async def close(self) -> None:
        """Closes the database connection pool."""
        await self.engine.dispose()
