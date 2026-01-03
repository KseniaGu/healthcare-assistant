import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy import text

from backend.data_processing.laboratory_test import LaboratoryTestProcessor
from backend.utils.common_funcs import get_logger
from configs.constants import BASE_PATIENT_NAME
from configs.data import DataSettings
from configs.database import PostgreSQLSettings
from configs.models import ModelSettings
from configs.paths import PathSettings
from configs.testing import TestingSettings
from src.databases.postgres.models import *

testing_settings = TestingSettings()
logger, _ = get_logger("testing")


@pytest.fixture(scope="session")
async def pipeline():
    postgresql_settings = PostgreSQLSettings(database_name=testing_settings.test_database_name)
    processor = LaboratoryTestProcessor(ModelSettings(), DataSettings(), PathSettings(), postgresql_settings)
    patient_id = await processor.database.get_patient(BASE_PATIENT_NAME)
    return processor, patient_id


@pytest.fixture
async def empty_test_database(pipeline):
    """Truncates only used test database tables after each test."""
    processor, _ = pipeline
    assert processor.database.engine.url.database == testing_settings.test_database_name, "Current database is not for testing!"
    tables_to_clean = set()

    def _mark_for_cleaning(*table_names):
        tables_to_clean.update(table_names)

    yield _mark_for_cleaning

    if tables_to_clean:
        logger.info(f"\nTruncating tables: {tables_to_clean}")
        try:
            async with processor.database.engine.begin() as conn:
                for table in tables_to_clean:
                    await conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
        except Exception as e:
            logger.error(f"Error while trying to truncate tables {tables_to_clean}: {e}")


@pytest.fixture
def remove_paths():
    def _remove_paths(paths: tuple[Path]):
        for path in paths:
            if not path.exists():
                continue
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

    return _remove_paths


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == pytest.ExitCode.OK.value:
        try:
            postgresql_settings = PostgreSQLSettings(database_name=testing_settings.test_database_name)
            sync_engine = create_engine(postgresql_settings.get_connection_url(), echo=False)
            with sync_engine.begin() as conn:
                Base.metadata.drop_all(bind=conn)
                conn.execute(text("DROP TABLE alembic_version CASCADE;"))
            logger.info("All test database schemas dropped successfully.")
        except Exception as e:
            logger.error(f"Sessionfinish hook failed: {e}")
