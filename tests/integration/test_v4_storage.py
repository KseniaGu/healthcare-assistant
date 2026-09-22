"""Opt-in PostgreSQL test using only a temporary schema inside a rolled-back transaction.

Run: poetry run python -m unittest tests.integration.test_v4_storage -v
This does not use the destructive pytest session hooks or initialize the PDF pipeline.
"""

import runpy
import unittest
import uuid
from datetime import datetime
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.data_processing.restore_observations import restore_observations
from backend.utils.helpers import observation_values
from configs.constants import DEFAULT_LAB_TEST_NAME
from configs.data import DataSettings
from configs.database import PostgreSQLSettings
from src.databases.postgres.models import Artifact, Base, Patient, Pipeline, PipelineRun, Report, TestObservation


class V4StorageTests(unittest.TestCase):
    def test_migration_restoration_and_rerun_preserve_existing_data(self):
        engine = create_engine(PostgreSQLSettings().get_connection_url())
        self.addCleanup(engine.dispose)
        migration = runpy.run_path(
            str(
                Path(__file__).resolve().parents[2]
                / ("src/databases/postgres/migrations/alembic/versions/c84d120a6e31_preserve_observation_text.py")
            )
        )
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                schema = "v4_test_" + uuid.uuid4().hex
                connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
                # Force creation in the temporary schema despite same-named public tables.
                Base.metadata.create_all(connection, checkfirst=False)
                with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
                    patient = Patient(name="Synthetic")
                    pipeline = Pipeline(name="Synthetic", version="1", task=1, status=1)
                    artifact = Artifact(storage_path="synthetic.pdf")
                    session.add_all([patient, pipeline, artifact])
                    session.flush()
                    run = PipelineRun(pipeline_id=pipeline.id, patient_id=patient.id)
                    session.add(run)
                    session.flush()
                    source = {
                        "results": [
                            {"test_name": "Legacy scalar", "value": "4.50", "unit": "mg/L", "flag": "Normal"},
                            {"name": "Qualitative", "value": "negative", "unit": None},
                            {"name": "Range", "value": "0-1", "unit": "cells"},
                            {"name": "Limit", "value": "< 3.0", "unit": "mg/L"},
                            {"name": "Duplicate", "value": "1"},
                            {"name": "Duplicate", "value": "2"},
                            {"name": None, "value": "7"},
                            {"name": "Pending", "value": "Выполняется"},
                        ]
                    }
                    report = Report(
                        pipeline_run_id=run.id,
                        patient_id=patient.id,
                        source_path="synthetic.pdf",
                        source_artifact_id=artifact.id,
                        report_hash="a" * 64,
                        report_date=datetime(2025, 1, 1),
                        raw_json=source,
                    )
                    session.add(report)
                    session.flush()
                    old = TestObservation(
                        report_id=report.id,
                        patient_id=patient.id,
                        test_name=f"{DEFAULT_LAB_TEST_NAME}_{report.uuid}_0",
                        observed_value=4.5,
                        observed_value_inequality="",
                        unit=None,
                        flag="normal",
                    )
                    session.add(old)
                    session.flush()
                    old_id, report_id, patient_id = old.id, report.id, patient.id
                    session.commit()
                with Operations.context(MigrationContext.configure(connection)):
                    migration["downgrade"]()  # Simulate the pre-v4 schema, only in this temporary schema.
                    migration["upgrade"]()
                    migration["upgrade"]()  # Also covers fresh DEV create_all before migration.
                with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
                    first = restore_observations(session)
                    self.assertEqual(first["added"], 3)
                    self.assertEqual(first["renamed"], 1)
                    self.assertEqual(first["text_restored"], 1)
                    self.assertEqual(len(first["skipped"]), 4)
                    observations = list(session.scalars(select(TestObservation).order_by(TestObservation.id)))
                    self.assertEqual(len(observations), 4)
                    self.assertEqual(observations[0].id, old_id)
                    self.assertEqual(observations[0].observed_value, 4.5)
                    self.assertEqual(observations[0].observed_value_text, "4.50")
                    self.assertEqual(observations[0].test_name, "Legacy scalar")
                    self.assertEqual(observations[0].unit, "mg/L")
                    self.assertEqual(observations[0].flag.value, "normal")
                    self.assertEqual(observations[1].observed_value_text, "negative")
                    self.assertIsNone(observations[1].observed_value)
                    self.assertEqual(observations[2].observed_value_text, "0-1")
                    self.assertIsNone(observations[2].observed_value)
                    self.assertEqual(observations[3].observed_value_inequality, "<")
                    self.assertEqual(session.get(Report, report_id).raw_json, source)
                    second = restore_observations(session)
                    self.assertEqual(
                        {k: v for k, v in second.items() if k != "skipped"},
                        {
                            "added": 0,
                            "renamed": 0,
                            "text_restored": 0,
                            "units_restored": 0,
                        },
                    )
                    # A new v4 decimal-comma value round-trips with literal precision intact.
                    values = observation_values({"value": "2,30", "unit": "mg/L"}, DataSettings().common_inequalities)
                    fresh = TestObservation(report_id=report_id, patient_id=patient_id, test_name="Fresh", **values)
                    session.add(fresh)
                    session.flush()
                    session.expire(fresh)
                    self.assertEqual(fresh.observed_value, 2.3)
                    self.assertEqual(fresh.observed_value_text, "2,30")
                    self.assertEqual(fresh.flag.value, "unknown")
                    session.commit()
            finally:
                transaction.rollback()  # Removes the synthetic schema and every test row.


if __name__ == "__main__":
    unittest.main()
