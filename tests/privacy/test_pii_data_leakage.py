import fitz
import pytest

from configs.testing import TestingSettings
from src.databases.postgres.models import *

testing_settings = TestingSettings()


@pytest.fixture(scope="module")
async def anonymized_document(pipeline):
    processor, patient_id = pipeline
    context = await processor.anonymize(testing_settings.test_user_data_path, patient_id=patient_id)
    anonymized_document_text = "\n".join(page.get_text() for page in fitz.open(context.anonymized_path))
    return anonymized_document_text, context.anonymized_path


@pytest.mark.parametrize("entity, pii_text", testing_settings.test_user_PII_data.items())
async def test_pii_data_leakage(anonymized_document, entity, pii_text, empty_test_database, remove_paths):
    anonymized_document_text, anonymized_path = anonymized_document
    error_message = f"Error: {entity} (%s) was not marked as PII data by anonymizer!"
    for pii_text_value in pii_text:
        assert pii_text_value.lower() not in anonymized_document_text.lower(), error_message % pii_text_value

    # Clean database and remove generated test files
    tables_to_drop = (Artifact.__tablename__, PipelineRun.__tablename__, PipelineStep.__tablename__)
    empty_test_database(*tables_to_drop)
    remove_paths(paths=(anonymized_path,))
