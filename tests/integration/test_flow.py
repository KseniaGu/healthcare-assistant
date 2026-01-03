from datetime import timezone

from backend.utils.common_funcs import read_file
from backend.utils.schemas import PipelineStepSchema
from configs.models import ModelSettings
from configs.testing import TestingSettings
from src.databases.postgres.models import *
from tests.utils import check_reports, check_test_observations, check_pipeline_run

testing_settings = TestingSettings()
model_settings = ModelSettings()


async def test_pipeline_saves_to_database(mocker, pipeline, empty_test_database, remove_paths):
    """Tests that if the APIs return valid data, it gets saved to the database correctly."""
    processor, patient_id = pipeline

    # Setup mocks
    mask_pii_return_value = PipelineStepSchema(
        step_type=StepTypeEnum.anonymize,
        tool_name=model_settings.pii_extractor.tool_name,
        model_version=model_settings.pii_extractor.model_path,
        provider=model_settings.pii_extractor.provider,
        config={"tokenizer_path": model_settings.pii_extractor.tokenizer_path},
        finished_at=datetime.now(timezone.utc),
    )
    parse_return_value = read_file(testing_settings.test_user_data_parsed_path)
    gemini_output_path = testing_settings.test_user_data_gemini_output_path
    generate_return_value = (read_file(gemini_output_path), {}, gemini_output_path)

    mocker.patch.object(processor, "mask_pii", return_value=mask_pii_return_value)
    mocker.patch.object(processor.document_parser, "parse", return_value=parse_return_value)
    mocker.patch.object(processor.data_processor, "generate", return_value=generate_return_value)

    # Run pipeline
    context = await processor.anonymize(testing_settings.test_user_data_path, patient_id)
    await processor.run(context)

    # Verify database state
    await check_reports(processor.database, testing_settings)
    await check_test_observations(processor.database, testing_settings)
    await check_pipeline_run(processor.database, context.pipeline_run_id)

    # Clean database used tables and remove generated test files
    tables_to_drop = (
        Artifact.__tablename__, PipelineRun.__tablename__, PipelineStep.__tablename__, Report.__tablename__,
        TestCatalog.__tablename__, TestObservation.__tablename__
    )
    empty_test_database(*tables_to_drop)
    parsed_output_file = testing_settings.test_user_data_path.name.replace(".pdf", "_parsed.txt")
    processed_output_file = testing_settings.test_user_data_path.name.replace(".pdf", "_processed.json")
    paths_to_remove = (
        processor.path_settings.parsed_documents_dir / parsed_output_file,
        processor.path_settings.processed_documents_dir / processed_output_file,
    )
    remove_paths(paths=paths_to_remove)
