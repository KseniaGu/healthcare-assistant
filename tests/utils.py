import re

from configs.testing import TestingSettings
from src.databases.postgres.database import PostgreSQLDB
from src.databases.postgres.models import Report, TestObservation, PipelineRun


def drop_space(text: str) -> str:
    """Drops spaces in strings with decimals (values, value ranges, etc.) to make unified format."""
    return text.replace(" ", "")


async def check_reports(database: PostgreSQLDB, testing_settings: TestingSettings):
    """Checks matching of report data saved to the database."""
    report = await database.select(Report)
    all_reports = report.all()
    assert len(all_reports) > 0
    last_report = all_reports[-1]
    results = last_report.raw_json["results"]

    mismatch_message = "Database observation %s mismatch"
    for i, result in enumerate(results):
        for key in ("name", "value", "unit", "reference_range"):
            try:
                result[key] = drop_space(result[key]) if key != "name" else result[key]
                equality = re.match(getattr(testing_settings, f"test_user_observation_{key}s")[i], result[key])
                assert equality, mismatch_message % key
            except KeyError as e:
                if "'unit'" in e:
                    equality = result["units"] == getattr(testing_settings, f"test_user_observation_{key}s")[i]
                    assert equality, mismatch_message % key


async def check_test_observations(database: PostgreSQLDB, testing_settings: TestingSettings):
    """Checks matching of report data saved to the database."""
    test_observations = await database.select(TestObservation)
    all_test_observations = test_observations.all()
    all_test_observation_names = "\n".join(test_observation.test_name for test_observation in all_test_observations)

    test_missing_message = "Database test observation is missing."
    for name in testing_settings.test_user_observation_names:
        assert re.findall(name, all_test_observation_names), test_missing_message


async def check_pipeline_run(database: PostgreSQLDB, pipeline_run_id):
    """Checks whether pipeline run was successfully saved."""
    pipeline_runs = await database.select(PipelineRun)
    all_pipeline_runs = pipeline_runs.all()
    all_pipeline_run_ids = [pipeline_run.id for pipeline_run in all_pipeline_runs]
    assert pipeline_run_id in all_pipeline_run_ids, "Pipeline run is not saved."
