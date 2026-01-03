import json
import re

import jsonschema
import pytest
from jsonschema import validate, ValidationError

from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.common_funcs import read_file
from backend.utils.helpers import clean_json_markdown
from backend.utils.prompt_engine import PromptEngine
from configs.data import DataSettings
from configs.enums import PromptTypeEnum
from configs.models import ModelSettings
from configs.paths import PathSettings
from configs.testing import TestingSettings
from tests.utils import drop_space

testing_settings = TestingSettings()
path_settings = PathSettings()
data_settings = DataSettings()
model_settings = ModelSettings()


@pytest.fixture(scope="module")
def gemini_model():
    return GeminiClient(model_settings.data_processor, path_settings)


@pytest.fixture()
def prompt_metadata():
    input_text = read_file(testing_settings.test_user_data_parsed_path)
    prompt_engine = PromptEngine(path_settings.prompts_dir)
    prompt_version = data_settings.prompt_version
    prompt_type = PromptTypeEnum.laboratory_test_processor
    prompt, system_prompt, _ = prompt_engine.render(
        PromptTypeEnum.laboratory_test_processor, prompt_version, input_text=input_text
    )
    return prompt, system_prompt, prompt_version, prompt_type


def test_document_processing(gemini_model, prompt_metadata):
    prompt, system_prompt, prompt_version, prompt_type = prompt_metadata
    model_output, usage_metadata, raw_output_path = gemini_model.generate(
        prompt, prompt_type, prompt_version, **system_prompt, model_name=model_settings.data_processor.model_name
    )

    assert isinstance(model_output, str), "Gemini client produced output of the wrong type (str excepted)"
    try:
        structured_output = json.loads(clean_json_markdown(model_output))
    except json.JSONDecodeError as e:
        pytest.fail(f"Gemini produced invalid JSON syntax: {e}\nOutput was: {model_output}")

    structured_output_to_validate = structured_output
    if isinstance(structured_output, list):
        structured_output_to_validate = {"date": None, "results": structured_output}

    try:
        validate(
            instance=structured_output_to_validate, schema=system_prompt["response_json_schema"],
            format_checker=jsonschema.FormatChecker()
        )
    except ValidationError as e:
        pytest.fail(f"Gemini's produced output doesn't match the schema: {e}")

    mismatch_message = f"Expected first test to be '%s', got '%s'"
    for key in ("name", "value", "reference_range"):
        first_output = structured_output[0][key]
        first_output = drop_space(first_output) if key != "name" else first_output
        expected_result = getattr(testing_settings, f"test_user_observation_{key}s")[0]
        assert re.match(expected_result, first_output), mismatch_message % (expected_result, first_output)

    # Rename output path to mark the file as testing
    new_raw_output_path = raw_output_path.with_name(raw_output_path.stem + "_test" + raw_output_path.suffix)
    raw_output_path.rename(new_raw_output_path)
