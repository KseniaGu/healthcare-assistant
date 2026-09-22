import re

import pytest

from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.common_funcs import read_file
from backend.utils.extraction import parse_and_validate_extraction
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
    structured_output = parse_and_validate_extraction(model_output, system_prompt["response_json_schema"])
    results = structured_output["results"]
    assert len(results) == len(testing_settings.test_user_observation_names)
    for index, result in enumerate(results):
        for key in ("name", "value", "unit", "reference_range"):
            actual = str(result[key])
            actual = drop_space(actual) if key != "name" else actual
            expected = getattr(testing_settings, f"test_user_observation_{key}s")[index]
            assert re.fullmatch(expected, actual), f"Row {index}, {key}: expected {expected!r}, got {actual!r}"

    # Rename output path to mark the file as testing
    new_raw_output_path = raw_output_path.with_name(raw_output_path.stem + "_test" + raw_output_path.suffix)
    raw_output_path.rename(new_raw_output_path)
