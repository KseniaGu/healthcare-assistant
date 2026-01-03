import re

import pytest

from backend.models.llm_clients.llama_parse import LlamaParseClient
from configs.models import ModelSettings
from configs.testing import TestingSettings

testing_settings = TestingSettings()


@pytest.fixture(scope="module")
def llama_parse_model():
    model_settings = ModelSettings()
    return LlamaParseClient(model_settings.document_parser)


def test_document_parsing(llama_parse_model):
    text_content = llama_parse_model.parse(testing_settings.test_user_data_anonymized_path)
    assert text_content, "LlamaParse returned no text content."

    # Check for specific expected observations data
    missing_names = []
    for name in testing_settings.test_user_observation_names:
        if not re.findall(name, text_content):
            missing_names.append(name)

    missing_values = []
    for value in testing_settings.test_user_observation_values:
        if not re.findall(value, text_content):
            missing_values.append(value)

    assert not missing_names, f"Parser failed to extract the following observation names: {missing_names}"
    assert not missing_values, f"Parser failed to extract the following observation values: {missing_values}"
