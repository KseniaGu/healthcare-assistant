import os
from pathlib import Path

import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
from google.genai.types import GenerateContentResponse

from backend.models.llm_clients.base import LLMClient, local_logger
from backend.utils.common_funcs import write_file
from backend.utils.exceptions import BaseLLMError
from configs.enums import PromptTypeEnum
from configs.models import GoogleGeminiSettings
from configs.paths import PathSettings


class GeminiClient(LLMClient):

    def __init__(self, config: GoogleGeminiSettings, path_settings: PathSettings):
        super().__init__(config)
        self.path_settings = path_settings
        self.client = genai.Client(api_key=config.api_key.get_secret_value())

    def _normalize_parameters(self) -> types.GenerateContentConfig:
        """Translates pydantic config fields into a native google-genai configuration object."""
        tools = []
        if self.config.use_grounding:
            tools.append(types.Tool(google_search={}))

        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            tools=tools,
        )

        return config

    def _get_response_content(self, response: GenerateContentResponse) -> str | dict:
        """Extracts model generation content from the response object.

        Args:
            response: The raw response from API.

        Returns:
            The processed response.
        """
        if self.params.tools:
            processed_response = {"content": None, "tool_calls": [], }

            # Extract content from the first candidate
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        processed_response["content"] = part.text
                        break

            # Extract function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fn = part.function_call
                        processed_response["tool_calls"].append(
                            {
                                "name": fn.name,
                                "arguments": dict(fn.args) if fn.args else {},
                            }
                        )

            return processed_response
        else:
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
            return ""

    @staticmethod
    def _get_response_usage_metadata(response: GenerateContentResponse) -> dict:
        """Gets usage metadata from response object."""
        usage_metadata = {}
        if response.usage_metadata:
            fields = ("candidates_token_count", "prompt_token_count", "thoughts_token_count", "total_token_count")
            for field in fields:
                if hasattr(response.usage_metadata, field):
                    usage_metadata[field] = getattr(response.usage_metadata, field)

        return usage_metadata

    def save_output(self, output: str, prompt_type: PromptTypeEnum, prompt_version: str) -> Path:
        prompt_llm_outputs_dir = self.path_settings.llm_outputs_dir / str(prompt_type.value) / prompt_version
        if os.path.exists(prompt_llm_outputs_dir):
            current_prompt_outputs = os.listdir(prompt_llm_outputs_dir)
            output_path = prompt_llm_outputs_dir / f"{len(current_prompt_outputs)}.txt"
            write_file(output, output_path)
        else:
            os.makedirs(prompt_llm_outputs_dir)
            output_path = prompt_llm_outputs_dir / "0.txt"
            write_file(output, output_path)
        return output_path

    def generate(
            self,
            prompt: str,
            prompt_type: PromptTypeEnum = PromptTypeEnum.laboratory_test_processor,
            prompt_version: str = "v1",
            system_instruction: str = None,
            response_json_schema: dict = None,
            model_name: str = None
    ) -> (str, Path):
        """Executes the Gemini API call."""
        model_name = model_name if model_name is not None else self.config.model_name
        if system_instruction is not None:
            self.params.system_instruction = system_instruction
        if response_json_schema is not None:
            self.params.response_json_schema = response_json_schema
        try:
            assert prompt is not None and prompt != ""
            response = self.client.models.generate_content(model=model_name, contents=[prompt], config=self.params)
            parsed_response = self._get_response_content(response)
            usage_metadata = self._get_response_usage_metadata(response)
            raw_output_path = self.save_output(parsed_response, prompt_type, prompt_version)

            return parsed_response, usage_metadata, raw_output_path

        except AssertionError:
            message = f"Failed to generate content via Gemini client: prompt is empty"
            local_logger.error(message)
            raise BaseLLMError(message)
        except APIError as e:
            message = f"Failed to generate content via Gemini client. API error: {e}"
            local_logger.error(message)
            raise BaseLLMError(message)
        except Exception as e:
            message = f"An unexpected error occurred while generating content via Gemini client: {e}"
            local_logger.error(message)
            raise BaseLLMError(message)
