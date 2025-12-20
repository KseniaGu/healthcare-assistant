from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.constants import ENV_PATH
from configs.enums import ToolNameEnum, ProviderNameEnum


class BaseLLMSettings(BaseSettings):
    """Base settings for language model configurations.
    
    Provides common configuration options for all language model services.
    """
    model_name: str
    tool_name: ToolNameEnum
    provider: ProviderNameEnum
    temperature: float = 0.
    max_tokens: int = 4096

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def model_config_with_prefix(cls, prefix: str) -> SettingsConfigDict:
        return SettingsConfigDict(**cls.model_config) | {'env_prefix': prefix}


class GoogleGeminiSettings(BaseLLMSettings):
    """Configuration settings for Google's Gemini models.
    
    Extends base LLM settings with Gemini-specific configurations.
    """
    model_config = BaseLLMSettings.model_config_with_prefix("GEMINI_")
    tool_name: ToolNameEnum = ToolNameEnum.gemini
    provider: ProviderNameEnum = ProviderNameEnum.google
    api_key: SecretStr | None = None
    use_grounding: bool = False
    response_mime_type: str = "application/json"


# TODO: Test OpenAI models
class OpenAISettings(BaseLLMSettings):
    """Configuration settings for OpenAI models.
    
    Extends base LLM settings with OpenAI-specific configurations.
    """
    model_config = BaseLLMSettings.model_config_with_prefix("OPENAI_")
    api_key: SecretStr | None = None


class LlamaParseSettings(BaseSettings):
    """Configuration settings for LlamaParse document processing.
    
    Contains settings for document parsing and processing using LlamaParse.
    """
    model_config = BaseLLMSettings.model_config_with_prefix("LLAMAPARSE_")
    tool_name: ToolNameEnum = ToolNameEnum.llama_parse
    provider: ProviderNameEnum = ProviderNameEnum.self_hosted
    api_key: SecretStr | None = None
    result_type: str = "text"  # One of ("markdown", "text", "json") supported
    max_pages: int | None = None
    language: str | None = "ru"
    parse_mode: str | None = "parse_page_without_llm"
    disable_ocr: bool = True
    disable_image_extraction: bool = True
    fast_mode: bool | None = None  # True here is equivalent to disable_ocr=True and disable_image_extraction=True


# Settings for the models that run locally (downloaded from huggingface hub etc.)
class LocalModelSettings(BaseSettings):
    """Configuration for locally hosted models.
    
    Contains settings for models that are run locally rather than through an API.
    """
    tool_name: ToolNameEnum
    provider: ProviderNameEnum = ProviderNameEnum.local
    model_path: str = "urchade/gliner_multi-v2.1"
    tokenizer_path: str | None = "microsoft/deberta-v3-large"
    device_map: str | None = "auto"


class ModelSettings(BaseSettings):
    """Aggregates model configuration settings.
    
    Centralized configuration for all model-related settings.
    """
    pii_extractor: LocalModelSettings = LocalModelSettings(
        tool_name=ToolNameEnum.gliner, model_path="urchade/gliner_multi-v2.1",
        tokenizer_path="microsoft/deberta-v3-large"
    )
    document_parser: LlamaParseSettings = LlamaParseSettings(parse_mode="parse_page_without_llm")
    data_processor: GoogleGeminiSettings = GoogleGeminiSettings(model_name="gemini-2.5-flash-lite")
