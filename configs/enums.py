from enum import IntEnum, StrEnum


class StepTypeEnum(IntEnum):
    """A class representing different stages in the document processing pipeline."""
    other = 0
    anonymize = 1
    parse = 2
    process = 3
    postprocess = 4


class TestFlagTypeEnum(StrEnum):
    """A class indicating whether a test result is below, above or within the reference range."""
    low = "low"
    high = "high"
    normal = "normal"


class ToolNameEnum(StrEnum):
    """A class representing different tools and services integrated into the system."""
    gemini = "Gemini"
    gliner = "GLiNER"
    llama_parse = "LLamaParse"
    paddleocr = "PaddleOCR"


class ProviderNameEnum(StrEnum):
    """A class representing different service providers for various AI/ML services."""
    local = "local"
    self_hosted = "self-hosted"
    openai = "OpenAI"
    google = "Google"
    hugging_face = "Hugging Face"


class PromptTypeEnum(StrEnum):
    """A class representing different types of prompts used with language models."""
    laboratory_test_processor = "laboratory_test_processor"


class TaskEnum(IntEnum):
    """A class representing different processing tasks that can be performed."""
    laboratory_results = 1


class PipelineStatusEnum(IntEnum):
    """A class representing the possible states of a processing pipeline."""
    draft = 0
    active = 1
    deprecated = 2
