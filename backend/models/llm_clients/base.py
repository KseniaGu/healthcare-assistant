from abc import ABC, abstractmethod
from typing import Dict, Any
from configs.constants import DATA_PROCESSING_LOGGING
from backend.utils.common_funcs import get_logger

local_logger, _ = get_logger(DATA_PROCESSING_LOGGING)

class LLMClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(self, config: Any):
        self.config = config
        self.params = self._normalize_parameters()

    @abstractmethod
    def _normalize_parameters(self) -> Dict[str, Any]:
        """Translates normalized config params (like max_output_tokens) to vendor-specific names."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Sends the request to the vendor API and returns the text response."""
        pass
