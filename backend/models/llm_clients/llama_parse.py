from pathlib import Path
from typing import Any

from llama_parse import LlamaParse
from llama_parse.base import ResultType

from backend.models.llm_clients.base import local_logger
from configs.models import LlamaParseSettings


class LlamaParseClient:

    def __init__(self, config: LlamaParseSettings):
        self.config = config

        self._normalize_parameters()
        self._initialize_client()

    def _normalize_parameters(self):
        """Normalizes the result type parameter to the expected format."""
        self.config.result_type = {
            "markdown": ResultType.MD, "text": ResultType.TXT, "json": ResultType.JSON
        }[self.config.result_type]

    def _initialize_client(self):
        """Initializes the LlamaParse client."""
        try:
            self.client = LlamaParse(
                api_key=self.config.api_key.get_secret_value(),
                result_type=self.config.result_type,
                # parsing_instruction=self.config.parsing_instruction,
                max_pages=self.config.max_pages,
                language=self.config.language,
                parse_mode=self.config.parse_mode,
                disable_image_extraction=self.config.disable_image_extraction,
                fast_mode=self.config.fast_mode,
            )
            local_logger.info("LlamaParse client initialized successfully.")
        except Exception as e:
            local_logger.error(f"Error initializing LlamaParse client: {e}")
            self.client = None

    def parse(self, file_path: Path, **kwargs) -> list[Any]:
        """Parses a document file using the LlamaParse service.

        Args:
            file_path: The local path to the document.
            **kwargs: Additional arguments to pass to the parser.

        Returns:
            A list of document objects containing the parsed content.
        """
        try:
            documents = self.client.load_data(file_path, **kwargs)
            return documents
        except Exception as e:
            local_logger.error(f"Failed to parse document {file_path}. Error: {e}")
            return []
