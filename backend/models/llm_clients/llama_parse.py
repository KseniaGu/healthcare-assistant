from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from llama_parse import LlamaParse

from backend.models.llm_clients.base import local_logger
from backend.utils.exceptions import DocumentParsingError

if TYPE_CHECKING:
    from configs.models import LlamaParseSettings


class LlamaParseClient:
    def __init__(self, config: LlamaParseSettings):
        self.config = config

    async def _parse(self, file_path: Path, **kwargs) -> str:
        # Streamlit runs each request on a new loop; HTTP connections must not outlive it.
        client = LlamaParse(
            api_key=self.config.api_key.get_secret_value(),
            result_type=self.config.result_type,
            max_pages=self.config.max_pages,
            language=self.config.language,
            parse_mode=self.config.parse_mode,
            disable_image_extraction=self.config.disable_image_extraction,
            fast_mode=self.config.fast_mode,
            ignore_errors=False,
        )
        async with client.aclient:
            documents = await client.aload_data(file_path, **kwargs)
            return "\n".join(doc.text for doc in documents)

    def parse(self, file_path: Path, **kwargs) -> str:
        """Parse a document, closing its HTTP client before the worker loop ends."""
        try:
            return asyncio.run(self._parse(file_path, **kwargs))
        except Exception as e:
            local_logger.exception("Document parsing failed")
            raise DocumentParsingError(f"Failed to parse document {file_path}: {e}") from e
