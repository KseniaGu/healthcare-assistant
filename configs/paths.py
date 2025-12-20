from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

from configs.constants import DATA_DIR, ROOT_DIR


class PathSettings(BaseSettings):
    """Configuration for file system paths used throughout the application.
    
    Manages all directory paths for storing and organizing application data.
    """
    RAW_DOCUMENTS: Path = DATA_DIR / "raw_documents"
    ANONYMIZED_DOCUMENTS_DIR: Path = DATA_DIR / "anonymized"
    PARSED_DOCUMENTS_DIR: Path = DATA_DIR / "parsed"
    PROCESSED_DOCUMENTS_DIR: Path = DATA_DIR / "processed"
    PROMPTS_DIR: Path = ROOT_DIR / "backend/prompts"
    LLM_OUTPUTS_DIR: Path = DATA_DIR / "llm_outputs"

    @model_validator(mode='after')
    def ensure_directory_existence(self):
        """Ensures all required directories exist.
        
        Creates any missing directories in the required paths if they don't exist.
        """
        required_dirs = [
            self.RAW_DOCUMENTS, self.ANONYMIZED_DOCUMENTS_DIR, self.PARSED_DOCUMENTS_DIR, self.PROCESSED_DOCUMENTS_DIR,
            self.LLM_OUTPUTS_DIR
        ]

        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        return self
