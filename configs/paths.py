from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

from configs.constants import DATA_DIR, ROOT_DIR


class PathSettings(BaseSettings):
    """Configuration for file system paths used throughout the application.
    
    Manages all directory paths for storing and organizing application data.
    """
    raw_documents_dir: Path = DATA_DIR / "raw_documents"
    anonymized_documents_dir: Path = DATA_DIR / "anonymized"
    parsed_documents_dir: Path = DATA_DIR / "parsed"
    processed_documents_dir: Path = DATA_DIR / "processed"
    prompts_dir: Path = ROOT_DIR / "backend/prompts"
    llm_outputs_dir: Path = DATA_DIR / "llm_outputs"

    @model_validator(mode='after')
    def ensure_directory_existence(self):
        """Ensures all required directories exist.
        
        Creates any missing directories in the required paths if they don't exist.
        """
        required_dirs = [
            self.raw_documents_dir, self.anonymized_documents_dir, self.parsed_documents_dir,
            self.processed_documents_dir, self.llm_outputs_dir
        ]

        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        return self
