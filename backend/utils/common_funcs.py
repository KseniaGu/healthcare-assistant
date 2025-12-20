import json
import logging
import os
import pickle
import tempfile
from logging.handlers import BufferingHandler
from pathlib import Path
from typing import Any

import yaml

from backend.utils.exceptions import DocumentParsingError


def write_file(file: Any, path: Path):
    """Writes data to a file with the specified format.
    
    Args:
        file: The data to be written to the file.
        path: The path where the file should be written.
    """
    extension = os.path.splitext(path)[1]
    if extension == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(file, f)
    elif extension == ".txt":
        with open(path, "w") as f:
            f.write(file)
    elif extension == ".json":
        with open(path, "w") as f:
            json.dump(file, f)
    else:
        raise ValueError(f"Unsupported extension: {extension} (.pickle, .txt, .json are only available for now)")


def write_temp_file(file: Any, file_suffix: str = ".pdf", file_prefix: str = "temp_file_") -> Path:
    """Writes data to a temporary file.
    
    Args:
        file: The data to be written to the temporary file.
        file_suffix: The suffix for the temporary file.
        file_prefix: The prefix for the temporary file.
        
    Returns:
        The path to the created temporary file.
    """
    try:
        if isinstance(file, (bytes, bytearray)):
            fd, path = tempfile.mkstemp(suffix=file_suffix, prefix=file_prefix)
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(file)
            return Path(path)
        else:
            raise DocumentParsingError(f"Failed to write temporary file: unsupported file type ({type(file)})")
    except OSError as e:
        raise DocumentParsingError(f"Error while attempting to save temporary document file: {str(e)}")


def read_file(path: Path) -> Any:
    """Reads data from a file with the specified format.
    
    Args:
        path: The path to the file to be read.
        
    Returns:
        The data read from the file.
    """
    extension = os.path.splitext(path)[1]
    try:
        if extension == ".pickle":
            with open(path, "rb") as f:
                file = pickle.load(f)
        elif extension == ".yaml":
            with open(path, "r") as f:
                file = yaml.safe_load(f)
        elif extension == ".txt":
            file = open(path, "r").read()
        elif extension == ".json":
            with open(path, "r") as f:
                file = json.load(f)
        else:
            print(f"Unsupported extension: {extension} (.pickle, .yaml, .txt, .json are only available)")
            return None
    except FileNotFoundError:
        print(f"File {path} not found")
        return None
    return file


# Logging helpers
class MemoryLogHandler(BufferingHandler):
    """A logging handler that stores log records in memory.
    
    This handler extends BufferingHandler to store log records in memory
        and provides a method to retrieve the formatted log messages.
    """
    def __init__(self, capacity: int):
        """Initializes the handler with a buffer of the specified capacity.
        
        Args:
            capacity: Maximum number of log records to store.
        """
        super().__init__(capacity)

    def get_logs(self) -> list[str]:
        """Gets all log messages currently in the buffer.
        
        Returns:
            A list of formatted log messages.
        """
        return [self.format(record) for record in self.buffer]


def get_logger(name: str = __name__) -> tuple[logging.Logger, MemoryLogHandler]:
    """Gets a configured logger instance with a memory handler.
    
    Args:
        name: The name of the logger. Defaults to the module name.
    """
    logger = logging.getLogger(name)
    base_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=base_format, handlers=[logging.StreamHandler()])

    memory_handler = MemoryLogHandler(100)
    memory_handler.setFormatter(logging.Formatter(base_format))
    logger.addHandler(memory_handler)
    return logger, memory_handler
