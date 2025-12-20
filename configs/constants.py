from pathlib import Path

# Base
ROOT_DIR: Path = Path(__file__).parent.parent.resolve()
DATA_DIR: Path = ROOT_DIR / "data"
ENV_PATH: Path = ROOT_DIR / ".env"
DATA_PROCESSING_LOGGING = 'Laboratory results processing'

DATA_DIR.mkdir(exist_ok=True)

# PII extraction using GLiNER models
GLINER_MAX_WINDOW_SIZE = 300
GLINER_TOKENS_OVERLAP = 50

# Test observations
DEFAULT_LAB_TEST_NAME = "Unknown test name"
