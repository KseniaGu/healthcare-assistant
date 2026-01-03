from pydantic_settings import BaseSettings


class DataSettings(BaseSettings):
    """Data settings configuration."""
    # PII extraction
    pii_entities: tuple = (
        "person", "phone number", "email address", "date of birth", "age", "gender", "organization", "location"
    )
    pii_prediction_threshold: float = 0.4

    # Data processing
    prompt_version: str = "v3"

    # Text validation
    common_words: set = {
        'и', 'в', 'не', 'на', 'я', 'быть', 'с', 'он', 'что', 'а', 'это', 'как', 'по', 'но', 'они', 'к', 'у', 'ты',
        'из', 'мы'
    }
    common_inequalities: tuple = ("<", ">", "≤", "≥", "~", "≈")
