from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.constants import ENV_PATH
from configs.constants import ROOT_DIR


class TestingSettings(BaseSettings):
    """Tests settings configuration."""
    test_database_name: str = "test_db"
    test_user_data_path: Path = ROOT_DIR / "tests/samples/test_patient.pdf"
    test_user_data_anonymized_path: Path = ROOT_DIR / "tests/samples/test_patient_anonymized.pdf"
    test_user_data_parsed_path: Path = ROOT_DIR / "tests/samples/test_patient_parsed.txt"
    test_user_data_gemini_output_path: Path = ROOT_DIR / "tests/samples/test_patient_gemini_output.txt"
    test_user_PII_data: dict = {
        "person": ("ПОПОВА", "АННА", "ФЕДОРОВНА"),
        "organization": ("КОРОВКА-СПБ",),
        "age": ("29 лет",),
        "location": ("123456, Балтийск)", "ул. Удачная", "д. 33"),
    }
    test_user_observation_names: tuple = ("Железо", "Ферритин", "T3 свободный", "Т4 свободный", "АТ\-ТГ", "АТ\-ТПО")
    test_user_observation_values: tuple = ("18\.8", "21", "4\.35", "12\.11", "<\s*3\.*0*", "<\s*3\.*0*")
    test_user_observation_units: tuple = ("мкмоль/л", "мкг/л", "пмоль/л", "пмоль/л", "МЕ/мл", "МЕ/мл")
    test_user_observation_reference_ranges: tuple = (
        "9\s*\-\s*30\.4", "15\s*-\s*204", "3\s*-\s*5\.6", "9\s*-\s*19\.05", "<\s*4\.11", "<\s*5\.6"
    )

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="POSTGRES_",
    )
