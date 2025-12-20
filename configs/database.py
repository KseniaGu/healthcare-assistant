from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.constants import ENV_PATH


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL database settings configuration."""
    username: SecretStr | None = None
    password: SecretStr | None = None
    database_name: str | None = None
    auto_migrate: bool | None = None
    host: str = "localhost"
    port: str = "5432"

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="POSTGRES_",
    )

    def get_connection_url(self, driver: str = "psycopg") -> str:
        username = self.username.get_secret_value()
        password = self.password.get_secret_value()
        driver_name = f"postgresql+{driver}" if driver else "postgresql"
        return f"{driver_name}://{username}:{password}@{self.host}:{self.port}/{self.database_name}"
