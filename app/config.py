import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MemAgent 统一配置管理类"""
    
    # LLM Settings
    LLM_API_KEY: str = "your_api_key_here"
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # PostgreSQL / pgvector Settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "memagent"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "shibing624/text2vec-base-chinese"
    EMBEDDING_DIMENSION: int = 768

    # Memory Architecture Settings
    WORKING_MEMORY_WINDOW_SIZE: int = 10
    WORKING_MEMORY_TTL_SECONDS: int = 86400  # 1天
    EPISODIC_SUMMARY_INTERVAL: int = 5       # 每5轮总结一次
    FORGETTING_THRESHOLD: float = 0.3
    TIME_DECAY_LAMBDA: float = 0.01

    @property
    def postgres_async_url(self) -> str:
        """获取 asyncpg 连接字符串"""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn(self) -> str:
        """获取标准 psycopg2/asyncpg DSN"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
