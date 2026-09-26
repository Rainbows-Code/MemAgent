from app.config import settings


def test_settings_load():
    """测试配置文件正确加载且具备默认值"""
    assert settings.WORKING_MEMORY_WINDOW_SIZE == 10
    assert settings.EPISODIC_SUMMARY_INTERVAL == 5
    assert settings.POSTGRES_DB == "memagent"
    assert "postgresql://" in settings.postgres_dsn
    assert "postgresql+asyncpg://" in settings.postgres_async_url
