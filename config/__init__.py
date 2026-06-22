from functools import lru_cache
from .settings import Settings

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

__all__ = ["Settings", "settings", "get_settings"]
