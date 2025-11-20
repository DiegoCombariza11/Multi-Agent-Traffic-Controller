from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Optional
import os

@dataclass(frozen=True)
class Settings:
    data_path: Path
    ollama_model: str
    ollama_base_url: str
    cache_size: int
    port: int
    suggestion_timeout: float
    ngrok_authtoken: Optional[str]

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("REPORTER_MODEL", "llama3:8b"))
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_CACHE_SIZE = int(os.getenv("REPORTER_CACHE_SIZE", "512"))
DEFAULT_PORT = int(os.getenv("REPORTER_PORT", "9000"))
DEFAULT_TIMEOUT = float(os.getenv("SUGGESTION_TIMEOUT", "30"))

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return immutable settings with precedence:
    1. REPORTER_DATA_PATH env var
    2. resumen_simulacion.json (si existe)
    3. edge_summary.json (si existe)
    If none exist, raise a clear error to avoid silent fallbacks.
    """
    env_path = os.getenv("REPORTER_DATA_PATH")
    if env_path:
        data_path = Path(env_path)
    else:
        candidates = [
            Path("resumen_simulacion.json"),
            Path("edge_summary.json"),
            Path("reporter/core/resumen_simulacion.json"),
            Path("reporter/core/edge_summary.json"),
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if not found:
            raise FileNotFoundError(
                "No se encontró archivo de datos. Coloca 'resumen_simulacion.json' o 'edge_summary.json' o define REPORTER_DATA_PATH"
            )
        data_path = found
    return Settings(
        data_path=data_path,
        ollama_model=DEFAULT_OLLAMA_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        cache_size=DEFAULT_CACHE_SIZE,
        port=DEFAULT_PORT,
        suggestion_timeout=DEFAULT_TIMEOUT,
        ngrok_authtoken=os.getenv("NGROK_AUTHTOKEN"),
    )
