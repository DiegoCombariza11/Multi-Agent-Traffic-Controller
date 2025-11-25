"""CLI local para hacer preguntas sobre el tráfico sin iniciar FastAPI ni ngrok.

Uso:
    python reporter_cli.py "¿Estado general del tráfico?"
Modo interactivo:
    python reporter_cli.py
    > (escribe tu pregunta y Enter, Ctrl+C para salir)

Variables de entorno útiles:
    REPORTER_DATA_PATH   Ruta al JSON de datos (si no usa nombres por defecto)
    OLLAMA_MODEL / REPORTER_MODEL  Modelo Ollama (default: llama3:8b)
    REPORTER_CACHE_SIZE  Tamaño del caché LRU (default 512)

Si no se encuentra un archivo de datos compatible se mostrará un error claro.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

from core.config import get_settings, DEFAULT_OLLAMA_MODEL
from services.data_loader import load_data, load_agents_summary, load_agents
from core.indexes import Indexes
from core.cache import AnswerCache
from services.llm_client import LLMClient
from services.analyzer import TrafficAnalyzer


def init_components():
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[error] Configuración de datos no encontrada: {exc}")
    data = load_data(settings.data_path)
    
    # Load agents summary if available
    agents_path = settings.data_path.parent / "agents_summary.json"
    agents_data = load_agents_summary(agents_path)
    agents_records = load_agents(agents_path)
    
    indexes = Indexes(data, agents_records)
    cache = AnswerCache(settings.cache_size)
    llm = LLMClient(settings.ollama_model)
    analyzer = TrafficAnalyzer(indexes, data, agents_data)
    return settings, data, indexes, cache, llm, analyzer


def ask_once(question: str, cache: AnswerCache, analyzer: TrafficAnalyzer, llm: LLMClient):
    cached = cache.get(question)
    if cached is not None:
        return {"question": question, "answer": cached, "cached": True}
    answer = analyzer.analyze(question, llm)
    cache.set(question, answer)
    return {"question": question, "answer": answer, "cached": False}


def interactive(cache: AnswerCache, analyzer: TrafficAnalyzer, llm: LLMClient):
    print("[cli] Interactive Mode. Ctrl+C to quit.")
    while True:
        try:
            q = input("› Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[cli] Quitting...")
            break
        if not q:
            continue
        result = ask_once(q, cache, analyzer, llm)
        print("[answer]", result["answer"])


def main(argv: list[str]):
    settings, _data, _indexes, cache, llm, analyzer = init_components()
    if len(argv) > 1:
        question = " ".join(argv[1:]).strip()
        out = ask_once(question, cache, analyzer, llm)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        interactive(cache, analyzer, llm)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
