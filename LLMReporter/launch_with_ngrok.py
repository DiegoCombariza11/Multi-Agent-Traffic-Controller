"""Advanced launcher: inicia la API (uvicorn) + ngrok opcional + warmup Ollama.

Ahora abre DOS nuevas consolas en Windows: una para uvicorn (FastAPI) y otra para ngrok
si se habilita el túnel, manteniendo la consola principal para mensajes y control.

Uso básico:
    python launch_with_ngrok.py --port 9000
Con túnel:
    python launch_with_ngrok.py --port 9000 --tunnel
Con warmup (ejecuta una petición mínima al modelo):
    python launch_with_ngrok.py --port 9000 --tunnel --warmup "¿Estado general del tráfico?"

Variables opcionales:
    REPORTER_MODEL (default: llama3)
    REPORTER_DATA_PATH
    REPORTER_CACHE_SIZE
    NGROK_AUTHTOKEN

Requisitos:
    pip install fastapi uvicorn pyngrok
    Instalar ngrok y agregarlo al PATH.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from reporter.core.config import get_settings

APP_IMPORT = "reporter.api.app:app"  # Ruta ASGI


def launch_process(cmd: list[str], name: str) -> subprocess.Popen:
    """Lanza un proceso en una nueva consola (Windows) mostrando el comando.

    En sistemas no Windows simplemente lanza en la consola actual.
    """
    print(f"[launcher] iniciando {name}: {' '.join(cmd)}")
    creationflags = 0
    if os.name == "nt":
        # CREATE_NEW_CONSOLE disponible solo en Windows
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return subprocess.Popen(cmd, creationflags=creationflags)


def locate_python() -> str:
    """Intentar usar intérprete de venv si existe, si no usar sys.executable."""
    repo_root = Path(__file__).resolve().parent
    candidates = [".venv", "venv", "env"]
    suffix = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    for name in candidates:
        candidate = repo_root / name / suffix[0] / suffix[1]
        if candidate.exists():
            return str(candidate)
    return sys.executable


def ensure_ngrok() -> Optional[str]:
    return shutil.which("ngrok")


def start_uvicorn(port: int, reload: bool) -> subprocess.Popen:
    py = locate_python()
    cmd = [py, "-m", "uvicorn", APP_IMPORT, "--host", "0.0.0.0", "--port", str(port)]
    if reload:
        cmd.append("--reload")
    return launch_process(cmd, "uvicorn")


def start_ngrok(port: int) -> Optional[subprocess.Popen]:
    # Evitar depender de datos (get_settings) solo para obtener el token
    ngrok_bin = ensure_ngrok()
    if not ngrok_bin:
        print("[launcher] ngrok no encontrado; omitiendo túnel")
        return None
    auth_token = os.getenv("NGROK_AUTHTOKEN")
    if auth_token:
        subprocess.run([ngrok_bin, "config", "add-authtoken", auth_token], check=False)
    cmd = [ngrok_bin, "http", str(port)]
    return launch_process(cmd, "ngrok")


def warmup_model(question: str):
    try:
        settings = get_settings()
        model = settings.ollama_model if hasattr(settings, "ollama_model") else settings.model  # retrocompatibilidad
    except Exception:
        from reporter.core.config import DEFAULT_OLLAMA_MODEL  # type: ignore
        model = DEFAULT_OLLAMA_MODEL
    print(f"[warmup] Ejecutando prompt inicial contra modelo {model}")
    try:
        proc = subprocess.Popen([
            "ollama", "run", model
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_b, err_b = proc.communicate(question.encode("utf-8"), timeout=60)
        out = out_b.decode("utf-8", errors="replace") if out_b else ""
        err = err_b.decode("utf-8", errors="replace") if err_b else ""
        if err:
            print("[warmup] stderr:", err.strip())
        print("[warmup] respuesta parcial:", out.strip()[:200], "...")
    except FileNotFoundError:
        print("[warmup] Ollama no encontrado en PATH, omitiendo warmup")
    except Exception as exc:  # noqa: BLE001
        print("[warmup] Error warmup:", exc)


def parse_args():
    p = argparse.ArgumentParser(description="Inicia API + ngrok opcional + warmup de modelo")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--tunnel", action="store_true", help="Habilitar túnel ngrok")
    p.add_argument("--reload", action="store_true", help="uvicorn --reload para desarrollo")
    p.add_argument("--warmup", type=str, help="Prompt inicial para calentar el modelo Ollama")
    return p.parse_args()


def main():
    args = parse_args()
    procs: list[subprocess.Popen] = []

    uvicorn_proc = start_uvicorn(args.port, args.reload)
    procs.append(uvicorn_proc)

    if args.tunnel:
        ngrok_proc = start_ngrok(args.port)
        if ngrok_proc:
            procs.append(ngrok_proc)

    if args.warmup:
        th = threading.Thread(target=warmup_model, args=(args.warmup,), daemon=True)
        th.start()

    print("[launcher] Servidor iniciado. Ctrl+C para salir.")

    def handle_signal(signum, _frame):  # noqa: ANN001
        print(f"[launcher] Señal {signum} recibida, cerrando...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
