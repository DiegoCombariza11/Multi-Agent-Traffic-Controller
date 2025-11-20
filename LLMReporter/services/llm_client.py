import subprocess
from typing import Optional

class LLMClient:
    def __init__(self, model: str):
        self.model = model

    def ask(self, prompt: str, model: Optional[str] = None) -> str:
        m = model or self.model
        # Usamos binario y decodificamos manualmente para evitar UnicodeDecodeError en cp1252
        proc = subprocess.Popen([
            "ollama", "run", m
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_bytes, _err_bytes = proc.communicate(prompt.encode("utf-8"))
        try:
            out = out_bytes.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            out = out_bytes.decode(errors="ignore")
        return out
