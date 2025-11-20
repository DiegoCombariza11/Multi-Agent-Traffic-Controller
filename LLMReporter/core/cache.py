from collections import OrderedDict
from typing import Optional

class AnswerCache:
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def _normalize(key: str) -> str:
        return " ".join(key.lower().strip().split())

    def get(self, key: str) -> Optional[str]:
        nk = self._normalize(key)
        val = self._store.get(nk)
        if val is not None:
            self._store.move_to_end(nk)
        return val

    def set(self, key: str, value: str):
        nk = self._normalize(key)
        if nk in self._store:
            self._store.move_to_end(nk)
        self._store[nk] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def stats(self) -> dict:
        return {"size": len(self._store), "max_size": self.max_size}
