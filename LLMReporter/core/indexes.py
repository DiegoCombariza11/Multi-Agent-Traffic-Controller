from collections import defaultdict
from typing import Dict, List, Optional
import re
from .models import SimulationRecord

_STREET_PATTERN = re.compile(r"(calle|carrera|avenida|kra|kr|cl)\s*\d+", re.IGNORECASE)

class Indexes:
    def __init__(self, data: List[SimulationRecord]):
        self.by_name: Dict[str, List[SimulationRecord]] = defaultdict(list)
        self.by_edge: Dict[str, SimulationRecord] = {}
        for rec in data:
            if rec.name:
                self.by_name[rec.name.lower()] += [rec]
            self.by_edge[rec.edge_id.lower()] = rec

    def detect_street(self, text: str) -> Optional[str]:
        m = _STREET_PATTERN.search(text.lower())
        return m.group(0) if m else None

    def approximate(self, query: str) -> List[SimulationRecord]:
        q = query.lower()
        out: List[SimulationRecord] = []
        for name, items in self.by_name.items():
            if q in name:
                out.extend(items)
        return out
