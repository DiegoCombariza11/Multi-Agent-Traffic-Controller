import json
from pathlib import Path
from typing import List
from reporter.core.models import SimulationRecord

def load_data(path: Path) -> List[SimulationRecord]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [SimulationRecord.from_dict(r) for r in raw]
