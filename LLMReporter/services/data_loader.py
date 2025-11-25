import json
from pathlib import Path
from typing import Any, Dict, List
from core.models import SimulationRecord, AgentRecord

def load_data(path: Path) -> List[SimulationRecord]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [SimulationRecord.from_dict(r) for r in raw]


def load_agents_summary(path: Path) -> Dict[str, Any]:
    """Load regional agents intervention summary (raw dict)."""
    if not path.exists():
        return {"total_regions": 0, "regions": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_agents(path: Path) -> List[AgentRecord]:
    """Load regional agents as structured records."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [AgentRecord.from_dict(r) for r in data.get("regions", [])]
