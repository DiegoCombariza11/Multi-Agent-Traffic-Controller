from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class SimulationRecord:
    edge_id: str
    osmids: List[str]
    highway: Optional[str]
    name: Optional[str]
    avg_traveltime_s: Optional[float] = None
    avg_congestion_pct: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimulationRecord":
        return SimulationRecord(
            edge_id=d.get("edge_id", ""),
            osmids=d.get("osmids", []) or [],
            highway=d.get("highway"),
            name=d.get("name"),
            avg_traveltime_s=d.get("avg_traveltime_s"),
            avg_congestion_pct=d.get("avg_congestion_pct"),
            raw=d,
        )
