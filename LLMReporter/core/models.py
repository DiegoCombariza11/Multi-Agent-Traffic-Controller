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


@dataclass
class AgentRecord:
    """Regional agent intervention record."""
    region_id: str
    total_interventions: int
    avg_queue_at_intervention: float
    intervention_periods: List[Dict[str, Any]]
    traffic_lights: List[str]
    raw: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentRecord":
        return AgentRecord(
            region_id=d.get("region_id", ""),
            total_interventions=d.get("total_interventions", 0),
            avg_queue_at_intervention=d.get("avg_queue_at_intervention", 0.0),
            intervention_periods=d.get("intervention_periods", []),
            traffic_lights=d.get("traffic_lights", []),
            raw=d,
        )
