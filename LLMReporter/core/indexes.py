from collections import defaultdict
from typing import Dict, List, Optional
import re
from .models import SimulationRecord, AgentRecord

_STREET_PATTERN = re.compile(r"(calle|carrera|avenida|kra|kr|cl)\s*\d+", re.IGNORECASE)
_AGENT_PATTERN = re.compile(r"(first|second|third|fourth|fifth)[_\s-]?agent", re.IGNORECASE)

class Indexes:
    def __init__(self, data: List[SimulationRecord], agents: Optional[List[AgentRecord]] = None):
        self.by_name: Dict[str, List[SimulationRecord]] = defaultdict(list)
        self.by_edge: Dict[str, SimulationRecord] = {}
        self.by_region: Dict[str, AgentRecord] = {}
        
        for rec in data:
            if rec.name:
                self.by_name[rec.name.lower()] += [rec]
            self.by_edge[rec.edge_id.lower()] = rec
        
        if agents:
            for agent in agents:
                self.by_region[agent.region_id.lower()] = agent

    def detect_street(self, text: str) -> Optional[str]:
        m = _STREET_PATTERN.search(text.lower())
        return m.group(0) if m else None
    
    def detect_agent(self, text: str) -> Optional[str]:
        """Detect agent mentions in query (e.g., 'First_Agent', 'second agent')."""
        m = _AGENT_PATTERN.search(text.lower())
        if m:
            # Normalize to standard format: "First_Agent", "Second_Agent", etc.
            agent_name = m.group(1).capitalize() + "_Agent"
            return agent_name
        return None
    
    def get_agent(self, region_id: str) -> Optional[AgentRecord]:
        """Get agent record by region ID."""
        return self.by_region.get(region_id.lower())
    
    def get_all_agents(self) -> List[AgentRecord]:
        """Get all agent records."""
        return list(self.by_region.values())

    def approximate(self, query: str) -> List[SimulationRecord]:
        q = query.lower()
        out: List[SimulationRecord] = []
        for name, items in self.by_name.items():
            if q in name:
                out.extend(items)
        return out
