from typing import Any, Dict, List
from core.indexes import Indexes
from core.models import SimulationRecord, AgentRecord
from utils.prompt import build_prompt

class TrafficAnalyzer:
    def __init__(self, indexes: Indexes, full_data: List[SimulationRecord], agents_data: Dict[str, Any] | None = None):
        self.indexes = indexes
        self.full_data = full_data
        self.agents_data = agents_data or {}

    def select_records(self, question: str) -> List[SimulationRecord]:
        street = self.indexes.detect_street(question)
        if street:
            hits = self.indexes.approximate(street)
            if hits:
                return hits
        approx = self.indexes.approximate(question)
        if approx:
            return approx
        return self.full_data
    
    def select_agents(self, question: str) -> List[AgentRecord]:
        """Select relevant agent records based on question.
        
        Returns agents ONLY if explicitly asked about them.
        """
        agent_id = self.indexes.detect_agent(question)
        if agent_id:
            agent = self.indexes.get_agent(agent_id)
            return [agent] if agent else []
        
        # Only include agents if question explicitly mentions them
        q_lower = question.lower()
        agent_keywords = [
            "agente", "agent", "regional", 
            "intervención", "intervention", "intervenciones",
            "first_agent", "second_agent", "third_agent", "fourth_agent", "fifth_agent"
        ]
        if any(keyword in q_lower for keyword in agent_keywords):
            return self.indexes.get_all_agents()
        
        # Default: no agents data in prompt
        return []

    def analyze(self, question: str, llm_client) -> str:
        records = self.select_records(question)
        agents = self.select_agents(question)
        prompt = build_prompt(question, records, self.agents_data, agents)
        return llm_client.ask(prompt)
