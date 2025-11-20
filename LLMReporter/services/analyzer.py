from typing import List
from reporter.core.indexes import Indexes
from reporter.core.models import SimulationRecord
from reporter.utils.prompt import build_prompt

class TrafficAnalyzer:
    def __init__(self, indexes: Indexes, full_data: List[SimulationRecord]):
        self.indexes = indexes
        self.full_data = full_data

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

    def analyze(self, question: str, llm_client) -> str:
        records = self.select_records(question)
        prompt = build_prompt(question, records)
        return llm_client.ask(prompt)
