"""Utilidades para construir el prompt enviado al modelo LLM.

La idea es condensar registros relevantes y la pregunta del usuario
con un formato estable y fácil de extender.
"""
from __future__ import annotations

from typing import List
from reporter.core.models import SimulationRecord

MAX_RECORDS = 12  # límite para no generar prompts enormes


def format_record(r: SimulationRecord) -> str:
    name = r.name or r.edge_id
    t = r.avg_traveltime_s if r.avg_traveltime_s is not None else 0.0
    c = r.avg_congestion_pct if r.avg_congestion_pct is not None else 0.0
    return f"via={name} tmedio={t:.2f}s congestion={c:.2f}%"


def build_prompt(question: str, records: List[SimulationRecord]) -> str:
    subset = records[:MAX_RECORDS]
    lines = "\n".join(format_record(r) for r in subset)
    return (
        "Analiza los siguientes datos de tráfico y responde a la pregunta en español.\n"  # instrucción breve
        f"Pregunta: {question}\n"  # pregunta del usuario
        f"Registros ({len(subset)}/{len(records)}):\n{lines}\n"  # lista formateada
        "Si la pregunta implica comparación, menciona las vías con mayor y menor congestión."
    )
