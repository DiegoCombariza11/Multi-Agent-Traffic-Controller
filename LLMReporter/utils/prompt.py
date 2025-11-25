"""Utilidades para construir el prompt enviado al modelo LLM.

La idea es condensar registros relevantes y la pregunta del usuario
con un formato estable y fácil de extender.
"""
from __future__ import annotations

from typing import Any, Dict, List
from core.models import SimulationRecord, AgentRecord

MAX_RECORDS = 12  # límite para no generar prompts enormes


def format_record(r: SimulationRecord) -> str:
    name = r.name or r.edge_id
    t = r.avg_traveltime_s if r.avg_traveltime_s is not None else 0.0
    c = r.avg_congestion_pct if r.avg_congestion_pct is not None else 0.0
    return f"via={name} tmedio={t:.2f}s congestion={c:.2f}%"


def format_agent_record(agent: AgentRecord) -> str:
    """Format a single agent record for prompt."""
    periods_summary = ""
    if agent.intervention_periods:
        periods_summary = f", {len(agent.intervention_periods)} periodos"
    
    tls = len(agent.traffic_lights)
    return (
        f"  - {agent.region_id}: {agent.total_interventions} intervenciones, "
        f"cola promedio={agent.avg_queue_at_intervention:.1f}s{periods_summary}, "
        f"controla {tls} semáforos"
    )


def format_agents_summary(agents_data: Dict[str, Any]) -> str:
    """Format regional agents summary for inclusion in prompt (legacy format)."""
    if not agents_data or agents_data.get("total_regions", 0) == 0:
        return "No hay datos de agentes regionales disponibles."
    
    lines = [f"\nAgentes Regionales (Total: {agents_data['total_regions']}):"]
    for region in agents_data.get("regions", []):
        region_id = region.get("region_id", "Unknown")
        interventions = region.get("total_interventions", 0)
        steps = region.get("total_steps_intervening", 0)
        avg_queue = region.get("avg_queue_at_intervention", 0)
        tls = len(region.get("traffic_lights_controlled", []))
        
        lines.append(
            f"  - {region_id}: {interventions} intervenciones, "
            f"{steps} pasos activo, cola promedio={avg_queue:.1f}s, "
            f"controla {tls} semáforos"
        )
    
    return "\n".join(lines)


def format_agents_list(agents: List[AgentRecord]) -> str:
    """Format list of agent records for prompt."""
    if not agents:
        return "No hay datos de agentes regionales disponibles."
    
    lines = [f"\nAgentes Regionales (Total: {len(agents)}):"]
    for agent in agents:
        lines.append(format_agent_record(agent))
    
    return "\n".join(lines)


def build_prompt(
    question: str, 
    records: List[SimulationRecord], 
    agents_data: Dict[str, Any] | None = None,
    agents: List[AgentRecord] | None = None
) -> str:
    subset = records[:MAX_RECORDS]
    lines = "\n".join(format_record(r) for r in subset)
    
    agents_section = ""
    # Only include agents section if agents data is provided AND not empty
    if agents:
        # Use structured agent records if available
        agents_section = "\n" + format_agents_list(agents)
    elif agents_data:
        # Fallback to legacy dict format
        agents_section = "\n" + format_agents_summary(agents_data)
    
    base_instruction = "Analiza los siguientes datos de tráfico y responde a la pregunta en español."
    if agents_section:
        base_instruction += " Incluye información de agentes regionales SOLO si la pregunta es explícitamente sobre agentes, intervenciones o coordinación regional."
    
    return (
        f"{base_instruction}\n"
        f"Pregunta: {question}\n"  # pregunta del usuario
        f"Registros de Tráfico ({len(subset)}/{len(records)}):\n{lines}"  # lista formateada
        f"{agents_section}\n"  # datos de agentes regionales (solo si están presentes)
        "Si la pregunta implica comparación, menciona las vías con mayor y menor congestión."
    )
