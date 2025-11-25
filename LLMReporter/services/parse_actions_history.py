"""Parse actions_history.csv and generate a JSON summary of regional agent interventions.

Metrics exported:
- total_interventions: total number of intervention events per region
- steps_intervening: total steps each region was active
- avg_queue_at_intervention: average queue length when intervention started
- traffic_lights_affected: list of traffic lights controlled by each region
- intervention_timeline: list of intervention periods (start, end, reason)

Usage:
    python parse_actions_history.py \
        --actions-history actions_history.csv \
        --output LLMReporter/core/agents_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate actions_history.csv into agent intervention summary")
    p.add_argument("--actions-history", type=Path, default=Path("actions_history.csv"), help="Path to actions_history.csv")
    p.add_argument("--output", type=Path, default=Path("LLMReporter/core/agents_summary.json"), help="Output JSON path")
    return p.parse_args()


@dataclass
class RegionStats:
    region_id: str
    total_interventions: int = 0
    steps_intervening: int = 0
    queue_sum: float = 0.0
    queue_count: int = 0
    traffic_lights: Set[str] = field(default_factory=set)
    intervention_periods: List[Dict] = field(default_factory=list)
    current_intervention_start: int | None = None


def parse_regional_queues(queues_str: str) -> Dict[str, float]:
    """Parse 'First_Agent:123.5,Second_Agent:45.2' into dict."""
    result = {}
    if not queues_str:
        return result
    for pair in queues_str.split(","):
        if ":" in pair:
            region, value = pair.split(":", 1)
            try:
                result[region.strip()] = float(value.strip())
            except ValueError:
                pass
    return result


def parse_actions_history(csv_path: Path) -> Dict[str, RegionStats]:
    stats: Dict[str, RegionStats] = {}
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prev_step_active: Dict[str, bool] = defaultdict(bool)
        
        for row in reader:
            step = int(row.get("step", 0))
            traffic_light = row.get("traffic_light", "")
            intervened = row.get("intervened", "").lower() == "true"
            region = row.get("region", "").strip()
            reason = row.get("reason", "")
            
            # Parse regional queues
            regional_queues = parse_regional_queues(row.get("regional_queues", ""))
            
            # Track per region
            for region_id, queue in regional_queues.items():
                if region_id not in stats:
                    stats[region_id] = RegionStats(region_id=region_id)
                
                # If this region intervened on this traffic light
                if intervened and region and region_id in region:
                    stats[region_id].traffic_lights.add(traffic_light)
                    stats[region_id].steps_intervening += 1
                    
                    # Track intervention start
                    if not prev_step_active[region_id]:
                        stats[region_id].total_interventions += 1
                        stats[region_id].current_intervention_start = step
                        stats[region_id].queue_sum += queue
                        stats[region_id].queue_count += 1
                    
                    prev_step_active[region_id] = True
                else:
                    # Intervention ended
                    if prev_step_active[region_id] and stats[region_id].current_intervention_start is not None:
                        stats[region_id].intervention_periods.append({
                            "start_step": stats[region_id].current_intervention_start,
                            "end_step": step - 1,
                            "duration_steps": step - stats[region_id].current_intervention_start,
                        })
                        stats[region_id].current_intervention_start = None
                    
                    prev_step_active[region_id] = False
        
        # Close any open interventions
        for region_id, stat in stats.items():
            if prev_step_active[region_id] and stat.current_intervention_start is not None:
                stat.intervention_periods.append({
                    "start_step": stat.current_intervention_start,
                    "end_step": step,
                    "duration_steps": step - stat.current_intervention_start + 1,
                })
    
    return stats


def build_summary(stats: Dict[str, RegionStats]) -> Dict:
    summary = {
        "total_regions": len(stats),
        "regions": []
    }
    
    for region_id, stat in sorted(stats.items()):
        avg_queue = stat.queue_sum / stat.queue_count if stat.queue_count > 0 else 0.0
        
        summary["regions"].append({
            "region_id": region_id,
            "total_interventions": stat.total_interventions,
            "total_steps_intervening": stat.steps_intervening,
            "avg_queue_at_intervention": round(avg_queue, 2),
            "traffic_lights_controlled": sorted(list(stat.traffic_lights)),
            "intervention_periods": stat.intervention_periods,
        })
    
    return summary


def main() -> None:
    args = parse_args()
    
    if not args.actions_history.exists():
        print(f"Error: {args.actions_history} not found!")
        print("Run the simulation first: python ./agents/Agents_orchestator.py")
        return
    
    stats = parse_actions_history(args.actions_history)
    summary = build_summary(stats)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Wrote agent intervention summary to {args.output}")
    print(f"   Total regions: {summary['total_regions']}")
    for region in summary["regions"]:
        print(f"   - {region['region_id']}: {region['total_interventions']} interventions, {region['total_steps_intervening']} steps active")


if __name__ == "__main__":
    main()
