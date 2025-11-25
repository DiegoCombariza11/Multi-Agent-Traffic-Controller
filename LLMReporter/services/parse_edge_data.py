"""Parse SUMO edgeData.xml and generate a JSON summary per edge.

Metrics exported:
- avg_traveltime_s: mean of 'traveltime' attribute across intervals (ignoring zeros)
- avg_congestion_pct: mean of 'occupancy' * 100 across intervals (ignoring missing)
- intervals_with_data: number of intervals that had non-zero sampledSeconds
- osmids: list of original OSM ids (from net.xml 'origId' param)
- edge_id: SUMO edge id
- name / highway: if available from net file

Usage:
    python parse_edge_data.py \
        --edge-data SumoData/edgeData.xml \
        --net SumoData/TestLightsSogamosoNet.net.xml \
        --output SumoData/edge_summary.json
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate SUMO edgeData.xml metrics into JSON")
    p.add_argument("--edge-data", type=Path, default=Path("SumoData") / "edgeData.xml", help="Path to edgeData.xml")
    p.add_argument("--net", type=Path, default=Path("SumoData") / "TestLightsSogamosoNet.net.xml", help="Path to SUMO network net.xml")
    p.add_argument("--output", type=Path, default=Path("SumoData") / "edge_summary.json", help="Output JSON path")
    p.add_argument("--include-zero", action="store_true", help="Include intervals with sampledSeconds=0 in averages")
    return p.parse_args()


@dataclass
class EdgeAgg:
    edge_id: str
    traveltime_sum: float = 0.0
    traveltime_count: int = 0
    occupancy_sum: float = 0.0
    occupancy_count: int = 0
    intervals_with_data: int = 0

    def add(self, attrs: Dict[str, str], include_zero: bool) -> None:
        sampled_seconds = float(attrs.get("sampledSeconds", "0"))
        if sampled_seconds > 0 or include_zero:
            self.intervals_with_data += 1
            # traveltime
            if "traveltime" in attrs:
                tt = float(attrs.get("traveltime", "0"))
                if tt > 0:
                    self.traveltime_sum += tt
                    self.traveltime_count += 1
            # occupancy
            if "occupancy" in attrs:
                occ = float(attrs.get("occupancy", "0"))
                if occ > 0:
                    self.occupancy_sum += occ
                    self.occupancy_count += 1

    def avg_traveltime(self) -> float:
        if not self.traveltime_count:
            return 0.0
        return self.traveltime_sum / self.traveltime_count

    def avg_occupancy_pct(self) -> float:
        if not self.occupancy_count:
            return 0.0
        return (self.occupancy_sum / self.occupancy_count) * 100.0


@dataclass
class EdgeMeta:
    edge_id: str
    osmids: List[str]
    highway: str | None
    name: str | None


def load_net_metadata(net_path: Path) -> Dict[str, EdgeMeta]:
    meta: Dict[str, EdgeMeta] = {}
    # Large file: iterate efficiently
    for event, elem in ET.iterparse(net_path, events=("end",)):
        if elem.tag == "edge" and elem.get("id"):
            edge_id = elem.get("id")
            highway = elem.get("type")
            name = elem.get("name")
            orig_ids: List[str] = []
            for child in elem.findall("param"):
                if child.get("key") == "origId" and child.get("value"):
                    orig_ids = child.get("value").split()
            meta[edge_id] = EdgeMeta(edge_id=edge_id, osmids=orig_ids, highway=highway, name=name)
        elem.clear()
    return meta


def aggregate_edge_data(edge_data_path: Path, include_zero: bool) -> Dict[str, EdgeAgg]:
    agg: Dict[str, EdgeAgg] = {}
    # The file has a <meandata><interval><edge .../></interval></meandata> structure
    for event, elem in ET.iterparse(edge_data_path, events=("end",)):
        if elem.tag == "edge" and elem.get("id"):
            edge_id = elem.get("id")
            if edge_id not in agg:
                agg[edge_id] = EdgeAgg(edge_id=edge_id)
            agg[edge_id].add(elem.attrib, include_zero)
        elem.clear()
    return agg


def build_summary(agg: Dict[str, EdgeAgg], meta: Dict[str, EdgeMeta]) -> List[Dict]:
    summary: List[Dict] = []
    for edge_id, a in agg.items():
        m = meta.get(edge_id)
        summary.append(
            {
                "edge_id": edge_id,
                "osmids": m.osmids if m else [],
                "highway": m.highway if m else None,
                "name": m.name if m else None,
                "avg_traveltime_s": round(a.avg_traveltime(), 3),
                "avg_congestion_pct": round(a.avg_occupancy_pct(), 3),
                "intervals_with_data": a.intervals_with_data,
            }
        )
    summary.sort(key=lambda x: x["edge_id"])
    return summary


def main() -> None:
    args = parse_args()
    meta = load_net_metadata(args.net)
    agg = aggregate_edge_data(args.edge_data, include_zero=args.include_zero)
    summary = build_summary(agg, meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(summary)} edge summaries to {args.output}")


if __name__ == "__main__":
    main()
