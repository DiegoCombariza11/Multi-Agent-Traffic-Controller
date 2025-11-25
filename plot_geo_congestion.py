import argparse
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import folium
from branca.colormap import linear
from folium.plugins import HeatMap
import pandas as pd
import sumolib

MetricSummary = Dict[str, float]
TLSPoints = List[Dict[str, float]]

METRIC_SUFFIXES = {
    "waiting": "_accumulated_waiting_time",
    "stopped": "_stopped",
    "speed": "_average_speed",
}


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    df.rename(columns=lambda col: col.strip(), inplace=True)
    return df


def build_aggregator(name: str) -> Callable[[pd.Series], float]:
    def last_valid(series: pd.Series) -> float:
        non_nan = series.dropna()
        return float(non_nan.iloc[-1]) if not non_nan.empty else float("nan")

    aggregations: Dict[str, Callable[[pd.Series], float]] = {
        "mean": lambda s: float(s.mean()),
        "median": lambda s: float(s.median()),
        "max": lambda s: float(s.max()),
        "min": lambda s: float(s.min()),
        "last": last_valid,
    }
    if name not in aggregations:
        valid = ", ".join(sorted(aggregations))
        raise ValueError(f"Unknown aggregator '{name}'. Valid options: {valid}")
    return aggregations[name]


def summarize_tls_metrics(
    df: pd.DataFrame,
    suffix: str,
    aggregator: Callable[[pd.Series], float],
) -> MetricSummary:
    summary: MetricSummary = {}
    for column in df.columns:
        if not column.endswith(suffix):
            continue
        tls_id = column[: -len(suffix)]
        series = pd.to_numeric(df[column], errors="coerce")
        summary[tls_id] = aggregator(series)
    return summary


def build_tls_points(net: sumolib.net.Net, metrics: MetricSummary) -> TLSPoints:
    points: TLSPoints = []
    for tls_id, value in metrics.items():
        if math.isnan(value):
            continue
        node = net.getNode(tls_id)
        if node is None:
            continue
        x, y = node.getCoord()
        lon, lat = net.convertXY2LonLat(x, y)
        points.append({"id": tls_id, "lat": lat, "lon": lon, "value": value})
    return points


def add_road_layer(net: sumolib.net.Net, fmap: folium.Map) -> None:
    roads = folium.FeatureGroup(name="Red vial", show=True)
    for edge in net.getEdges():
        if edge.getFunction() in {"internal", "walkingarea", "connector"}:
            continue
        shape = edge.getShape()
        if not shape or len(shape) < 2:
            continue
        coords = []
        for x, y in shape:
            lon, lat = net.convertXY2LonLat(x, y)
            coords.append((lat, lon))
        roads.add_child(folium.PolyLine(coords, color="#666666", weight=1.5, opacity=0.5))
    roads.add_to(fmap)


def normalize_values(points: TLSPoints) -> Dict[str, float]:
    values = [pt["value"] for pt in points]
    if not values:
        return {}
    min_val = min(values)
    max_val = max(values)
    if math.isclose(max_val, min_val):
        return {pt["id"]: 1.0 for pt in points}
    return {pt["id"]: (pt["value"] - min_val) / (max_val - min_val) for pt in points}


def add_scenario_layers(
    fmap: folium.Map,
    label: str,
    points: TLSPoints,
    show: bool,
) -> None:
    if not points:
        print(f"[WARN] Scenario '{label}' has no TLS metrics to plot.")
        return

    normalized = normalize_values(points)
    heat_data = [
        [pt["lat"], pt["lon"], normalized.get(pt["id"], 0.0)]
        for pt in points
        if pt["id"] in normalized
    ]

    colormap = linear.YlOrRd_09.scale(min(pt["value"] for pt in points), max(pt["value"] for pt in points))
    heat_group = folium.FeatureGroup(name=f"Mapa de calor - {label}", show=show)
    HeatMap(
        heat_data,
        min_opacity=0.35,
        radius=35,
        blur=25,
        max_val=1.0,
    ).add_to(heat_group)
    heat_group.add_to(fmap)

    marker_group = folium.FeatureGroup(name=f"Intersecciones - {label}", show=False)
    for pt in points:
        color = colormap(pt["value"]) if not math.isnan(pt["value"]) else "#999999"
        popup = folium.Popup(
            f"{label}<br>ID: {pt['id']}<br>Valor: {pt['value']:.2f}",
            max_width=300,
        )
        folium.CircleMarker(
            location=(pt["lat"], pt["lon"]),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup,
        ).add_to(marker_group)
    marker_group.add_to(fmap)


def compute_center(net: sumolib.net.Net) -> Tuple[float, float]:
    xs, ys = [], []
    for node in net.getNodes():
        x, y = node.getCoord()
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        raise RuntimeError("SUMO network contains no nodes.")
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    lon, lat = net.convertXY2LonLat(center_x, center_y)
    return lat, lon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera mapas de calor georreferenciados de congestión por escenario.",
    )
    parser.add_argument("--net", type=str, default="./sumoData/TestLightsSogamosoNet.net.xml", help="Archivo SUMO .net.xml con la red de Sogamoso.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="./datos_baseline.csv",
        help="CSV de la simulación base sin entrenamiento (por defecto ./datos_baseline.csv).",
    )
    parser.add_argument(
        "--locals",
        type=str,
        default="./graphics/datos_IA_evaluacion_conn1_ep1.csv",
        help="CSV de la simulación con agentes locales (por defecto ./graphics/datos_IA_evaluacion_conn1_ep1.csv).",
    )
    parser.add_argument(
        "--regional",
        type=str,
        default="./metrics/orchestrator_eval_conn1_ep1.csv",
        help="CSV de la simulación con agentes locales+regionales (por defecto ./metrics/orchestrator_eval_conn1_ep1.csv).",
    )
    parser.add_argument("--baseline-label", type=str, default="Base", help="Etiqueta para la capa base.")
    parser.add_argument("--locals-label", type=str, default="Agentes Locales", help="Etiqueta para la capa local.")
    parser.add_argument("--regional-label", type=str, default="Locales + Regionales", help="Etiqueta para la capa regional.")
    parser.add_argument(
        "--metric-type",
        type=str,
        default="waiting",
        choices=sorted(METRIC_SUFFIXES.keys()),
        help="Tipo de métrica por semáforo: waiting (tiempo acumulado), stopped o speed.",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "median", "max", "min", "last"],
        help="Cómo resumir cada serie temporal por semáforo.",
    )
    parser.add_argument("--output", type=str, default="./graphics/congestion_map.html", help="Ruta del HTML generado.")
    parser.add_argument("--zoom", type=int, default=14, help="Nivel inicial de zoom del mapa.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abrir automáticamente el HTML generado en el navegador predeterminado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.metric_type not in METRIC_SUFFIXES:
        valid = ", ".join(sorted(METRIC_SUFFIXES))
        raise ValueError(f"Unsupported metric type '{args.metric_type}'. Use one of: {valid}")

    aggregator = build_aggregator(args.aggregator)
    suffix = METRIC_SUFFIXES[args.metric_type]

    baseline_metrics = summarize_tls_metrics(load_csv(args.baseline), suffix, aggregator)
    locals_metrics = summarize_tls_metrics(load_csv(args.locals), suffix, aggregator)
    regional_metrics = summarize_tls_metrics(load_csv(args.regional), suffix, aggregator)

    net = sumolib.net.readNet(args.net, withInternal=False)
    center_lat, center_lon = compute_center(net)

    fmap = folium.Map(location=(center_lat, center_lon), zoom_start=args.zoom, tiles="CartoDB Positron")
    add_road_layer(net, fmap)

    scenarios: Sequence[Tuple[str, MetricSummary]] = (
        (args.baseline_label, baseline_metrics),
        (args.locals_label, locals_metrics),
        (args.regional_label, regional_metrics),
    )

    for idx, (label, metrics) in enumerate(scenarios):
        points = build_tls_points(net, metrics)
        add_scenario_layers(fmap, label, points, show=(idx == 0))

    folium.LayerControl(collapsed=False).add_to(fmap)

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    fmap.save(args.output)
    print(f"Mapa interactivo guardado en: {args.output}")
    if args.show:
        try:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(args.output))
        except Exception as e:
            print(f"[WARN] No se pudo abrir el navegador automáticamente: {e}")


if __name__ == "__main__":
    main()
