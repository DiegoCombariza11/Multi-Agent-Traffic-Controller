import argparse
import math
import os
from typing import Callable, Dict, List

import pandas as pd
import sumolib
from stable_baselines3 import DQN

from agents.Agents_orchestator import _apply_phase_limits, _build_regions, _ensure_model_path
from env_factory import build_vec_env

METRIC_SUFFIXES = {
    "waiting": "_accumulated_waiting_time",
    "stopped": "_stopped",
    "speed": "_average_speed",
}

AGGREGATORS: Dict[str, Callable[[pd.Series], float]] = {
    "mean": lambda s: float(s.mean()),
    "median": lambda s: float(s.median()),
    "max": lambda s: float(s.max()),
    "min": lambda s: float(s.min()),
    "last": lambda s: float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan"),
}


def _collect_infos(
    sim_dir: str,
    model_path: str,
    csv_prefix: str,
    max_steps: int,
    use_gui: bool,
) -> List[Dict[str, float]]:
    env, traffic_lights, action_sizes = build_vec_env(
        sim_dir=sim_dir,
        output_csv=csv_prefix,
        use_gui=use_gui,
        num_seconds=max_steps,
        fixed_ts=True,
        sumo_warnings=False,
        return_parallel_env=True,
    )

    tl_index_map = {tl: idx for idx, tl in enumerate(traffic_lights)}
    regions = _build_regions()
    model = DQN.load(_ensure_model_path(model_path))

    obs = env.reset()
    step = 0
    infos_history: List[Dict[str, float]] = []
    last_info: List[dict] = [{}]

    while step < max_steps:
        actions, _ = model.predict(obs, deterministic=True)
        actions = actions.copy()
        _apply_phase_limits(actions, action_sizes, tl_index_map)

        info_dict = last_info[0] if isinstance(last_info, list) and last_info else {}
        for region in regions:
            region.step(info_dict, actions, tl_index_map)
        _apply_phase_limits(actions, action_sizes, tl_index_map)

        obs, _, dones, infos = env.step(actions)
        last_info = infos
        info_for_storage = infos[0] if isinstance(infos, list) and infos else None
        if info_for_storage:
            infos_history.append(dict(info_for_storage))

        step += 1
        if any(dones):
            break

    env.close()
    return infos_history


def _aggregate_tls_metrics(df: pd.DataFrame, metric_suffix: str, aggregator: Callable[[pd.Series], float]) -> Dict[str, float]:
    tls_values: Dict[str, float] = {}
    for column in df.columns:
        if not column.endswith(metric_suffix):
            continue
        if column == metric_suffix:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        tls_id = column[: -len(metric_suffix)]
        tls_values[tls_id] = aggregator(series)
    return tls_values


def _map_edges_to_tls(net: sumolib.net.Net, tls_metrics: Dict[str, float]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for edge in net.getEdges():
        if edge.getFunction() in {"internal", "walkingarea", "connector"}:
            continue
        tls_id = edge.getToNode().getID()
        if tls_id not in tls_metrics:
            continue
        osm_attr = edge.getParam("origId", "")
        osm_ids = osm_attr.split() if osm_attr else [""]
        for osm_id in osm_ids:
            rows.append(
                {
                    "edge_id": edge.getID(),
                    "osm_id": osm_id or edge.getID(),
                    "tls_id": tls_id,
                    "lanes": edge.getLaneNumber(),
                    "length_m": edge.getLength(),
                    "metric_value": tls_metrics[tls_id],
                }
            )
    return rows


def _normalize_metric(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    values = [row["metric_value"] for row in rows if not math.isnan(row["metric_value"])]
    if not values:
        for row in rows:
            row["metric_norm"] = float("nan")
        return
    max_val = max(values)
    min_val = min(values)
    for row in rows:
        value = row["metric_value"]
        if math.isnan(value) or math.isclose(max_val, min_val):
            row["metric_norm"] = float("nan")
        else:
            row["metric_norm"] = (value - min_val) / (max_val - min_val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el orquestador SUMO-RL y genera un resumen promedio de congestión "
            "por tramo (OSM id) para toda la red de Sogamoso."
        )
    )
    parser.add_argument("--sim-dir", type=str, default="./sumoData", help="Directorio con net/route de SUMO.")
    parser.add_argument("--model", type=str, default="./models/sumo_rl_final_model_v6.zip", help="Modelo DQN entrenado.")
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default="./metrics/orchestrator_eval",
        help="Prefijo para los CSV estándar exportados por sumo-rl.",
    )
    parser.add_argument("--net", type=str, default="./sumoData/TestLightsSogamosoNet.net.xml", help="Archivo .net.xml.")
    parser.add_argument("--max-steps", type=int, default=3600, help="Número máximo de pasos simulados.")
    parser.add_argument("--use-gui", action="store_true", help="Mostrar interfaz gráfica de SUMO.")
    parser.add_argument(
        "--metric-type",
        type=str,
        default="waiting",
        choices=sorted(METRIC_SUFFIXES.keys()),
        help="Tipo de métrica de congestión a resumir (waiting/stopped/speed).",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=sorted(AGGREGATORS.keys()),
        help="Cómo resumir la serie temporal de cada semáforo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./metrics/orchestrator_edge_congestion.csv",
        help="Ruta del CSV de resumen por vía.",
    )
    parser.add_argument("--top", type=int, default=15, help="Cantidad de tramos más congestionados que se imprimirán.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.metric_type not in METRIC_SUFFIXES:
        raise ValueError(f"Tipo de métrica desconocido: {args.metric_type}")

    if args.aggregator not in AGGREGATORS:
        raise ValueError(f"Aggregador no soportado: {args.aggregator}")

    infos_history = _collect_infos(
        sim_dir=args.sim_dir,
        model_path=args.model,
        csv_prefix=args.csv_prefix,
        max_steps=args.max_steps,
        use_gui=args.use_gui,
    )

    if not infos_history:
        raise RuntimeError("No se capturaron métricas durante la ejecución del orquestador.")

    df = pd.DataFrame(infos_history)
    metric_suffix = METRIC_SUFFIXES[args.metric_type]
    aggregator = AGGREGATORS[args.aggregator]
    tls_metrics = _aggregate_tls_metrics(df, metric_suffix, aggregator)

    net = sumolib.net.readNet(args.net, withInternal=False)
    rows = _map_edges_to_tls(net, tls_metrics)
    _normalize_metric(rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("metric_value", ascending=args.metric_type == "speed", inplace=True)
    summary_df.to_csv(args.output, index=False)
    print(f"Resumen guardado en: {args.output} ({len(summary_df)} tramos)")

    if args.top > 0 and not summary_df.empty:
        print("Top tramos congestionados:")
        print(summary_df.head(args.top)[["osm_id", "edge_id", "tls_id", "metric_value", "metric_norm"]])


if __name__ == "__main__":
    main()
