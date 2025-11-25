import os
import sys
from typing import Dict, List

import numpy as np
from stable_baselines3 import DQN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env_factory import build_vec_env
from regional_agent import RegionalAgent

SIM_DIR = "./sumoData"
MODEL_PATH = "./models/sumo_rl_final_model_v6"
MAX_STEPS = 360


def _ensure_model_path(path: str) -> str:
    return path if path.endswith(".zip") else f"{path}.zip"


def _build_regions() -> List[RegionalAgent]:
    # release_threshold (hysteresis) defaults to queue_threshold - 1
    return [
        RegionalAgent(
            "First_Agent",
            [
                "5237321225",
                "cluster_1016185040_1016195865_9372174234_9372174235",
                "cluster_1016191445_1016199645",
            ],
            queue_threshold=200,
            min_intervention_steps=10,
        ),
        RegionalAgent(
            "Second_Agent",
            [
                "cluster_1016184655_1016193545_1016201012_9372174241",
                "cluster_1016184252_1016191971",
                "1016184376",
            ],
            queue_threshold=200,
            min_intervention_steps=10,
        ),
        RegionalAgent(
            "Third_Agent",
            [
                "cluster_1016189230_1016195084_5921362584_9371895544",
                "cluster_1016185522_1016193644_5237321232_5921362889_#1more",
                "cluster_1016192514_4084049672_5921362888",
            ],
            queue_threshold=200,
            min_intervention_steps=10,
        ),
        RegionalAgent(
            "Fourth_Agent",
            [
                "cluster_1016191801_1016199119",
                "cluster_1016184997_1016188102_1016195714_5921483463",
                "cluster_1016191702_1016195035",
            ],
            queue_threshold=200,
            min_intervention_steps=10,
        ),
        RegionalAgent(
            "Fifth_Agent",
            [
                "cluster_1016183180_1016185346_1016193503_1016194749",
                "GS_cluster_1016191643_1016191860_1016192269_1016198634",
            ],
            queue_threshold=200,
            min_intervention_steps=10,
        ),
    ]


def _apply_phase_limits(actions: np.ndarray, action_sizes: Dict[str, int], tl_index_map: Dict[str, int]) -> None:
    """Clamp each TL action to the valid number of phases learned during training."""

    def clamp(action_row: np.ndarray) -> None:
        for tl_id, idx in tl_index_map.items():
            limit = action_sizes.get(tl_id)
            if not limit or limit <= 0 or idx >= action_row.shape[-1]:
                continue
            action_row[idx] = int(action_row[idx]) % limit

    if actions.ndim == 1:
        clamp(actions)
    else:
        for row in actions:
            clamp(row)


def run():
    env, traffic_lights, action_sizes = build_vec_env(
        sim_dir=SIM_DIR,
        output_csv="./metrics/orchestrator_eval",
        use_gui=False,
        num_seconds=3600,
        fixed_ts=True,
        sumo_warnings=False,
        return_parallel_env=True,
    )

    tl_index_map = {tl: idx for idx, tl in enumerate(traffic_lights)}
    regions = _build_regions()
    model = DQN.load(_ensure_model_path(MODEL_PATH))


    import csv
    history = []

    obs = env.reset()
    step = 0
    last_info: List[dict] = [{}]

    # Detect if actions are 1D or 2D (multi-agent)
    is_2d = False
    actions_test, _ = model.predict(obs, deterministic=True)
    actions_test = np.array(actions_test)
    if actions_test.ndim == 2:
        is_2d = True

    while step < MAX_STEPS:
        actions, _ = model.predict(obs, deterministic=True)
        actions = np.array(actions, copy=True)
        _apply_phase_limits(actions, action_sizes, tl_index_map)
        rl_actions = actions.copy()  # Guardar acciones RL antes de intervención

        info_dict = last_info[0] if isinstance(last_info, list) and last_info else {}
        
        # Debug: mostrar claves de info en los primeros pasos
        if step < 3:
            print(f"[DEBUG step={step}] info_dict keys: {list(info_dict.keys())[:20]}")
            stopped_keys = [k for k in info_dict.keys() if '_stopped' in k]
            print(f"[DEBUG step={step}] _stopped keys: {stopped_keys}")
        
        # Capturar colas regionales ANTES de aplicar intervenciones
        region_queues_before = {region.region_id: region.get_regional_queue(info_dict) for region in regions}
        
        interventions = []
        # Aplicar intervención regional y registrar motivos
        for region in regions:
            intervened = region.step(info_dict, actions, tl_index_map)
            if intervened:
                interventions.append({
                    "region_id": region.region_id,
                    "queue": region_queues_before[region.region_id],
                    "reason": f"queue >= {region.queue_threshold}"
                })
        _apply_phase_limits(actions, action_sizes, tl_index_map)

        # Preparar estructuras de estado regional para logging (usando colas capturadas)
        region_states = {
            region.region_id: {
                "queue": region_queues_before[region.region_id],
                "intervening": region.intervening,
                "remaining": region.remaining_steps,
                "targets": set(region.intersections),
            }
            for region in regions
        }

        # Guardar historial por semáforo
        if is_2d:
            # Multi-agent: cada fila es un agente, cada columna un semáforo
            for agent_idx, row in enumerate(rl_actions):
                for tl, idx in tl_index_map.items():
                    active_regions = [rid for rid, st in region_states.items() if st["intervening"] and tl in st["targets"]]
                    queues_str = ",".join([f"{rid}:{region_states[rid]['queue']}" for rid in region_states])
                    remaining_str = ",".join([f"{rid}:{region_states[rid]['remaining']}" for rid in region_states])
                    history.append({
                        "step": step,
                        "traffic_light": tl,
                        "agent": agent_idx,
                        "rl_action": int(row[idx]),
                        "final_action": int(actions[agent_idx, idx]),
                        "intervened": any(idx in [tl_index_map.get(tl2) for region in regions for tl2 in region.intersections] for region in regions if region.intervening),
                        "region": ",".join([region.region_id for region in regions if region.intervening and idx in [tl_index_map.get(tl2) for tl2 in region.intersections]]),
                        "reason": ",".join([f"queue={region.get_regional_queue(info_dict)} >= {region.queue_threshold}" for region in regions if region.intervening and idx in [tl_index_map.get(tl2) for tl2 in region.intersections]])
                        ,
                        "regional_queues": queues_str,
                        "remaining_steps": remaining_str,
                        "active_regions": ",".join(active_regions),
                    })
        else:
            # 1D: un solo vector de acciones
            for tl, idx in tl_index_map.items():
                intervened_regions = [region for region in regions if region.intervening and idx in [tl_index_map.get(tl2) for tl2 in region.intersections]]
                active_regions = [rid for rid, st in region_states.items() if st["intervening"] and tl in st["targets"]]
                queues_str = ",".join([f"{rid}:{region_states[rid]['queue']}" for rid in region_states])
                remaining_str = ",".join([f"{rid}:{region_states[rid]['remaining']}" for rid in region_states])
                history.append({
                    "step": step,
                    "traffic_light": tl,
                    "rl_action": int(rl_actions[idx]),
                    "final_action": int(actions[idx]),
                    "intervened": bool(intervened_regions),
                    "region": ",".join([region.region_id for region in intervened_regions]),
                    "reason": ",".join([f"queue={region.get_regional_queue(info_dict)} >= {region.queue_threshold}" for region in intervened_regions])
                    ,
                    "regional_queues": queues_str,
                    "remaining_steps": remaining_str,
                    "active_regions": ",".join(active_regions),
                })

        obs, reward, dones, infos = env.step(actions)
        last_info = infos
        step += 1

        # Break BEFORE VecMonitor can auto-reset
        if np.any(dones):
            print(f"[INFO] Episode done detected at step {step}")
            break

    print(f"\n[INFO] Simulation completed: {step} steps executed out of {MAX_STEPS} max")
    print(f"[INFO] Total simulated time: ~{step * 10} seconds ({step * 10 / 3600:.2f} hours)")

    # Guardar historial en CSV
    csv_fields = [
        "step",
        "traffic_light",
        "rl_action",
        "final_action",
        "intervened",
        "region",
        "reason",
        "regional_queues",
        "remaining_steps",
        "active_regions",
    ]
    with open("./metrics/actions_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in csv_fields})

    env.close()


if __name__ == "__main__":
    run()
