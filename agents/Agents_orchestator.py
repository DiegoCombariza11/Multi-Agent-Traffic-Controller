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
MAX_STEPS = 3600


def _ensure_model_path(path: str) -> str:
    return path if path.endswith(".zip") else f"{path}.zip"


def _build_regions() -> List[RegionalAgent]:
    return [
        RegionalAgent(
            "First_Agent",
            [
                "5237321225",
                "cluster_1016185040_1016195865_9372174234_9372174235",
                "cluster_1016191445_1016199645",
            ],
            queue_threshold=20,
        ),
        RegionalAgent(
            "Second_Agent",
            [
                "cluster_1016184655_1016193545_1016201012_9372174241",
                "cluster_1016184252_1016191971",
                "1016184376",
            ],
            queue_threshold=20,
        ),
        RegionalAgent(
            "Third_Agent",
            [
                "cluster_1016189230_1016195084_5921362584_9371895544",
                "cluster_1016185522_1016193644_5237321232_5921362889_#1more",
                "cluster_1016192514_4084049672_5921362888",
            ],
            queue_threshold=20,
        ),
        RegionalAgent(
            "Fourth_Agent",
            [
                "cluster_1016191801_1016199119",
                "cluster_1016184997_1016188102_1016195714_5921483463",
                "cluster_1016191702_1016195035",
            ],
            queue_threshold=20,
        ),
        RegionalAgent(
            "Fifth_Agent",
            [
                "cluster_1016183180_1016185346_1016193503_1016194749",
                "GS_cluster_1016191643_1016191860_1016192269_1016198634",
            ],
            queue_threshold=20,
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
        use_gui=True,
        num_seconds=MAX_STEPS,
        fixed_ts=True,
        sumo_warnings=False,
        return_parallel_env=True,
    )

    tl_index_map = {tl: idx for idx, tl in enumerate(traffic_lights)}
    regions = _build_regions()
    model = DQN.load(_ensure_model_path(MODEL_PATH))

    obs = env.reset()
    step = 0
    last_info: List[dict] = [{}]

    while step < MAX_STEPS:
        actions, _ = model.predict(obs, deterministic=True)
        actions = np.array(actions, copy=True)
        _apply_phase_limits(actions, action_sizes, tl_index_map)
        if step < 5:
            print("raw actions normalized:", actions)

        info_dict = last_info[0] if isinstance(last_info, list) and last_info else {}
        for region in regions:
            region.step(info_dict, actions, tl_index_map)
        _apply_phase_limits(actions, action_sizes, tl_index_map)


        obs, reward, dones, infos = env.step(actions)
        last_info = infos
        
        step += 1

        if np.any(dones):
            break

    env.close()


if __name__ == "__main__":
    run()
