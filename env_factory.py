import os
from typing import Dict, List, Optional, Tuple, Union

import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor
from sumo_rl import parallel_env

from reward import reward_function


def _resolve_route_file(sim_dir: str, preferred_route: Optional[str] = None) -> str:
    if preferred_route is not None:
        return preferred_route

    lite_route = os.path.join(sim_dir, "osm.passenger.trips_lite.xml")
    if os.path.exists(lite_route):
        return lite_route

    return os.path.join(sim_dir, "osm.passenger.trips.xml")


def build_vec_env(
    sim_dir: str,
    output_csv: Optional[str] = None,
    use_gui: bool = False,
    num_seconds: int = 3600,
    delta_time: int = 10,
    min_green: int = 10,
    max_green: int = 60,
    fixed_ts: bool = False,
    sumo_warnings: bool = True,
    additional_sumo_cmd: str = "--duration-log.disable true",
    time_to_teleport: int = 300,
    route_file: Optional[str] = None,
    return_parallel_env: bool = False,
) -> Union[VecMonitor, Tuple[VecMonitor, List[str], Dict[str, int]]]:
    """Create the same SUMO RL environment stack used during training/eval."""

    net_file = os.path.join(sim_dir, "TestLightsSogamosoNet.net.xml")
    resolved_route = _resolve_route_file(sim_dir, route_file)

    par_env = parallel_env(
        net_file=net_file,
        route_file=resolved_route,
        out_csv_name=output_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
        min_green=min_green,
        max_green=max_green,
        fixed_ts=fixed_ts,
        reward_fn=reward_function,
        sumo_warnings=sumo_warnings,
        time_to_teleport=time_to_teleport,
        additional_sumo_cmd=additional_sumo_cmd,
    )

    agent_ids = list(par_env.possible_agents)
    action_sizes: Dict[str, int] = {}
    for agent in agent_ids:
        action_space = par_env.action_spaces[agent] if hasattr(par_env, "action_spaces") else par_env.action_spaces(agent)
        action_sizes[agent] = getattr(action_space, "n", 1)

    vec_env = ss.pad_observations_v0(par_env)
    vec_env = ss.pad_action_space_v0(vec_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(vec_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    vec_env = VecMonitor(vec_env)

    if return_parallel_env:
        return vec_env, agent_ids, action_sizes

    return vec_env
