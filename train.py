import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from sumo_rl import parallel_env
import supersuit as ss
from reward import reward_function


def make_env(sim_dir, output_dir, use_gui=False):
    net_file = os.path.join(sim_dir, "TestLightsSogamosoNet.net.xml")
    
    route_file_lite = os.path.join(sim_dir, "osm.passenger.trips_lite.xml")
    if os.path.exists(route_file_lite):
        route_file = route_file_lite
        print(">> Usando TRÁFICO LIGERO (Lite) para entrenamiento rápido")
    else:
        route_file = os.path.join(sim_dir, "osm.passenger.trips.xml")

    out_csv = os.path.join(output_dir, "resultados_train")

    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=50000, 
        
        delta_time=10,     
        min_green=10,
        max_green=60,
        
        fixed_ts=False,    
        reward_fn=reward_function, 
        
        sumo_warnings=False,
        time_to_teleport=300,
        additional_sumo_cmd="--duration-log.disable true"
    )
    
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    
    env = VecMonitor(env)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str, default="./sumoData")
    parser.add_argument("--output_dir", type=str, default="./metrics")
    parser.add_argument("--output_model_dir", type=str, default="./models")
    parser.add_argument("--steps", type=int, default=100000) 
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    print(f"--- TRAINING PHASE ---")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    env = make_env(args.sim_dir, args.output_dir, args.gui)
    
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001, 
        buffer_size=100000,   
        learning_starts=2000, 
        batch_size=256, 
        gamma=0.99,           
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.5, 
        exploration_final_eps=0.05 
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=os.path.join(args.output_dir, 'logs/'), 
        name_prefix='sumo_dqn_v2'
    )

    print(f"Entrenando {args.steps} pasos... (Paciencia, esto toma tiempo)")
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    
    save_path = os.path.join(args.output_model_dir, "sumo_rl_final_model_v6")
    model.save(save_path)
    print("Modelo guardado exitosamente.")
    env.close()