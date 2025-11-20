import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from sumo_rl import parallel_env
import supersuit as ss

from reward import reward_function

def run_eval(sim_dir, model_path):
    print(f"--- LOADING MODEL: {model_path} ---")
    
    net_file = os.path.join(sim_dir, "TestLightsSogamosoNet.net.xml")
    route_file = os.path.join(sim_dir, "osm.passenger.trips.xml")
    
    out_csv = "datos_IA_evaluacion"
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=True,          
        num_seconds=3600, 
        min_green=10,
        max_green=60,
        delta_time=10,         
        reward_fn=reward_function,
        sumo_warnings=False,
        fixed_ts=True
    )
    
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    if not model_path.endswith(".zip"):
        model_path += ".zip"
    
    try:
        model = DQN.load(model_path)
    except:
        print("ERROR: No se pudo cargar el modelo. Verifica la ruta.")
        return

    print("\n>>> INICIANDO SIMULACIÓN... <<<")
    
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    actions_stats = {0: 0, 1: 0} 

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
    
            # if isinstance(action, np.ndarray):
            #     unique, counts = np.unique(action, return_counts=True)
            #     for u, c in zip(unique, counts):
            #         actions_stats[int(u)] += int(c)
            # else:
            #     actions_stats[int(action)] += 1

            step_result = env.step(action)
            
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            total_reward += np.sum(reward)
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Paso {step_count} | Acciones acumuladas: Keep={actions_stats[0]}, Change={actions_stats[1]}")
            
            if isinstance(done, np.ndarray):
                if done.any(): break
            elif done:
                break
                
    except KeyboardInterrupt:
        print("\nStop usuario.")
    finally:
        env.close()
        print(f"Fin. Recompensa Total: {total_reward:.2f}")
        print(f"Decisiones finales: Mantener (0): {actions_stats[0]} | Cambiar (1): {actions_stats[1]}")

if __name__ == "__main__":
    run_eval(sim_dir="./sumoData", model_path="./models/sumo_rl_final_model_v6")