import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Librerías de entorno
import sumo_rl
from sumo_rl import parallel_env
import supersuit as ss
from custom_reward import fairness_reward

def make_env(sim_dir, output_dir, use_gui=False):
    # --- 1. GESTIÓN DE ARCHIVOS ---
    net_file = os.path.join(sim_dir, "TestLightsSogamosoNet.net.xml")
    
    # Priorizar el archivo de tráfico ligero
    route_file_lite = os.path.join(sim_dir, "osm.passenger.trips_lite.xml")
    route_file_full = os.path.join(sim_dir, "osm.passenger.trips.xml")
    
    if os.path.exists(route_file_lite):
        print(f"--> Usando tráfico REDUCIDO: {route_file_lite}")
        route_file = route_file_lite
    else:
        print(f"--> Usando tráfico COMPLETO: {route_file_full}")
        route_file = route_file_full

    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Error: No encuentro la red en {net_file}")
    if not os.path.exists(route_file):
        raise FileNotFoundError(f"Error: No encuentro la ruta en {route_file}")

    out_csv = os.path.join(output_dir, "resultados_sumo_rl")

    # --- 2. CREACIÓN DEL ENTORNO ---
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=3600,
        
        # --- CONFIGURACIÓN DE SEMÁFOROS ---
        fixed_ts=True,     
        delta_time=5,     
        min_green=5,
        max_green=60,
        
        # --- METRICAS Y RECOMPENSA ---
        reward_fn=fairness_reward, 
        add_system_info=True,
        add_per_agent_info=True,
        
        # --- CORRECCIÓN DE ERRORES AQUÍ ---
        sumo_warnings=False, 
        time_to_teleport=300,
        additional_sumo_cmd="--duration-log.disable true"
    )
    
    # --- 3. WRAPPERS ---
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str, default="./sumoData", help="Carpeta con los archivos .xml")
    parser.add_argument("--output_dir", type=str, default="./sumoData", help="Carpeta para guardar modelo y csv")
    parser.add_argument("--gui", action="store_true", help="Activar GUI")
    parser.add_argument("--steps", type=int, default=100000, help="Pasos totales")
    args = parser.parse_args()

    print(f"--- INICIANDO ENTRENAMIENTO (FIXED) ---")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    env = make_env(args.sim_dir, args.output_dir, args.gui)
    
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000, 
        batch_size=256, 
        gamma=0.95, 
        target_update_interval=500,
        exploration_fraction=0.2, 
        exploration_final_eps=0.05,
        device="auto" 
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=os.path.join(args.output_dir, 'logs/'), 
        name_prefix='sumo_dqn'
    )

    print(f"Entrenando por {args.steps} pasos...")
    try:
        model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
        print("Entrenamiento finalizado.")
        save_path = os.path.join(args.output_dir, "sumo_rl_final_model")
        model.save(save_path)
        print(f"Modelo guardado en: {save_path}.zip")
    except Exception as e:
        print(f"Ocurrió un error durante el entrenamiento: {e}")
    finally:
        # Asegura cerrar el entorno para liberar puertos
        env.close()