import os
import gymnasium as gym
from stable_baselines3 import DQN
import sumo_rl
from sumo_rl import parallel_env
import supersuit as ss
import numpy as np  # <--- Necesario para manejar el array

def run_eval(sim_dir, model_path):
    print(f"--- CARGANDO MODELO IA: {model_path} ---")
    
    # 1. Rutas
    net_file = os.path.join(sim_dir, "TestLightsSogamosoNet.net.xml")
    
    # Buscar si existe la versión lite, si no, la normal
    route_file_lite = os.path.join(sim_dir, "osm.passenger.trips_lite.xml")
    if os.path.exists(route_file_lite):
        route_file = route_file_lite
        print("Usando tráfico LITE para evaluación.")
    else:
        route_file = os.path.join(sim_dir, "osm.passenger.trips.xml")
        print("Usando tráfico FULL para evaluación.")
    
    out_csv = "datos_IA"

    # 2. Crear el entorno IDÉNTICO al entrenamiento pero con GUI=True
    # Nota: sumo_warnings=False ayuda a limpiar la consola
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=True,          # <--- GUI ACTIVADA
        num_seconds=3600, 
        min_green=10,
        max_green=50,
        delta_time=5,
        reward_fn='pressure',
        sumo_warnings=False
    )
    
    # 3. Aplicar Wrappers (OBLIGATORIO igual que en train.py)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # 4. Cargar el modelo
    # Ajusta la ruta si tu modelo tiene extensión .zip o no
    if not model_path.endswith(".zip"):
        model_path += ".zip"
        
    if not os.path.exists(model_path):
        print(f"ERROR: No encuentro el modelo en {model_path}")
        return

    model = DQN.load(model_path)

    # 5. Bucle de Inferencia Corregido
    obs = env.reset()
    
    print("\nSimulación IA Iniciada...")
    print("-> Mira la ventana de SUMO que se abrió.")
    print("-> Dale al botón PLAY (triángulo verde).")
    
    done = False
    while True:
        # Predecir acción
        action, _states = model.predict(obs, deterministic=True)
        
        # Ejecutar paso
        obs, reward, done, info = env.step(action)
        
        # --- CORRECCIÓN DEL ERROR ---
        # 'done' es un array de numpy (ej: [False]).
        # Verificamos si ALGUNO es True.
        if isinstance(done, np.ndarray):
            if done.any():
                break
        else:
            if done:
                break

    env.close()
    print(f"Simulación IA Terminada. Datos guardados en {out_csv}.csv")

if __name__ == "__main__":
    # Ajusta las rutas según tu carpeta real
    # Ejemplo: python run_eval.py
    run_eval(sim_dir="./sumoData", model_path="./sumoData/sumo_rl_final_model")