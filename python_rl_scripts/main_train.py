import pandas as pd
import numpy as np
import math
import time
import os
from concurrent.futures import ProcessPoolExecutor
import config
from simulation_worker import run_episode

try:
    import libsumo as traci
except ImportError:
    import traci

def merge_q_tables(global_qt, new_qt_list):
    # (Igual que antes)
    temp_agg = {}
    for qt in new_qt_list:
        for tls, states in qt.items():
            if tls not in temp_agg: temp_agg[tls] = {}
            for state, values in states.items():
                if state not in temp_agg[tls]:
                    temp_agg[tls][state] = {'sum': np.zeros(2), 'count': 0}
                temp_agg[tls][state]['sum'] += values
                temp_agg[tls][state]['count'] += 1
    
    for tls, states in temp_agg.items():
        if tls not in global_qt: global_qt[tls] = {}
        for state, data in states.items():
            global_qt[tls][state] = data['sum'] / data['count']
    return global_qt

def main():
    args = config.get_arguments()
    
    # --- CONFIGURACIÓN DE DIRECTORIOS ---
    # Crear directorio de resultados si no existe
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Directorio creado: {args.output_dir}")
    
    # Verificar directorio de simulación
    if not os.path.exists(args.sim_dir):
        print(f"ERROR: No se encuentra el directorio de simulación: {args.sim_dir}")
        return

    print(f"--- INICIANDO ENTRENAMIENTO ---")
    print(f"Simulación en: {args.sim_dir} | Salida en: {args.output_dir}")

    # Detección inicial (Cambiamos de directorio momentáneamente)
    original_cwd = os.getcwd()
    os.chdir(args.sim_dir) # Entrar a carpeta simulación
    
    cmd = config.get_sumo_cmd(args)
    traci.start(cmd)
    tls_ids = traci.trafficlight.getIDList()
    traci.close()
    
    os.chdir(original_cwd) # Volver a raíz
    
    # ... (Inicialización de variables igual que antes) ...
    global_q_tables = {tls: {} for tls in tls_ids}
    history_global = []
    history_agents = []
    epsilon = args.epsilon_start
    episodes_completed = 0
    total_batches = math.ceil(args.episodes / args.batch_size)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
        for batch in range(total_batches):
            batch_start = time.time()
            tasks = []
            current_batch_cnt = 0
            
            for i in range(args.batch_size):
                if episodes_completed + i < args.episodes:
                    seed = np.random.randint(0, 999999)
                    tasks.append((global_q_tables, args, epsilon, seed))
                    current_batch_cnt += 1
            
            if not tasks: break
            
            print(f"\nLote {batch + 1}/{total_batches} (Epsilon: {epsilon:.3f})...")
            results = list(executor.map(run_episode, tasks))
            
            # ... (Procesamiento de resultados IGUAL que antes) ...
            batch_q_tables = []
            for i, (learned_qt, metrics) in enumerate(results):
                real_ep = episodes_completed + i + 1
                batch_q_tables.append(learned_qt)
                
                total_rew = sum(metrics['agent_rewards'].values())
                rms = math.sqrt(np.mean(metrics['squared_errors'])) if metrics['squared_errors'] else 0
                avg_spd = np.mean(metrics['mean_speed_accum']) if metrics['mean_speed_accum'] else 0
                
                history_global.append({
                    "Episode": real_ep, "Total_Reward": total_rew, "Arrived_Vehicles": metrics['total_arrived'],
                    "Avg_Speed": avg_spd, "RMS": rms, "Epsilon": epsilon
                })
                for tls in tls_ids:
                    history_agents.append({
                        "Episode": real_ep, "Agent_ID": tls,
                        "Reward": metrics['agent_rewards'][tls], "Waiting_Time": metrics['agent_waiting_time'][tls]
                    })

            global_q_tables = merge_q_tables(global_q_tables, batch_q_tables)
            episodes_completed += current_batch_cnt
            epsilon = max(args.min_epsilon, epsilon * args.epsilon_decay)
            
            print(f" > Fin Lote en {time.time() - batch_start:.1f}s. Reward: {history_global[-1]['Total_Reward']:.0f}")

    # --- GUARDADO FINAL USANDO LAS RUTAS ---
    print("\n--- GUARDANDO DATOS ---")
    
    # Unir rutas con os.path.join
    path_global = os.path.join(args.output_dir, "metrics_global.csv")
    path_agents = os.path.join(args.output_dir, "metrics_agents.csv")
    path_qtable = os.path.join(args.output_dir, "final_q_tables.csv")
    
    pd.DataFrame(history_global).to_csv(path_global, index=False)
    print(f"Guardado: {path_global}")
    
    pd.DataFrame(history_agents).sort_values(["Episode", "Agent_ID"]).to_csv(path_agents, index=False)
    print(f"Guardado: {path_agents}")
    
    q_list = []
    for tls, table in global_q_tables.items():
        for s, vals in table.items():
            q_list.append({"Agent": tls, "State": s, "A0": vals[0], "A1": vals[1]})
    pd.DataFrame(q_list).to_csv(path_qtable, index=False)
    print(f"Guardado: {path_qtable}")
    
    print(f"--- FIN DEL ENTRENAMIENTO ---")

if __name__ == "__main__":
    main()