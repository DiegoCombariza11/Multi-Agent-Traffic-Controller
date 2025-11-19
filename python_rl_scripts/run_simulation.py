import pandas as pd
import numpy as np
import os
import sys
import argparse
import time
import config  # Usamos tu config para leer rutas

# NOTA: Para visualización OBLIGATORIAMENTE usamos traci estándar
import traci

# --- CONSTANTES ---
MIN_GREEN_TIME = 15  # Debe coincidir con lo que usaste en training
SECONDS_PER_ACTION = 5

def load_q_tables(csv_path):
    """Reconstruye el diccionario de Q-Tables desde el CSV."""
    if not os.path.exists(csv_path):
        print(f"ERROR CRÍTICO: No se encuentra {csv_path}")
        sys.exit(1)

    print("Cargando cerebro de los agentes...")
    df = pd.read_csv(csv_path)
    q_table = {}
    
    for _, row in df.iterrows():
        agent = row['Agent']
        state = str(row['State']) # Asegurar string "0012"
        # Reconstruir vector [Valor0, Valor1]
        values = np.array([row['A0'], row['A1']])
        
        if agent not in q_table:
            q_table[agent] = {}
        q_table[agent][state] = values
        
    print(f"Cargados {len(q_table)} agentes correctamente.")
    return q_table

def get_state(tls_id, controlled_lanes):
    """
    Misma función de estado que usamos en simulation_worker.py
    IMPORTANTE: Debe ser IDÉNTICA a la del entrenamiento.
    """
    state = ""
    # Ordenamos carriles para consistencia
    lanes = sorted(controlled_lanes[tls_id])
    
    for lane in lanes:
        try:
            # Halting number (< 0.1 m/s)
            q = traci.lane.getLastStepHaltingNumber(lane)
        except:
            q = 0
            
        if q < 3: state += "0"
        elif q < 8: state += "1"
        elif q < 15: state += "2"
        else: state += "3"
    return state

def run_visual_simulation(args):
    # 1. Cargar Tablas Q
    csv_path = os.path.join(args.output_dir, "final_q_tables.csv")
    q_tables = load_q_tables(csv_path)

    # 2. Preparar SUMO-GUI
    original_dir = os.getcwd()
    
    # Namespace dummy para reutilizar la función de comando
    # Forzamos GUI = True y time-to-teleport alto para ver si hay bloqueos
    sumo_cmd = ["sumo-gui", 
                "-c", args.sumo_cfg,
                "--no-step-log", "true",
                "--waiting-time-memory", "1000",
                "--time-to-teleport", "300"]
    
    print(f"Iniciando simulación visual en: {args.sim_dir}")
    
    try:
        os.chdir(args.sim_dir)
        traci.start(sumo_cmd)
        
        # 3. Detección Inicial
        tls_ids = traci.trafficlight.getIDList()
        controlled_lanes = {tls: list(set(traci.trafficlight.getControlledLanes(tls))) for tls in tls_ids}
        
        # Detectar fases
        num_phases = {}
        for tls in tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            num_phases[tls] = len(logic.phases)

        # Variables de control
        last_phase_change = {tls: 0 for tls in tls_ids}
        current_states = {tls: get_state(tls, controlled_lanes) for tls in tls_ids}
        step = 0
        
        print("Simulación corriendo... Presiona PLAY en la ventana de SUMO.")

        while traci.simulation.getMinExpectedNumber() > 0:
            current_time = traci.simulation.getTime()
            actions = {}

            # --- A. DECISIÓN (MODO EXPLOTACIÓN) ---
            for tls in tls_ids:
                # 1. Respetar tiempo mínimo de verde (Física del semáforo)
                if (current_time - last_phase_change[tls]) < MIN_GREEN_TIME:
                    actions[tls] = 0 # Obligado a mantener
                else:
                    # 2. Consultar al Cerebro (Q-Table)
                    state = current_states[tls]
                    
                    if tls in q_tables and state in q_tables[tls]:
                        # Elegir la MEJOR acción (Argmax) -> Epsilon es 0 aquí
                        action = np.argmax(q_tables[tls][state])
                    else:
                        # Si el estado nunca se vio en entrenamiento, mantenemos por seguridad
                        # (O puedes imprimir un aviso para debug)
                        action = 0 
                    
                    actions[tls] = action

            # --- B. ACTUAR ---
            for tls, act in actions.items():
                if act == 1:
                    curr = traci.trafficlight.getPhase(tls)
                    traci.trafficlight.setPhase(tls, (curr + 1) % num_phases[tls])
                    last_phase_change[tls] = current_time
            
            # --- C. AVANZAR ---
            # Avanzamos varios pasos para fluidez, pero en GUI se verá continuo
            for _ in range(SECONDS_PER_ACTION):
                traci.simulationStep()
                step += 1
            
            # Actualizar estados para la siguiente decisión
            current_states = {tls: get_state(tls, controlled_lanes) for tls in tls_ids}

            # (Opcional) Pequeña pausa si en tu PC corre demasiado rápido
            # time.sleep(0.1)

        print(f"Simulación finalizada. Vehículos totales llegados: {traci.simulation.getArrivedNumber()}")
        traci.close()

    except Exception as e:
        print(f"Simulación interrumpida: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str, default=config.DEFAULT_CONFIG["sim_dir"])
    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--sumo_cfg", type=str, default=config.DEFAULT_CONFIG["sumo_cfg"])
    args = parser.parse_args()
    
    run_visual_simulation(args)