import numpy as np
import copy
import os
import sys
import config

# --- CONSTANTES DE CONTROL ---
MAX_STEPS = 3600
SECONDS_PER_ACTION = 5    # Frecuencia de "pensamiento" del agente
MIN_GREEN_TIME = 15       # Segundos mínimos obligatorios antes de permitir otro cambio
SPEED_SAMPLE_RATE = 10

# --- FUNCIONES AUXILIARES ---

def get_state(tls_id, traci_instance, controlled_lanes):
    """Estado: Discretización de la cola de espera"""
    state = ""
    # Ordenamos para consistencia
    lanes = sorted(controlled_lanes[tls_id])
    for lane in lanes:
        try:
            # Usamos halted (velocidad < 0.1 m/s)
            q = traci_instance.lane.getLastStepHaltingNumber(lane)
        except:
            q = 0
        
        if q < 3: state += "0"      # Libre
        elif q < 8: state += "1"    # Leve
        elif q < 15: state += "2"   # Moderado
        else: state += "3"          # Crítico
    return state

def get_advanced_reward(tls_id, traci_instance, controlled_lanes):
    """
    Recompensa mejorada:
    1. Penalización base: Tiempo de espera acumulado.
    2. Penalización crítica: Saturación de carril (Jam penalty).
    """
    total_waiting_time = 0
    jam_penalty = 0
    
    for lane in controlled_lanes[tls_id]:
        # A. Sumar tiempo de espera de cada vehículo (Preciso)
        veh_ids = traci_instance.lane.getLastStepVehicleIDs(lane)
        for veh_id in veh_ids:
            try:
                total_waiting_time += traci_instance.vehicle.getWaitingTime(veh_id)
            except:
                pass

        # B. Detectar Saturación (Evitar Deadlocks)
        # Si el carril está lleno de carros (independientemente de si esperan o no)
        try:
            # Cantidad de carros / Capacidad del carril (Longitud / 7.5m)
            num_veh = traci_instance.lane.getLastStepVehicleNumber(lane)
            capacity = traci_instance.lane.getLength(lane) / 7.5 
            
            if num_veh > (capacity * 0.8): # Si está al 80% de capacidad
                jam_penalty += 500 # Castigo fuerte extra
        except:
            pass

    # La recompensa es negativa. Queremos acercarnos a 0.
    reward = -1.0 * (total_waiting_time + jam_penalty)
    return reward

# --- WORKER PRINCIPAL ---

def run_episode(input_data):
    initial_q_tables, args, epsilon, seed = input_data
    
    np.random.seed(seed)
    try:
        import libsumo as traci
    except ImportError:
        import traci

    original_dir = os.getcwd()
    
    # Estructuras de métricas
    metrics = {
        'agent_rewards': {},
        'agent_waiting_time': {},
        'total_arrived': 0,
        'mean_speed_accum': [],
        'squared_errors': []
    }
    local_q_tables = copy.deepcopy(initial_q_tables)

    try:
        os.chdir(args.sim_dir)
        sumo_cmd = config.get_sumo_cmd(args)
        traci.start(sumo_cmd)
        
        # Setup Red
        tls_ids = traci.trafficlight.getIDList()
        controlled_lanes = {tls: list(set(traci.trafficlight.getControlledLanes(tls))) for tls in tls_ids}
        
        # Fases y Temporizadores
        num_phases = {}
        last_phase_change = {} # Para controlar el Min Green Time
        
        for tls in tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            num_phases[tls] = len(logic.phases)
            last_phase_change[tls] = 0 # Iniciamos en tiempo 0
            
            metrics['agent_rewards'][tls] = 0.0
            metrics['agent_waiting_time'][tls] = 0.0

        current_states = {tls: get_state(tls, traci, controlled_lanes) for tls in tls_ids}
        step = 0
        
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            current_time = traci.simulation.getTime()
            actions = {}
            
            # --- A. DECISIÓN CON RESTRICCIÓN DE TIEMPO ---
            for tls in tls_ids:
                # 1. Verificar si cumplimos el tiempo mínimo de verde
                time_since_change = current_time - last_phase_change[tls]
                
                if time_since_change < MIN_GREEN_TIME:
                    # Aún no podemos cambiar. Forzamos acción 0 (Mantener)
                    actions[tls] = 0
                else:
                    # 2. Si ya pasó el tiempo, el Agente decide
                    if current_states[tls] not in local_q_tables[tls]:
                        local_q_tables[tls][current_states[tls]] = np.zeros(2)
                    
                    qs = local_q_tables[tls][current_states[tls]]
                    
                    if np.random.rand() < epsilon:
                        action = np.random.randint(2)
                    else:
                        action = np.argmax(qs)
                    actions[tls] = action

            # --- B. ACTUAR ---
            for tls, act in actions.items():
                if act == 1: 
                    # Cambiar Fase
                    curr = traci.trafficlight.getPhase(tls)
                    traci.trafficlight.setPhase(tls, (curr + 1) % num_phases[tls])
                    # Actualizar el timer
                    last_phase_change[tls] = current_time
                else:
                    # Mantener Fase
                    pass
            
            # --- C. AVANZAR SIMULACIÓN ---
            step_rewards = {tls: 0 for tls in tls_ids}
            
            for _ in range(SECONDS_PER_ACTION):
                traci.simulationStep()
                step += 1
                
                # Calcular recompensa en cada paso
                for tls in tls_ids:
                    r = get_advanced_reward(tls, traci, controlled_lanes)
                    step_rewards[tls] += r
                    # Guardamos el valor absoluto de la espera (sin el castigo de jam) para el reporte
                    # (Aprox, ya que el reward incluye jam, pero sirve de referencia)
                    metrics['agent_waiting_time'][tls] += abs(r)

                if step % SPEED_SAMPLE_RATE == 0:
                    ids = traci.vehicle.getIDList()
                    if ids:
                        metrics['mean_speed_accum'].append(np.mean([traci.vehicle.getSpeed(v) for v in ids]))

            # --- D. APRENDIZAJE ---
            for tls in tls_ids:
                # Solo aprendemos si tuvimos la LIBERTAD de decidir.
                # Si fuimos forzados por MinGreenTime, técnicamente no tomamos la decisión
                # pero para simplificar Q-learning, actualizamos igual o podemos filtrar.
                # Aquí actualizamos siempre para propagar el valor del estado.
                
                reward = step_rewards[tls]
                metrics['agent_rewards'][tls] += reward
                
                next_s = get_state(tls, traci, controlled_lanes)
                if next_s not in local_q_tables[tls]:
                    local_q_tables[tls][next_s] = np.zeros(2)
                
                # Bellman Update
                old_val = local_q_tables[tls][current_states[tls]][actions[tls]]
                max_next = np.max(local_q_tables[tls][next_s])
                
                target = reward + args.gamma * max_next
                td_error = target - old_val
                new_val = old_val + args.alpha * td_error
                
                local_q_tables[tls][current_states[tls]][actions[tls]] = new_val
                metrics['squared_errors'].append(td_error ** 2)
                
                current_states[tls] = next_s

        metrics['total_arrived'] = traci.simulation.getArrivedNumber()
        traci.close()
    
    except Exception as e:
        print(f"Error Worker: {e}")
        try: traci.close()
        except: pass
        raise e
    finally:
        os.chdir(original_dir)
    
    return local_q_tables, metrics