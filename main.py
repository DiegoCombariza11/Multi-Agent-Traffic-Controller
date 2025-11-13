import traci
import sys
import os
import time
import google.generativeai as genai
from agent import RLAgent # ¡Importa la clase que acabamos de crear!

# --- 1. CONFIGURACIÓN INICIAL (DATOS DE TU COMPAÑERO DE SUMO) ---

# Ubicación del archivo de configuración de SUMO
SUMO_CONFIG_FILE = "path/to/your/simulation.sumocfg"

# IDs de los semáforos (deben coincidir con el .net.xml)
TLS_ID_LIST = ["J1", "J2", "J3", "J4"]

# Definición de las Fases (ACCIONES que los agentes pueden tomar)
# Esto es CRÍTICO.
# Asumimos: Fase 0 = N-S Verde, Fase 2 = E-O Verde
AGENT_ACTIONS = [0, 2] 

# Mapeo de Fases Verdes a Fases Amarillas (¡Para transiciones seguras!)
# Tu compañero de SUMO debe darte esto.
# {tls_id: {green_phase_index: yellow_phase_index}}
YELLOW_PHASE_MAP = {
    "J1": {0: 1, 2: 3}, # ej: En J1, la fase 0 (N-S) es seguida por la 1 (N-S Amarilla)
    "J2": {0: 1, 2: 3},
    "J3": {0: 1, 2: 3},
    "J4": {0: 1, 2: 3}
}

# Carriles de entrada para cada agente (LOS "OJOS")
# {tls_id: {"N": [lista_carriles], "S": [...], ...}}
AGENT_ENTRY_LANES = {
    "J1": {
        "N": ["lane_J1_N_0", "lane_J1_N_1"], "S": ["lane_J1_S_0"],
        "E": ["lane_J1_E_0"], "W": ["lane_J1_W_0"]
    },
    "J2": {
        "N": ["lane_J2_N_0"], "S": ["lane_J2_S_0"],
        "E": ["lane_J2_E_0"], "W": ["lane_J2_W_0"]
    },
    # ... (Configuración para J3 y J4) ...
    "J3": {}, 
    "J4": {}
}

# IDs de los carteles viales
VMS_IDS = ["vms_1", "vms_2"]

# --- 2. PARÁMETROS DE SIMULACIÓN Y LLM ---

# Tiempos del ciclo del semáforo (en segundos)
TIME_GREEN_PHASE = 15  # Duración mínima de una fase verde
TIME_YELLOW_PHASE = 4   # Duración de la fase amarilla
STEP_LENGTH = 1.0       # Dejar en 1.0 para 1 segundo por paso

# Configuración del LLM (Google Gemini)
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    print(f"Error al configurar el LLM: {e}")
    llm_model = None

# --- 3. FUNCIONES AUXILIARES ---

def start_simulation():
    """Inicia SUMO-GUI y TraCI."""
    # sumocfg: el archivo de config
    # step-length: 1 paso = 1 segundo (importante)
    sumo_cmd = ["sumo-gui", "-c", SUMO_CONFIG_FILE, "--step-length", str(STEP_LENGTH)]
    traci.start(sumo_cmd)

def call_llm_api(system_data):
    """
    Envía datos al LLM y actualiza los carteles VMS en SUMO.
    """
    if not llm_model:
        print("LLM no configurado. Saltando actualización de VMS.")
        return

    # El prompt "inteligente" que discutimos
    prompt = f"""
    Eres un experto en gestión de tráfico.
    Contexto de la Simulación (Telemetría en VIVO):
    ```json
    {system_data}
    ```
    Tu Tarea:
    Genera un mensaje corto (máx 10 palabras) para un cartel vial (VMS) 
    basado en la zona crítica y el tiempo de espera.
    """
    
    try:
        response = llm_model.generate_content(prompt)
        vms_message = response.text.strip().replace('"', '') # Limpiar la respuesta
        
        print(f"[LLM] Mensaje generado: {vms_message}")

        # Actualizar todos los carteles en la simulación
        for vms_id in VMS_IDS:
            traci.vms.setText(vms_id, vms_message)
            
    except Exception as e:
        print(f"Error al llamar al LLM: {e}")

# --- 4. BUCLE PRINCIPAL DE EJECUCIÓN ---

def run_simulation():
    """
    Orquesta la simulación completa: 
    Crea agentes, ejecuta el bucle, y maneja el aprendizaje.
    """
    start_simulation()
    
    # 1. Crear los 4 agentes
    agents = {}
    for tls_id in TLS_ID_LIST:
        agents[tls_id] = RLAgent(
            agent_id=f"Agent_{tls_id}",
            traffic_light_id=tls_id,
            actions=AGENT_ACTIONS,
            entry_lane_ids=AGENT_ENTRY_LANES.get(tls_id, {})
        )

    # 2. Variables para el bucle de control
    step = 0
    
    # Diccionarios para almacenar el estado de cada agente
    agent_state = {tls_id: None for tls_id in TLS_ID_LIST}
    agent_action = {tls_id: None for tls_id in TLS_ID_LIST}
    agent_time_in_phase = {tls_id: 0 for tls_id in TLS_ID_LIST}

    # Inicializar el estado de todos los agentes
    for tls_id, agent in agents.items():
        agent_state[tls_id] = agent.get_state()
        # Iniciar con una acción aleatoria (exploración inicial)
        agent_action[tls_id] = agent.choose_action(agent_state[tls_id])

    # 3. Bucle de Simulación Principal
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        
        # Incrementar el tiempo en la fase actual para cada agente
        for tls_id in TLS_ID_LIST:
            agent_time_in_phase[tls_id] += 1

        # --- LÓGICA DE CONTROL DE SEMÁFOROS (para cada agente) ---
        for tls_id, agent in agents.items():
            
            # ¿Está el agente actualmente en una fase amarilla?
            current_phase = traci.trafficlight.getPhase(tls_id)
            is_in_yellow_phase = current_phase in YELLOW_PHASE_MAP[tls_id].values()

            # --- A. Si está en FASE AMARILLA ---
            if is_in_yellow_phase:
                # Solo esperar a que termine el tiempo de amarillo
                if agent_time_in_phase[tls_id] >= TIME_YELLOW_PHASE:
                    # El amarillo terminó. Poner el verde que se eligió ANTES.
                    traci.trafficlight.setPhase(tls_id, agent_action[tls_id])
                    agent_time_in_phase[tls_id] = 0 # Reiniciar contador
            
            # --- B. Si está en FASE VERDE ---
            elif agent_time_in_phase[tls_id] >= TIME_GREEN_PHASE:
                # El tiempo mínimo de verde ha terminado.
                # Es hora de APRENDER y DECIDIR de nuevo.
                
                # 1. APRENDER (del ciclo que acaba de terminar)
                reward = agent.get_reward()
                next_state = agent.get_state()
                
                agent.update_q_table(
                    agent_state[tls_id],  # El estado de la decisión anterior
                    agent_action[tls_id], # La acción que se tomó
                    reward,               # La recompensa obtenida
                    next_state            # El estado resultante
                )
                
                # 2. DECIDIR (para el próximo ciclo)
                new_action = agent.choose_action(next_state)
                
                # 3. Guardar estado y acción para la próxima actualización
                agent_state[tls_id] = next_state
                agent_action[tls_id] = new_action
                
                # 4. ACTUAR (Iniciar la transición)
                if current_phase == new_action:
                    # La decisión es MANTENER el verde.
                    # Simplemente reinicia el contador y sigue.
                    agent_time_in_phase[tls_id] = 0 
                else:
                    # La decisión es CAMBIAR.
                    # Iniciar la fase amarilla de transición.
                    yellow_phase = YELLOW_PHASE_MAP[tls_id][current_phase]
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    agent_time_in_phase[tls_id] = 0 # Reiniciar contador
                    
        # --- FIN DEL BUCLE DE CONTROL ---
        
        # --- LLM TRIGGER (Ejecutar cada 60 segundos) ---
        if step % 60 == 0:
            print(f"[Paso {step}] Comprobando estado del sistema para LLM...")
            # Aquí recolectarías los datos de "call_llm_api"
            system_data = {
                "timestamp": step,
                "zona_critica": "J1 (Terminal)", # (Ejemplo)
                "tiempo_espera_promedio_global": 120 # (Ejemplo, tú lo calcularías)
            }
            # call_llm_api(system_data) # Descomentar para activar
            
    # 4. Fin de la Simulación
    print("Simulación terminada. Guardando Q-Tables...")
    for agent in agents.values():
        agent.guardar_q_table()
        
    traci.close()
    print("Listo.")

# --- 5. PUNTO DE ENTRADA ---
if __name__ == "__main__":
    
    # Asegurarse de que las variables de entorno (API Key) están cargadas
    if not os.environ.get("GOOGLE_API_KEY"):
        print("¡ADVERTENCIA! La variable de entorno GOOGLE_API_KEY no está configurada.")
        
    run_simulation()