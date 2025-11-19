import traci
import numpy as np
import pandas as pd
import sys
import os
import math

# --- CONFIGURACIÓN ---
SUMO_CMD = ["sumo", "-c", "./sumoData/SumoConfigSim.sumocfg", "--no-step-log", "true", "--waiting-time-memory", "1000"]
# Usa "sumo-gui" si quieres ver la simulación visualmente.

# Parámetros de Q-Learning
ALPHA = 0.1       # Tasa de aprendizaje
GAMMA = 0.99      # Factor de descuento
EPSILON = 0     # Exploración inicial
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1     
SIMULATION_STEPS_PER_ACTION = 5 


class QLearningAgent:
    def __init__(self, agent_id, action_space_size):
        self.id = agent_id
        self.q_table = {} 
        self.action_space_size = action_space_size
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size) 
        else:
            return np.argmax(self.get_q_values(state)) 

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_values(state)[action]
        next_max = np.max(self.get_q_values(next_state))
        
        target = reward + self.gamma * next_max
        td_error = target - old_q
        new_q = old_q + self.alpha * td_error
        
        self.q_table[state][action] = new_q
        return td_error 


def get_state(tls_id, controlled_lanes):

    state = ""
    for lane in controlled_lanes[tls_id]:
        q = traci.lane.getLastStepHaltingNumber(lane)
        if q < 3: state += "0"      
        elif q < 10: state += "1"   
        else: state += "2"        
    return state

def get_reward(tls_id, controlled_lanes):

    waiting_time = 0
    for lane in controlled_lanes[tls_id]:
        waiting_time += traci.lane.getLastStepHaltingNumber(lane)
    return -1 * waiting_time

def get_total_phases(tls_id):

    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    return len(logic.phases)


def run():

    traci.start(SUMO_CMD)
    tls_ids = traci.trafficlight.getIDList() 
    
    controlled_lanes = {}
    num_phases = {}
    agents = {}
    
    print(f"Se detectaron {len(tls_ids)} semáforos controlables.")
    
    for tls in tls_ids:
        controlled_lanes[tls] = list(set(traci.trafficlight.getControlledLanes(tls)))
        num_phases[tls] = get_total_phases(tls)
        
        agents[tls] = QLearningAgent(tls, action_space_size=2)

    traci.close()

    global_metrics = [] 

    for episode in range(EPISODES):
        print(f"Iniciando Episodio {episode + 1}/{EPISODES}")
        traci.start(SUMO_CMD)
        
        step = 0
        episode_reward = 0
        episode_squared_errors = []
        
        current_states = {tls: get_state(tls, controlled_lanes) for tls in tls_ids}
        
        while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:
            
            actions = {}
            
            for tls in tls_ids:
                actions[tls] = agents[tls].choose_action(current_states[tls])
            

            for tls in tls_ids:
                if actions[tls] == 1:

                    current_p = traci.trafficlight.getPhase(tls)
                    next_p = (current_p + 1) % num_phases[tls]
                    traci.trafficlight.setPhase(tls, next_p)
                else:

                    pass
            step_reward_accum = {tls: 0 for tls in tls_ids}
            for _ in range(SIMULATION_STEPS_PER_ACTION):
                traci.simulationStep()
                step += 1

                for tls in tls_ids:
                    step_reward_accum[tls] += get_reward(tls, controlled_lanes)


            for tls in tls_ids:
                reward = step_reward_accum[tls]
                next_state = get_state(tls, controlled_lanes)

                td_error = agents[tls].learn(current_states[tls], actions[tls], reward, next_state)
                
                episode_squared_errors.append(td_error ** 2)
                episode_reward += reward # Suma global para evaluar la red completa
                
                current_states[tls] = next_state

        traci.close()

        rms = math.sqrt(np.mean(episode_squared_errors)) if episode_squared_errors else 0
        avg_epsilon = agents[tls_ids[0]].epsilon 
        
        global_metrics.append({
            "Episode": episode + 1, 
            "Global_Utility_Estimate": episode_reward, 
            "RMS_Error": rms,
            "Epsilon": avg_epsilon
        })
        
        print(f"  -> Recompensa Global: {episode_reward:.2f} | RMS: {rms:.4f}")

        for tls in tls_ids:
            agents[tls].epsilon = max(MIN_EPSILON, agents[tls].epsilon * EPSILON_DECAY)


    df_metrics = pd.DataFrame(global_metrics)
    df_metrics.to_csv("multi_agent_metrics.csv", index=False)
    print("\nMétricas guardadas en 'multi_agent_metrics.csv'")

    all_q_data = []
    for tls, agent in agents.items():
        for s, values in agent.q_table.items():
            row = {"Agent_ID": tls, "State": s}
            for i, v in enumerate(values):
                row[f"Action_{i}"] = v
            all_q_data.append(row)
            
    df_q = pd.DataFrame(all_q_data)
    df_q.to_csv("multi_agent_q_tables.csv", index=False)
    print("Tablas Q combinadas guardadas en 'multi_agent_q_tables.csv'")

if __name__ == "__main__":
    run()