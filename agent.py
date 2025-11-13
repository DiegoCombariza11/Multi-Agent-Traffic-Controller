import random
import pickle
import os
import traci

class RLAgent:
    """
    This class implements a Reinforcement Learning Agent (Tabular Q-Learning)
    to control a traffic light in SUMO.
    """
    def __init__(self, agent_id, traffic_light_id, actions, entry_lane_ids):
        
        # --- Identification and Configuration ---
        self.id = agent_id              # e.g., "Agent_J1"
        self.tls_id = traffic_light_id  # e.g., "J1" (ID in SUMO)
        self.actions = actions          # e.g., [0, 2] (Phases N-S and E-W)
        self.entry_lanes = entry_lane_ids # dict of lists, e.g., {"N": ["lane_N_0", ...], "E": [...]}

        # --- Q-Learning Parameters ---
        self.alpha = 0.1                # Learning rate
        self.gamma = 0.9                # Discount factor (importance of future rewards)
        self.epsilon = 1.0              # Initial exploration rate
        self.epsilon_min = 0.05         # Minimum exploration rate
        self.epsilon_decay = 0.9995     # Epsilon decay factor

        # --- Q-Table (The Brain) ---
        self.q_table_file = f"q_table_{self.id}.pkl"
        self.q_table = self.load_q_table()


    def load_q_table(self):
        """
        Loads the Q-Table from a .pkl file if it exists.
        If not, initializes an empty table (dictionary).
        """
        if os.path.exists(self.q_table_file):
            print(f"[{self.id}] Loading existing Q-Table...")
            with open(self.q_table_file, 'rb') as f:
                self.q_table = pickle.load(f)
            # If we load a table, we are in "execution" mode, so don't explore.
            self.epsilon = self.epsilon_min 
            return self.q_table
        else:
            print(f"[{self.id}] No Q-Table found. Starting a new one.")
            return {}

    def save_q_table(self):
        """Saves the current Q-Table to a .pkl file."""
        print(f"[{self.id}] Saving Q-Table to {self.q_table_file}...")
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def _discretize_queue(self, vehicle_count):
        """Converts a number of vehicles into a simple category."""
        if vehicle_count < 5:
            return "low"
        elif vehicle_count < 15:
            return "medium"
        else:
            return "high"

    def get_state(self):
        """
        Defines the current state. This is one of the most important steps.
        A simple and effective state combines the current phase and the queues.
        RETURNS A TUPLE (to be used as a dictionary key).
        """
        # 1. Current traffic light phase
        # (We use the phase index, e.g., 0 or 2)
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # 2. Measure queues on each entry lane
        # We use getLastStepHaltingNumber to count stopped vehicles
        
        # Sum the queues from all access directions
        queue_north = sum(traci.lane.getLastStepHaltingNumber(c) for c in self.entry_lanes.get("N", []))
        queue_south = sum(traci.lane.getLastStepHaltingNumber(c) for c in self.entry_lanes.get("S", []))
        queue_east = sum(traci.lane.getLastStepHaltingNumber(c) for c in self.entry_lanes.get("E", []))
        queue_west = sum(traci.lane.getLastStepHaltingNumber(c) for c in self.entry_lanes.get("O", []))
        
        # 3. Discretize the state
        state = (
            current_phase,
            self._discretize_queue(queue_north),
            self._discretize_queue(queue_south),
            self._discretize_queue(queue_east),
            self._discretize_queue(queue_west)
        )
        
        return state

    def choose_action(self, state):
        """
        Chooses an action using the Epsilon-Greedy policy.
        Explores (random action) or Exploits (best action from Q-Table).
        """
        # Epsilon decay (to explore less as we learn)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Decide to Explore vs. Exploit
        if random.random() < self.epsilon:
            # --- EXPLORE ---
            # Choose a random action from the possible ones
            return random.choice(self.actions)
        else:
            # --- EXPLOIT ---
            # Choose the best known action for this state
            # Filter the Q-Table for this state
            q_values = {action: self.q_table.get((state, action), 0.0) for action in self.actions}
            
            if not q_values:
                return random.choice(self.actions) # Fallback if the state is new
            
            # Return the action (key) with the highest Q-Value (value)
            best_action = max(q_values, key=q_values.get)
            return best_action

    def get_reward(self):
        """
        Calculates the reward.
        COMPLETE AND STABLE FUNCTION: Penalizes waiting, 
        giving MORE weight (punishment) to stopped buses.
        """
        total_penalty = 0.0
        
        # Iterate over all entry lanes this agent controls
        for direction, lane_list in self.entry_lanes.items():
            for lane in lane_list:
                # Get the IDs of vehicles on this lane
                vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane)
                
                for v_id in vehicles_on_lane:
                    # If the vehicle is stopped (speed < 0.1 m/s)
                    if traci.vehicle.getSpeed(v_id) < 0.1:
                        
                        # Here is the prioritization!
                        if traci.vehicle.getTypeID(v_id) == "bus":
                            # HIGH penalty for a stopped bus
                            total_penalty += 5.0
                        else:
                            # Normal penalty for a stopped car
                            total_penalty += 1.0

        # The reward is the NEGATIVE of the penalty.
        # We want to maximize the reward, which means minimizing the penalty.
        return -total_penalty


    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-Table using the Bellman equation (Q-Learning formula).
        """
        
        # 1. Get the current Q-Value (what we knew before)
        old_q_value = self.q_table.get((state, action), 0.0)
        
        # 2. Find the maximum possible Q-Value from the *next state*
        # (This is the "expected future reward")
        max_future_q = max([self.q_table.get((next_state, a), 0.0) for a in self.actions], default=0.0)
        
        # 3. The Q-Learning formula
        # new_Q = old_Q + alpha * (reward + gamma * max_future_Q - old_Q)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        
        # 4. Save the new value in the table
        self.q_table[(state, action)] = new_q_value