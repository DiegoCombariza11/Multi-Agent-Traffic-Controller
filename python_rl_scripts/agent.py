import numpy as np

class QLearningAgent:
    def __init__(self, agent_id, action_space_size=2, alpha=0.1, gamma=0.99):
        self.id = agent_id
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {} 

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        return self.q_table[state]

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.get_q_values(state))

    def update(self, state, action, reward, next_state):
        old_q = self.get_q_values(state)[action]
        next_max = np.max(self.get_q_values(next_state))
        
        target = reward + self.gamma * next_max
        td_error = target - old_q
        
        new_q = old_q + self.alpha * td_error
        self.q_table[state][action] = new_q
        
        return td_error