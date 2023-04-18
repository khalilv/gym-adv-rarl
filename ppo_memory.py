import numpy as np
class PPOMemory(object):
    def __init__(self, batch_size):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self): 
        n_states = len(self.states)
        batch_start = range(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        indices = indices.tolist()
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), self.actions, self.probabilities, self.values,\
        self.rewards, self.dones, batches
    
    def store_memory(self, state, action, probabilities, values, reward, next_state, done):
        self.states.append(state)
        self.probabilities.append(probabilities)
        self.values.append(values)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.next_state = []
        self.dones = []
    
    


