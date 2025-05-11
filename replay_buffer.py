from collections import deque, namedtuple
import random

# Define tuple representing a single transition in our environment
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Define class to model the replay memory
class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
