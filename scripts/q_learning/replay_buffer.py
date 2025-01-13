from collections import deque, namedtuple
import random


Replay = namedtuple('Replay', ['old_state', 'action', 'reward', 'next_state'])


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, old_state, action, new_state, reward):
        """Save a Replay"""
        self.memory.append(Replay(old_state, action, new_state, reward))

    def get_sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)
