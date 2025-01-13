from collections import deque, namedtuple
import random


Replay = namedtuple('Replay', ['old_state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, old_state, action, reward, new_state, done):
        """Save a Replay"""
        self.memory.append(Replay(old_state, action, reward, new_state, done))

    def get_sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)
