from collections import deque, namedtuple
import random


Replay = namedtuple('Replay', ['old_state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, old_state, action, reward, new_state, done):
        """
        Saves a Replay

        If the buffer would overflow, we remove the first (oldest) item from the buffer
        and push the new item to the replay buffer
        """

        if len(self.memory) == self.memory.maxlen:
            self.memory.popleft()  # pop the first (oldest) element of the replay buffer, once buffer is full

        self.memory.append(Replay(old_state, action, reward, new_state, done))

    def get_sample(self, batch_size):
        """
        Returns a random sample of the replay buffer
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)
