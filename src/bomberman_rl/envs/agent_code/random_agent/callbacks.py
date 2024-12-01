import numpy as np

from ...actions import ActionSpace

def setup(self):
    self.rng = np.random.default_rng()


def act(self, game_state: dict):
    return ActionSpace.sample()