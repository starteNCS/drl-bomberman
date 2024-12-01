import numpy as np

from bomberman_rl import Actions, ActionSpace


class RandomAgent:
    def __init__(self):
        self.setup()

    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, state: dict) -> int:
        action = Actions.BOMB.value
        while action == Actions.BOMB.value:
            action = ActionSpace.sample()
        return action
