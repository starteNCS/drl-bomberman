import numpy as np

from ...actions import ActionSpace, Actions
from ..interface import Agent as Base

class Agent(Base):
    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, game_state: dict, **kwargs):
        action = Actions.BOMB.value
        while action == Actions.BOMB.value:
            action = ActionSpace.sample()
        return action