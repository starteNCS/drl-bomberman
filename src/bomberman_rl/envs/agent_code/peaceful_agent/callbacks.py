import numpy as np

from ...actions import ActionSpace, Actions

def setup(self):
    self.rng = np.random.default_rng()


def act(self, game_state: dict, **kwargs):
    self.logger.info("Pick action at random, but no bombs.")
    action = Actions.BOMB.value
    while action == Actions.BOMB.value:
        action = ActionSpace.sample()
    return action