import numpy as np


def setup(self):
    np.random.seed()


def act(agent, game_state: dict, **kwargs):
    agent.logger.info("Pick action at random, but no bombs.")
    return np.random.choice(["RIGHT", "LEFT", "UP", "DOWN"])
