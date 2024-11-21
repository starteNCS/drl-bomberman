import numpy as np

from bomberman_rl import Actions

class Agent:
    def __init__(self):
        self.setup()
        self.setup_training()


    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment e.g. loading an (already trained) model.
        """
        self.rng = np.random.default_rng()


    def setup_training(self):
        """
        Before episode (optional). Use this to setup additional training related state e.g. a replay memory, learning rates etc.
        """
        pass


    def act(self, state):
        """
        Before step: return action based on state.

        :param state: The state of the environment.
        """
        return np.argmax(self.rng.random(len(Actions)))
    

    def game_events_occurred(
        self,
        old_state: dict,
        self_action: str,
        new_state: dict,
        events: list[str],
    ):
        """
        After step in environment (optional). Use this e.g. for model training.

        :param old_state: Old state of the environment.
        :param self_action: Performed action.
        :param new_state: New state of the environment.
        :param events: Events that occurred during step. These might be used for Reward Shaping.
        """
        pass
        

    def end_of_round(self):
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        pass
