import numpy as np

from bomberman_rl import Actions

class Agent:
    def __init__(self):
        self.setup()
        self.setup_training()


    def setup(self):
        """
        Get ready to act.
        """
        self.rng = np.random.default_rng()
        self.n_actions = len(Actions)


    def setup_training(self):
        """
        Get ready to train.
        """
        pass


    def act(self, state):
        """
        Act: pick an action based on the state.
        """
        return np.argmax(self.rng.random(self.n_actions))
    

    def game_events_occurred(
        self,
        old_state: dict,
        self_action: str,
        new_state: dict,
        events: list[str],
    ):
        """
        Train: update the model after step in the environment.

        :param old_state: The state the agent was in before.
        :param self_action: The action the agent performed.
        :param new_state: The state the agent is in now.
        :param events: The events that occurred when going from `old_state` to `new_state`.
        """
        pass
        

    def end_of_round(self):
        """
        Callback after episode ended.
        """
        pass
