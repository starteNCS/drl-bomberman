from bomberman_rl import events as e

# Custom events
SCORE_INCREASED = "SCORE_INCREASED"

class LearningAgent:
    def __init__(self):
        self.setup()
        self.setup_training()

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        pass

    def act(self, state: dict) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        raise NotImplementedError()

    def setup_training(self):
        """
        Before episode (optional). Use this to setup additional learning related state e.g. a replay memory, hyper parameters etc.
        """
        pass

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
        custom_events = self._custom_events(old_state, new_state)
        reward = self._shape_reward(events + custom_events)

    def end_of_round(self):
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        pass


    def _custom_events(self, old_state, new_state):
        """
        Just an idea!
        """
        custom_events = []
        if old_state["score"] < new_state["score"]:
            custom_events.append(SCORE_INCREASED)
        return custom_events

    def _shape_reward(self, events: list[str]) -> float:
        """
        Just an idea!
        """
        reward_mapping = {
            SCORE_INCREASED: 5,
            e.MOVED_DOWN: .1,
            e.MOVED_LEFT: .1,
            e.MOVED_UP: .1,
            e.MOVED_RIGHT: .1
        }
        return sum([reward_mapping.get(event, 0) for event in events])