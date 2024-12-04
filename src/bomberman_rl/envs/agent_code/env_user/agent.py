from ..interface import RLAgent

class Agent(RLAgent):
    def act(self, game_state, **kwargs):
        """
        Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        """
        return kwargs["env_user_action"]