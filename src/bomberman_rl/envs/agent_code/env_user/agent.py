from ..interface import Agent as Base

class Agent(Base):
    def act(self, game_state, **kwargs):
        return kwargs["env_user_action"]