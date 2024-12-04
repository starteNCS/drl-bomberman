import numpy as np
from gymnasium.spaces import Space, Discrete, MultiBinary, MultiDiscrete, Sequence, Dict, Text

from . import settings as s


def _multi_discrete_space(n=1):
    """
    Arena shaped space
    """
    if n == 1:
        return MultiBinary([s.COLS, s.ROWS])
    else:
        return MultiDiscrete(np.ones((s.COLS, s.ROWS)) * n)

def observation_space():
    SInt = Discrete(2 ** 20)
    SWalls = _multi_discrete_space()
    SCrates = _multi_discrete_space()
    SCoins = _multi_discrete_space()
    SBombs = _multi_discrete_space(s.BOMB_TIMER + 1) # 0 = no bomb
    SExplosions = _multi_discrete_space(15)
    SAgentPos = _multi_discrete_space()
    SOpponentsPos = _multi_discrete_space()
    SAgent = Dict({
        "score": SInt,
        "bombs_left": Discrete(2),
        "position": _multi_discrete_space()
    })
    SOpponents = Sequence(SAgent)
    return Dict({
        "round": SInt,
        "step": SInt,
        "walls": SWalls,
        "crates": SCrates,
        "coins": SCoins,
        "bombs": SBombs,
        "explosions": SExplosions,
        "self_pos": SAgentPos,
        "opponents_pos": SOpponentsPos,
        "self_info": SAgent,
        "opponents_info": SOpponents
    })


def delegate2gym(state):

    def _agent_delegate2gym(agent, pos):
        return {
            "score": agent[1],
            "bombs_left": int(agent[2]),
            "position": pos
        }
    
    if state is None:
        return None
    
    walls = (state["field"] == - 1).astype("int16")
    crates = (state["field"] == 1).astype("int16")

    coins = np.zeros(state["field"].shape, dtype="int16")
    if len(state["coins"]):
        coins[*zip(*state["coins"])] = 1

    bombs = np.zeros(state["field"].shape, dtype="int16")
    if len(state["bombs"]):
        pos, timer = zip(*state["bombs"])
        pos = list(pos)
        timer_feature = s.BOMB_TIMER - np.array(list(timer))
        bombs[*zip(*pos)] = timer_feature

    self_pos = np.zeros(state["field"].shape, dtype="int16")
    _, _, _, pos = state["self"]
    self_pos[*pos] = 1

    opponents_pos = np.zeros(state["field"].shape, dtype="int16")
    if len(state["others"]):
        positions = [pos for _, _, _, pos in state["others"]]
        opponents_pos[*zip(*positions)] = 1

    self_info = _agent_delegate2gym(state["self"], self_pos)
    
    single_opponents_pos = []
    for _, _, _, pos in state["others"]:
        single_opponent_pos = np.zeros(state["field"].shape, dtype="int16")
        single_opponent_pos[*pos] = 1
        single_opponents_pos.append(single_opponent_pos)
    opponents_info = tuple([_agent_delegate2gym(agent, pos) for agent, pos in zip(state["others"], single_opponents_pos)])

    return {
        "round": state["round"],
        "step": state["step"],
        "walls": walls,
        "crates": crates,
        "coins": coins,
        "bombs": bombs,
        "explosions": state["explosion_map"],
        "self_pos": self_pos,
        "opponents_pos": opponents_pos,
        "self_info": self_info,
        "opponents_info": opponents_info
    }