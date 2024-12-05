import numpy as np
from bomberman_rl import Actions
from collections import deque

def shortest_path(grid: np.ndarray, start: tuple, end: tuple):
    """
    Returns shortest path from start to end. Walkable cells are 0 only!
    """
    # valid starts?
    if grid[start] != 0 or grid[end] != 0:
        return []
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS:
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)
    
    while queue:
        current_pos, path = queue.popleft()
        if current_pos == end:
            return path
        for direction in directions:
            neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            if (0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                grid[neighbor] == 0 and
                neighbor not in visited):
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


def get_first_action(path: list[tuple[int,int]]):
    """
    Returns the first Action to achieve the given path.
    """
    if len(path) < 2:   # no legit path
        return Actions.WAIT.value
    
    start = path[0]
    next_step = path[1]

    col_diff = next_step[0] - start[0]
    row_diff = next_step[1] - start[1]

    # UP; DOWN; LEFT; RIGHT
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    direction_values = [Actions.UP.value, Actions.DOWN.value, Actions.LEFT.value, Actions.RIGHT.value]

    for i,dir in enumerate(directions):
        if dir[0] == col_diff and dir[1] == row_diff:
            return direction_values[i]
    raise RuntimeError('this should happen in get_first_action')



def shortest_path_to_value(grid: np.ndarray, start: tuple, search_value: int):
    """
    Returns shortest Path to the nearest cell with value search_value. Not passable cells shall be 1.
    """
    # UP; DOWN; LEFT; RIGHT
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    wall_value = 1
    
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)
    
    while queue:
        current_pos, path = queue.popleft()
        if grid[current_pos] == search_value:
            return path
        for direction in directions:
            neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            if (0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                neighbor not in visited and
                grid[neighbor] != wall_value):       # TODO check wall_value
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


class Agent:
    """
    Stick to this interface to enable later competition.
    (Demonstration only - do not inherit)
    """
    def __init__(self):
        self.setup()

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        # print('Hello World!')
        self.currently_in_own_explosion = False

    def act(self, state: dict, **kwargs) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        ACTION_REASON = '-'
        
        # move to coin or search for them in crates
        action, remaining_steps = self.coin_finder_policy(state)
        ACTION_REASON = 'moving to coin'
        if remaining_steps <= 0 or remaining_steps >= 15:   # 15 is max steps to coin
            action, remaining_steps = self.crate_destroyer_policy(state)
            ACTION_REASON = 'moving to crate'
            # place bomb if near crate
            if action == Actions.WAIT.value:
                action = Actions.BOMB.value
                self.currently_in_own_explosion = True
                ACTION_REASON = 'next to crate'

        # dodge explosions if neccessary
        dodge_action, dodge_steps = self.dodge_explosions_policy(state)
        if dodge_steps > 0:
            action = dodge_action
            ACTION_REASON = 'dodging'
        else:
            self.currently_in_own_explosion = False

        # check if next move runs into explosion
        if self.running_into_explosion(state, action):
            action = Actions.WAIT.value
            ACTION_REASON = 'running into explosion'

        ## debugging TODO
        debug_actions = {Actions.UP.value: 'UP', Actions.DOWN.value:'DOWN', Actions.LEFT.value:'LEFT', Actions.RIGHT.value:'RIGHT', Actions.WAIT.value:'WAIT', Actions.BOMB.value: 'BOMB'}
        # print('---------------------------')
        # print(debug_actions[action])
        # print(ACTION_REASON)
        # print(self.currently_in_own_explosion )
        return action
    

    def crate_destroyer_policy(self, state: dict) -> tuple[Actions, int]:
        """Returns the next action to get to the nearest crate and the remaining amount of steps to do so."""
        crate_value = 2
        find_crate_grid = state['crates'] * crate_value + state['walls'] + state['opponents_pos']
        agent_pos = np.argmax(state['self_info']['position'])
        agent_pos_tuple = (int(agent_pos/17), int(agent_pos) - int(agent_pos/17)*17)
        path = shortest_path_to_value(find_crate_grid, agent_pos_tuple, crate_value)[:-1]
        return get_first_action(path), len(path)-1
    

    def coin_finder_policy(self, state: dict) -> tuple[Actions, int]:
        """Returns the next action to get to the nearest coin and the remaining amount of steps to do so."""
        coin_value = 2
        find_coin_grid = state['coins'] * coin_value + state['walls'] + state['opponents_pos'] + state['crates']
        agent_pos = np.argmax(state['self_info']['position'])
        agent_pos_tuple = (int(agent_pos/17), int(agent_pos) - int(agent_pos/17)*17)
        path = shortest_path_to_value(find_coin_grid, agent_pos_tuple, coin_value)
        return get_first_action(path), len(path)-1
    

    def get_future_explosion_grid(self, state: dict):
        """
        Returns the grid for future explosions. 2 = wall, 1 = explosion.
        A specified bomb can be ignored with ignore_bomb_at.
        """
        propagation_radius = 3  # bombs propogate 3 steps to each directions
        bomb_grid = state['bombs']
        bomb_grid[bomb_grid > 0] = 1
        grid = bomb_grid + 2*(state['walls'] + state['crates'] + state['opponents_pos']) 
        coordinates = np.argwhere(grid == 1)
        # simulate the exploding bombs
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for coord in [tuple(coord) for coord in coordinates]:
            for dr, dc in directions:
                row, col = coord
                for _ in range(propagation_radius):
                    row += dr
                    col += dc
                    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
                        break
                    if grid[row, col] == 2:
                        break
                    grid[row, col] = 1
        return grid

    def dodge_explosions_policy(self, state: dict) -> tuple[Actions, int]:
        """Returns the next action to get quickly out of future explosions and the remaining amount of steps to do so."""
        grid = self.get_future_explosion_grid(state) - 1
        grid[grid < 0] = 2

        agent_pos = np.argmax(state['self_info']['position'])
        agent_pos_tuple = (int(agent_pos/17), int(agent_pos) - int(agent_pos/17)*17)
        path = shortest_path_to_value(grid, agent_pos_tuple, 2)
        return get_first_action(path), len(path)-1

    def running_into_explosion(self, state: dict, action: Actions) -> bool:
        """Checks if the action leads into an explosion."""
        if self.currently_in_own_explosion:
            return False
        if action == Actions.WAIT.value or action == Actions.BOMB.value: return
        directions_dict = {Actions.UP.value:(0, -1), Actions.DOWN.value:(0, 1), Actions.LEFT.value:(-1, 0), Actions.RIGHT.value:(1, 0)}
        agent_pos = np.argmax(state['self_info']['position'])
        agent_pos = np.array((int(agent_pos/17), int(agent_pos) - int(agent_pos/17)*17))
        next_pos = agent_pos + np.array(directions_dict[action])
        return state['explosions'][*next_pos] > 0 or self.get_future_explosion_grid(state)[*next_pos] == 1

