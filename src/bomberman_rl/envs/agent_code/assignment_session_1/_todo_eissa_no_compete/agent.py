import numpy as np
from collections import deque

from mpl_toolkits.axisartist.angle_helper import select_step_degree

import bomberman_rl.envs.settings as s

print = lambda *args, **kwargs: None
class Agent:
    """Stick to this interface to enable later competition.
    (Demonstration only - do not inherit)
    """

    def __init__(self):
        self.setup()

    def setup(self):
        """Before episode. Use this to setup action related state that is required to act on the environment.
        """
        self.deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, Down, Up, Right


    def is_in_bounds(self, x, y, rows, cols):
        """Checks if the client is inside the playground
        """
        return 0 <= x < rows and 0 <= y < cols


    def bfs(self, start, rows, cols, deltas, walls):
        """BFS to calculate distances to all reachable crates
        """
        distances = np.full((rows, cols), np.inf)  # Initialize distances as infinity
        distances[start] = 0
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in deltas:
                nx, ny = x + dx, y + dy
                if self.is_in_bounds(nx, ny, rows, cols) and walls[nx, ny] == 0 and distances[nx, ny] == np.inf:
                    distances[nx, ny] = distances[x, y] + 1
                    queue.append((nx, ny))
        Transpose_distances = distances.T
        return Transpose_distances

    def find_closest_crate(self, agent_position, crate_array, walls, deltas):
        """Find the minimum distance to any crate.
        """
        rows, cols = s.ROWS, s.COLS
        explosion_range = s.BOMB_POWER
        min_distance = float('inf')
        closest_crate = None

        # Calculate distances from the agent to all positions
        distances = self.bfs(agent_position, rows, cols, deltas, walls)
        print("distances : ", distances)

        for x in range(rows):
            for y in range(cols):
                if crate_array[x][y] == 1 and distances[x][y] < min_distance:
                    #print("x: ",x,"y: ",y,"distances[x][y] : ",distances[x][y])
                    min_distance = distances[x][y]
                    closest_crate = (y, x)

        return closest_crate, min_distance

    def decide_step(self, min_distance, agent_pos, crate_pos, agent_safe):
        """Decide the action to move the agent closer to the nearest crate.

        Parameters:
        - agent_pos: Tuple (x, y) representing the agent's position.
        - crate_pos: Tuple (x, y) representing the crate's position.
        - tile_is_free: Function that checks if a tile is free to move to.

        Returns:
        - Action string: "UP", "DOWN", "LEFT", "RIGHT", or "WAIT".
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
        """
        agent_x, agent_y = agent_pos
        crate_x, crate_y = crate_pos
        print(" min distance: ",min_distance)
        # Determine the direction to move
        if min_distance == 1 and agent_safe == True: # set a bomb
            return 5
        elif agent_x < crate_x:  # Move RIGHT
            return 1
        elif agent_x > crate_x:  # Move LEFT
            return 3
        elif agent_y < crate_y:  # Move DOWN
            return 2
        elif agent_y > crate_y:  # Move UP
            return 0

        # If no valid move is possible, WAIT
        return 4

    def calculate_explosion_radius(self, bomb_pos, walls, deltas):
        x_shape, y_shape = walls.shape
        explosion_mask = np.zeros((x_shape, y_shape), dtype=int)
        x,y = bomb_pos
        bomb_power = s.BOMB_POWER
        #print("bomb_power : ", bomb_power)
        for direction in deltas:
            for blast in range(bomb_power):
                #print("blast : ", blast)
                if blast == 0: # Do not modify the bomb_pos itself
                    continue
                else:
                    blast_pos_x, blast_pos_y = x + blast * direction[0], y + blast * direction[1]
                    #print("blast_pos_x ", blast_pos_x, " blast_pos_y: ", blast_pos_y)
                    if self.is_in_bounds(blast_pos_x, blast_pos_y, x_shape, y_shape):
                        if not walls[blast_pos_x, blast_pos_y] == 0:
                            #print("WALL")
                            continue
                        explosion_mask[blast_pos_x,blast_pos_y] = 1
                    else:
                        #print("Out of bounds")
                        continue
                #print(" explosion_mask : ",explosion_mask)
        return explosion_mask

    def is_tile_safe(self, pos, bombs, explosions, walls, crates):
        """Check if a tile is safe from explosions.

        Parameters:
        - position: Tuple (x, y) representing the tile position.
        - explosions: 2D array representing explosion danger zones.

        Returns:
        - True if the tile is safe, False otherwise.
        """
        #print("bombs map before : ", bombs)
        danger_map = bombs + explosions + walls + crates
        #print("danger map before : ", danger_map)
        # Add dynamic explosion zones for all bombs
        for x in range(bombs.shape[0]):  # Iterate over rows
            for y in range(bombs.shape[1]):
                bomb_pos = x,y
                #print("bomb_pos : ", bomb_pos)
                if bombs[x,y] != 0:
                    explosion_mask = self.calculate_explosion_radius(bomb_pos,walls, self.deltas)
                    danger_map += explosion_mask
        x, y = pos
        print("danger_map : ", danger_map)
        print("x : ", x, "y: ", y)
        print("danger_map[x, y] == 0 : ", danger_map[x, y] == 0)
        return danger_map[x, y] == 0, danger_map  # Safe if no bombs or explosions danger


    def find_nearest_safe_tile(self, agent_pos, danger_map, walls, deltas):
        """Find the nearest safe tile using BFS.

        Parameters:
        - start: Tuple (x, y) representing the agent's starting position.
        - explosions: 2D array representing explosion danger zones.
        - walls: 2D array representing walls.

        Returns:
        - nearest_safe_tile (x, y) coordinates or None if no safe tile is found.
        - distance to the safe tile or float('inf') if no safe tile is found.
        """

        # Initialize variables to track the nearest safe tile
        nearest_safe_tile = None
        min_distance = float('inf')
        distances = self.bfs(agent_pos, s.ROWS, s.COLS, deltas, walls)
        print("agent_pos : ", agent_pos)
        print("distances nearest safe tile : ", distances)
        # Iterate through the BFS distances and danger map to find the nearest safe tile
        rows, cols = danger_map.shape
        for y in range(cols):
            for x in range(rows):
                # Check if the tile is safe (danger map value is 0)
                if danger_map[x, y] == 0:
                    # If it's safe, compare the BFS distance
                    if distances[x, y] < min_distance:
                        min_distance = distances[x, y]
                        if (y,x) == agent_pos:
                            continue
                        nearest_safe_tile = (x, y)
        print("nearest_safe_tile : ", nearest_safe_tile, "min_distance : ", min_distance)
        return nearest_safe_tile, min_distance


    def act(self, state: dict, **kwargs) -> int:
        """Before step. Return action based on state.

        :param state: The state of the environment.
        """
        # Extract relevant state information
        walls = state['walls']
        crates = state['crates']
        bombs = state['bombs']  # List of bomb positions and timers
        explosions = state['explosions']  # Explosion map with blast radii
        coins = state['coins']
        opponents = state['opponents_pos']  # List of opponent positions
        self_pos = state['self_pos']
        my_position = next((x, y) for x, row in enumerate(self_pos) for y, val in enumerate(row) if val == 1)  # Assuming `self` contains (id, position, bombs)

        #check if agent is at a safe tile right now
        agent_safe,_ = self.is_tile_safe(my_position, bombs, explosions, walls, crates)
        print("agent safe: ", agent_safe)
        if(agent_safe):
            closest_crate, min_distance = self.find_closest_crate(my_position, crates, walls, self.deltas)
            print("best crate: ", closest_crate)
            action = self.decide_step(min_distance, my_position, closest_crate, agent_safe)
            print("action: ", action)
        else:
            _, danger_map = self.is_tile_safe(my_position, bombs, explosions, walls, crates)
            nearest_safe_tile, min_distance_safe_tile = self.find_nearest_safe_tile(my_position, danger_map, walls, self.deltas)
            print("nearest_safe_tile : ", nearest_safe_tile)
            action = self.decide_step(min_distance_safe_tile, my_position, nearest_safe_tile, agent_safe)
            print("action: ", action)
        return action
        #raise NotImplementedError()
        #wallah die waldfee