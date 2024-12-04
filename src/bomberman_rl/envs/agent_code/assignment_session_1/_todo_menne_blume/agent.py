import heapq

import numpy as np

from bomberman_rl import Actions

ACT_DIR = {
    0: (-1, 0),  # UP
    1: (0, 1),  # RIGHT
    2: (1, 0),  # DOWN
    3: (0, -1),  # LEFT
    4: (0, 0),  # Wait
    5: (0, 0),  # Bomb
}


class MasterRuleBasedAgent:
    def __init__(self):
        self.setup()

    def setup(self):
        pass

    def act(self, state: dict) -> int:
        self_pos = state["self_pos"]
        walls = state["walls"]
        bombs = state["bombs"]
        explosions = state["explosions"]
        crates = state["crates"]
        coins = state["coins"]

        r, c = np.where(self_pos == 1)
        pos = (c[0].item(), r[0].item())

        # calculate dangerous zones
        bomb_danger_zones = self.get_bomb_danger_zones(bombs, walls, crates)

        free_space = walls + explosions + crates
        free_space[free_space > 0] = -1
        free_space[free_space == 0] = 1
        free_space[free_space == -1] = 0

        free_space_danger = walls + bombs + explosions + crates + bomb_danger_zones
        free_space_danger[free_space_danger > 0] = -1
        free_space_danger[free_space_danger == 0] = 1
        free_space_danger[free_space_danger == -1] = 0

        if bomb_danger_zones[pos[1], pos[0]] == 1:
            action = self.look_for_targets_prim(free_space, pos, free_space_danger)
        elif (np.sum(crates) + np.sum(coins)) == 0:
            return Actions.WAIT.value
        elif np.sum(coins) > 0 and np.max(coins - bomb_danger_zones) > 0:
            action = self.look_for_targets_prim(
                free_space_danger, pos, coins - bomb_danger_zones
            )
            if action is Actions.WAIT.value:
                action = self.look_for_targets_prim(
                    free_space_danger, pos, crates - bomb_danger_zones
                )
        elif np.max(crates - bomb_danger_zones) > 0:
            action = self.look_for_targets_prim(
                free_space_danger, pos, crates - bomb_danger_zones, search_crates=True
            )
        else:
            return Actions.WAIT.value

        act_pos = tuple(map(lambda i, j: i + j, pos, ACT_DIR.get(action)))

        if crates[(act_pos[1], act_pos[0])] == 1:
            return Actions.BOMB.value

        return action

    def look_for_targets_prim(
        self, free_space, start, targets, logger=None, search_crates=False
    ):
        """
        Find direction of closest target using Prim's algorithm to form paths in a minimum spanning tree.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards the closest target or towards tile closest to any target.
            :param search_crates:
        """
        if len(targets) == 0:
            return None

        def get_neighbors(coord):
            """Returns valid neighbors within free space."""
            x, y = coord
            neighbors = [
                (nx, ny)
                for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                if 0 <= nx < free_space.shape[0]
                and 0 <= ny < free_space.shape[1]
                and free_space[ny, nx]
            ]
            return neighbors

        # convert (17, 17) array to list of coordinates
        targets = [tuple(coord) for coord in np.argwhere(targets > 0)]
        targets = [t[::-1] for t in targets]

        # Priority queue for edges
        pq = []
        heapq.heappush(pq, (0, start, start))  # (weight, current_node, parent_node)
        visited = set()
        parent_dict = {}
        best = start
        best_dist = np.inf

        while pq:
            weight, current, parent = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)
            parent_dict[current] = parent

            # Update the best target if the current node is closer
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d < best_dist:
                best = current
                best_dist = d

            # Add neighbors to the priority queue
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    heapq.heappush(
                        pq, (1, neighbor, current)
                    )  # All edges have weight 1

        if logger:
            logger.debug(f"Suitable target found at {best}")

        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start:
                if start[0] < current[0] and start[1] == current[1]:
                    return Actions.DOWN.value
                if start[0] > current[0] and start[1] == current[1]:
                    return Actions.UP.value
                if start[0] == current[0] and start[1] < current[1]:
                    return Actions.RIGHT.value
                if start[0] == current[0] and start[1] > current[1]:
                    return Actions.LEFT.value
                if search_crates:
                    return Actions.BOMB.value
                return Actions.WAIT.value
            current = parent_dict[current]

    def get_bomb_danger_zones(self, bombs, walls, crates):
        """
        Calculate all positions affected by bomb explosions.

        Args:
            bombs (np.array): 17x17 array of bomb locations.
            walls (np.array): 17x17 array of walls.
            crates (np.array): 17x17 array of crates.

        Returns:
            set: Set of positions affected by bombs.
        """
        danger_zones = np.zeros((17, 17))
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Up, Down

        for x in range(17):
            for y in range(17):
                if bombs[x, y]:
                    danger_zones[x, y] = 1  # Bomb position itself is dangerous

                    # Expand in all directions up to a distance of 3
                    for dx, dy in directions:
                        for distance in range(1, 4):  # Bomb has a reach of 3
                            nx, ny = x + dx * distance, y + dy * distance
                            if 0 <= nx < 17 and 0 <= ny < 17:
                                if walls[nx, ny]:  # Stop at walls
                                    break
                                if crates[nx, ny]:  # Stop at crates
                                    danger_zones[nx + 1, ny] = 1
                                    danger_zones[nx - 1, ny] = 1
                                    danger_zones[nx, ny + 1] = 1
                                    danger_zones[nx, ny - 1] = 1
                                danger_zones[nx, ny] = 1
                            else:
                                break

        return danger_zones
