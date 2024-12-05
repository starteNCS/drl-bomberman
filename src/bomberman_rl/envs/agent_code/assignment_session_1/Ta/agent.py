import numpy as np
from collections import deque
from bomberman_rl import Actions
from bomberman_rl.envs.settings import BOMB_TIMER, COLS, ROWS

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
        self.rng = np.random.default_rng()
        self.prev_field = None

    def act(self, state: dict, **kwargs) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        #print(state)
        # Gather information about the game state
        walls = state["walls"]
        crates = state["crates"]
        self_info = state["self_info"]
        score = self_info["score"]
        bombs_left = self_info["bombs_left"]
        coins = state["coins"]
        #print(coins)
        # Get individual coin locations
        coins_coordinates = np.nonzero(coins == 1)
        coins_coordinates = list(zip(coins_coordinates[0], coins_coordinates[1]))
        bombs = state["bombs"]
        explosions = state["explosions"]
        self_pos = state["self_pos"]
        opponents_pos = state["opponents_pos"]
        # Get and combine each opponents grid coordinates
        opponents_pos_coordinates = np.nonzero(opponents_pos == 1)
        opponents_pos_coordinates = list(zip(opponents_pos_coordinates[0], opponents_pos_coordinates[1]))
        # Get the own position coordinates
        self_pos_coordinates = np.nonzero(self_pos == 1) # [][] falls doch iwie mehrmals ne 1 drin is, geht aber eig nicht
        #print(self_pos_coordinates)

        # create a grid of the unpassable objects, walls, crates and placed bombs (, players?)
        unpassable_fields = bombs + walls + crates
        # without the own field because on there you can always walk away from even if there is a bomb
        unpassable_fields[self_pos_coordinates] = 0
        #print(np.transpose(unpassable_fields))
        
        # Mark all spots where an explosion will occur and combine that with the places where an explosion is
        bomb_future_explosion_fields = bombs # WIESO KLAPPT DAS NICHT!?!?!??!?!?!?!, WIESO NUR DAS DRUNTER
        #print(np.transpose(bomb_future_explosion_fields))
        i = 0
        bomb_future_explosion_fields = np.zeros_like(bombs)  # Initialize an empty grid
    
        for x in range(COLS):
            for y in range(ROWS):
                if bombs[x, y] >= 1:
                    minx = np.clip(x - 3, 0, COLS-1)
                    maxx = np.clip(x + 4, 0, COLS)
                    miny = np.clip(y - 3, 0, ROWS-1)
                    maxy = np.clip(y + 4, 0, ROWS)
                    # Slicing is von inklusive bis exklusive, deswegen hinterer teil des slicing so komisch wie er aussieht
                    bomb_future_explosion_fields[minx : maxx, y] = 1
                    bomb_future_explosion_fields[x, miny : maxy] = 1
                    #print(np.transpose(bomb_future_explosion_fields))
                    #i += 1
                    #print(i)

        bomb_danger_fields = explosions + bomb_future_explosion_fields
        #print(np.transpose(bomb_danger_fields))

        # Combine the walls, crates and lingering explosions
        non_traversable_fields = walls + crates + bomb_danger_fields
        non_traversable_fields[self_pos_coordinates] = 0

        # Ab hier alle IF abfragungen für fallunterscheidungen
        if bomb_danger_fields[self_pos_coordinates] == True:

            # Add the walls and crates to the flee area, as we cant escape through them
            bad_fields = bomb_danger_fields + walls
            bad_fields += crates

            # Get a cut-out array of places that can be reached, so no crates, walls, no bombs no explosions
            # NOT everything inside of the bomb timer can be reached
            # but its a lot easier to code and the array is small enough that no optimization should be needed
            possible_flee_area = bad_fields[int(np.clip(self_pos_coordinates[0] - BOMB_TIMER, 0, 17 - 1)) : int(np.clip(self_pos_coordinates[0] + BOMB_TIMER + 1, 0, 17)), int(np.clip(self_pos_coordinates[1] - BOMB_TIMER, 0, 17 - 1)) : int(np.clip(self_pos_coordinates[1] + BOMB_TIMER + 1, 0, 17))]
            #print(np.transpose(bad_fields))
            # Initialize a definitely not shortest distance 
            curr_shortest_distance = possible_flee_area.shape[0] ** 3
            curr_best_goal = self_pos_coordinates
            

            # Iterate through all places in the flee area
            # hier badfields austauschen mit possible flear area zum optimieren evt.,
            # aber dann aufpassen mit self pos und so passt dann ja nicht mher

            # Shortest path initialisieren
            curr_shortest_path = bad_fields.shape[0] ** 3
            # Optimale next field initialisieren
            optimal_next_field = self_pos_coordinates

            for y in range(bad_fields.shape[1]):
                for x in range(bad_fields.shape[0]):
                    if bad_fields[x, y] == False: # false, also 0 heißt keine Bombe
                        curr_distance = self.l1_distance((x, y), self_pos_coordinates)
                        #if curr_distance < curr_shortest_distance:
                        self_pos_tuple = (self_pos_coordinates[0], self_pos_coordinates[1])
                            #print(type(self_pos_tuple))
                        # check if we can actually reach that point, if yes set new best target location
                        #print(np.transpose(unpassable_fields))
                        path = self.can_reach_safe_spot(unpassable_fields, self_pos_tuple, (x, y))
                        #print(path)
                        if path is not None and curr_shortest_path > len(path):
                            #curr_shortest_distance = curr_distance
                            #curr_best_target_location = (x, y)
                            optimal_next_field = np.asarray(path[1])
                            # Set shortest path if its shorter than the current shortest path
                            curr_shortest_path = len(path)


            #print("ich habe optimalen pfad gesucht")

            #print(optimal_next_field)
            direction_we_want_x = optimal_next_field[0] - self_pos_coordinates[0]
            direction_we_want_y = optimal_next_field[1] - self_pos_coordinates[1]
            
            # Die action zurückgeben die uns in die richtung bringt die wir wollen
            if direction_we_want_x > 0:
                return Actions.RIGHT.value
            elif direction_we_want_x < 0:
                return Actions.LEFT.value
            if direction_we_want_y > 0:
                return Actions.DOWN.value
            elif direction_we_want_y < 0:
                return Actions.UP.value
            else: 
                return Actions.WAIT.value

            # Choose a random action of walking around
            #return self.rng.integers(low=0, high=3) 
        # If there is an adjacent crate plant a bomb
        elif bombs_left == True and (crates[np.clip(self_pos_coordinates[0] + 1, 0, 17), self_pos_coordinates[1]] == True or crates[np.clip(self_pos_coordinates[0] - 1, 0, 17), self_pos_coordinates[1]] == True or crates[self_pos_coordinates[0], np.clip(self_pos_coordinates[1] + 1, 0, 17)] == True or crates[self_pos_coordinates[0], np.clip(self_pos_coordinates[1] - 1, 0, 17)] == True):
            return Actions.BOMB.value #bomba
        # If there is an adjacent player plant a bomb too, this is just for "readability"
        elif (bombs_left > 0) and (self.is_opponent_adjacent(self_pos_coordinates, opponents_pos_coordinates)):
            return Actions.BOMB.value #bomba
        elif len(coins) != 0:
            closest_coin = (9999,9999) # hier bleibt der ganz evt stehen egi geht aber nicht weil
            # vorher sichergestellt dass mindestens ein coin auf der map
            closest_coin_distance = 99999
            for coin in coins_coordinates:
                curr_coin_distance = self.l1_distance(self_pos_coordinates, coin)
                if curr_coin_distance < closest_coin_distance or closest_coin is None:
                    closest_coin_distance = curr_coin_distance
                    closest_coin = coin
                    #print(closest_coin)
            path_to_coin = self.can_reach_safe_spot(non_traversable_fields, self_pos_coordinates, closest_coin)
            #print(path_to_coin)
            if path_to_coin is not None:
                optimal_next_field = np.asarray(path_to_coin[1])
                return self.turn_optimal_field_into_action(self_pos_coordinates, optimal_next_field)
            elif path_to_coin is None:
                possible_actions = self.get_possible_actions(non_traversable_fields, self_pos_coordinates)
                # Repeat until we dont go back to the same location again to prevent 
                repetetive_action_bool = True
                while(repetetive_action_bool):
                    # Select random action from the possible action list
                    chosen_action = self.rng.choice(possible_actions)
                    # Mimic action to prevent going back to the same field
                    mimicked_action_field = self.mimic_action(self_pos_coordinates, chosen_action)
                    if self.prev_field is not None and self.prev_field != mimicked_action_field and len(possible_actions) > 1:
                        repetetive_action_bool = False
                    elif self.prev_field is None:
                        repetetive_action_bool = False

                return chosen_action


        else: 
            #return self.rng.integers(low=0, high=4) # Do random action if nothing else is happening
            #print(np.transpose(non_traversable_fields))
            # Check through all traversable neighbor fields, no lingering explosion, no wall, no crate
            possible_actions = self.get_possible_actions(non_traversable_fields, self_pos_coordinates)
            # Repeat until we dont go back to the same location again to prevent 
            repetetive_action_bool = True
            while(repetetive_action_bool):
                # Select random action from the possible action list
                chosen_action = self.rng.choice(possible_actions)
                # Mimic action to prevent going back to the same field
                mimicked_action_field = self.mimic_action(self_pos_coordinates, chosen_action)
                if self.prev_field is not None and self.prev_field != mimicked_action_field and len(possible_actions) > 1:
                    repetetive_action_bool = False
                elif self.prev_field is None:
                    repetetive_action_bool = False

            return chosen_action

        raise NotImplementedError()

    # Manhatten Distance
    def l1_distance(self, pos_1, pos_2):
        return np.absolute(pos_1[0] - pos_2[0]) + np.absolute(pos_1[1] - pos_2[1])

    def mimic_action(self, position, action):
        pos = [position[0], position[1]]
        if action == 0:
            pos[1] += 1
        elif action == 1:
            pos[0] += 1
        elif action == 2:
            pos[1] -= 1
        elif action == 2:
            pos[0] -= 1
        return pos

    def can_reach_safe_spot(self, unpassable_field, start_location, target_location):
        queue = deque()
        visited = set()
        # Make sure the start location is a tuple
        start_location = (int(start_location[0]), int(start_location[1]))
        #print(type(start_location))
        distance = {start_location: 0}
        prev = {}

        queue.append(start_location)
        visited.add(start_location)

        while queue:
            field = queue.popleft()

            # Explore neighbor fields
            for neighbor in self.get_neighbors(unpassable_field, field):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance[neighbor] = distance[field] + 1
                    prev[neighbor] = field
                    queue.append(neighbor)
                
                if neighbor == target_location:
                    return self.get_shortest_path(prev, start_location, target_location)

        return None

    def get_neighbors(self, matrix, node):
        neighbors = []
        row, col = node

        # Check the top neighbor
        if row > 0 and matrix[row - 1][col] == 0:
            neighbors.append((row - 1, col))

        # Check the bottom neighbor
        if row < len(matrix) - 1 and matrix[row + 1][col] == 0:
            neighbors.append((row + 1, col))

        # Check the left neighbor
        if col > 0 and matrix[row][col - 1] == 0:
            neighbors.append((row, col - 1))

        # Check the right neighbor
        if col < len(matrix[0]) - 1 and matrix[row][col + 1] == 0:
            neighbors.append((row, col + 1))

        return neighbors

    def get_shortest_path(self, prev, start, end):
        path = []
        node = end

        while node != start:
            path.append(node)
            node = prev[node]

        path.append(start)
        path.reverse()

        return path

    def get_possible_actions(self, matrix, node):
        possible_actions = []
        row, col = node

        # Check the top neighbor
        if col > 0 and matrix[row, col - 1] == 0:
            possible_actions.append(Actions.UP.value)

        # Check the right neighbor
        if row < len(matrix) - 1 and matrix[row + 1, col] == 0:
            possible_actions.append(Actions.RIGHT.value)

        # Check the bottom neighbor
        if col < len(matrix[0]) - 1 and matrix[row, col + 1] == 0:
            possible_actions.append(Actions.DOWN.value)

        # Check the left neighbor
        if row > 0 and matrix[row - 1, col] == 0:
            possible_actions.append(Actions.LEFT.value)

        # Append waiting action
        if len(possible_actions) == 0:
            possible_actions.append(4)

        #print(possible_actions)

        return possible_actions
    
    def is_opponent_adjacent(self, player_pos_coordinates, opponents_pos_coordinates):
        for opponent_coordinate in opponents_pos_coordinates:
            # Reset position to be checked
            check_pos = [player_pos_coordinates[0], player_pos_coordinates[1]]
            # Iterate through all directions
            for a in range(4):
                # Up
                if a == 0:
                    check_pos[1] -= 1
                elif a == 1:
                    check_pos[0] += 1
                elif a == 2:
                    check_pos[1] += 1
                elif a == 3:
                    check_pos[0] -= 1
                if self.l1_distance(check_pos, opponent_coordinate) < 3:
                    return True
        return False
        
    def coin_is_pathable(self, non_traversable_fields, self_pos_coordinates, coins_coordinates):
        closest_coin = self_pos_coordinates # hier bleibt der ganz evt stehen egi geht aber nicht weil
        # vorher sichergestellt dass mindestens ein coin auf der map
        closest_coin_distance = 99999
        for coin in coins_coordinates:
            curr_coin_distance = self.l1_distance(self_pos_coordinates, coin)
            if curr_coin_distance < closest_coin or closest_coin is None:
                closest_coin_distance = curr_coin_distance
                closest_coin = coin
        
        path_to_coin = can_reach_safe_spot(non_traversable_fields, self_pos_coordinates, closest_coin)
        if path_to_coin is not None:
            return True
        else:
            return False

    def turn_optimal_field_into_action(self, self_pos_coordinates, optimal_next_field):
        direction_we_want_x = optimal_next_field[0] - self_pos_coordinates[0]
        direction_we_want_y = optimal_next_field[1] - self_pos_coordinates[1]
            
        # Die action zurückgeben die uns in die richtung bringt die wir wollen
        if direction_we_want_x > 0:
            return Actions.RIGHT.value
        elif direction_we_want_x < 0:
            return Actions.LEFT.value
        if direction_we_want_y > 0:
            return Actions.DOWN.value
        elif direction_we_want_y < 0:
            return Actions.UP.value
        else: 
            return Actions.WAIT.value
