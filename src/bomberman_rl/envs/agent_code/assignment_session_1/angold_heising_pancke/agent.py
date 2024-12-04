from enum import Enum
import numpy as np
from copy import copy, deepcopy
from itertools import count

class Agent:
    #Authors:
    #Lisa Angold
    #Daniel Heising
    #Justus Pancke
    
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
        self.movement_mapping = {
            (-1,0): 3,
            (0,1): 2,
            (1,0): 1,
            (0,-1): 0
        }
        self.current_path = None
        self.prediction_explosion_timers = np.zeros((17,17))
        self.distance_map = np.zeros((17,17))
        self.predicted_exploded_crates = np.zeros((17,17))
        self.last_state = None
        self.current_state = None
        self.own_pos = None
        self.opponent_pos = []
        self.crate_pos = []
        self.coin_pos = []
        self.wall_pos = []
        self.obstacle_map = np.zeros((17,17))
        self.hunting_counter = 0

    def act(self, state: dict, **kwargs) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        if state["step"] == 0:
            self.wall_pos = list(zip(*np.where(state["walls"] == 1)))

        self.update_state_info(state)
        
        if self.position_is_dangerous(self.own_pos):
            clear_lingering = np.vectorize(lambda x: 0 if (x == 2 or x == 1) else x)
            obstacle_map = (self.current_state["walls"] + self.current_state["crates"]) + self.prediction_explosion_timers + clear_lingering(state["explosions"])
            safe_spaces = self.get_safe_spaces(obstacle_map)
            path = self.find_fastest_unobstructed_path_to_object(safe_spaces, self.obstacle_map)
            if path != None:
                return self.followPath(path)
            
        if len(self.coin_pos) > 0:
            coin_positions = self.get_reachable_coins_on_field()
            if len(coin_positions) > 0:
                path = self.find_fastest_unobstructed_path_to_object(coin_positions, self.obstacle_map)
                if path != None:
                    return self.followPath(path)
                
        if len(self.crate_pos) > 0:
            path = self.get_path_to_best_crates()
            if path != None:
                return self.followPath(path)
        
        if self.next_to_player() and self.in_crossroad():
            if state["self_info"]["bombs_left"] > 0:
                self.hunting_counter = 0
                return 5
        else: self.hunting_counter += 1

        if self.next_to_player() and self.hunting_counter == 2:
            self.hunting_counter = 0
            if state["self_info"]["bombs_left"] > 0:
                return 5
    
        path = self.find_fastest_unobstructed_path_to_object(self.opponent_pos, self.obstacle_map)
        self.last_state = state
        if path != None:
            return self.followPath(path)
        return 4

    def in_crossroad(self):
        walls = self.current_state["walls"]
        pos = self.own_pos
        if walls[(pos[0], pos[1]-1)] == 0 and walls[(pos[0]-1, pos[1])] == 0:
            return True
        return False
    
    def update_state_info(self, state):
        walls = state["walls"]
        bombs = state["bombs"]
        crates = state["crates"]
        own_pos_map = state["self_pos"]
        opponent_pos_map = state["opponents_pos"]
        coins = state["coins"]

        #Update existing timers
        update_pred_explosion_timers = np.vectorize(lambda sec: 0 if sec == 4 else (sec+1 if sec > 0 else sec))
        #self.prediction_explosion_timers = update_pred_explosion_timers(self.prediction_explosion_timers)
        self.prediction_explosion_timers = np.zeros((17,17))

        for idx, val in np.ndenumerate(deepcopy(self.predicted_exploded_crates)):
            if crates[idx] == 0: self.predicted_exploded_crates[idx] = 0

        self.own_pos = list(zip(*np.where(own_pos_map == 1)))[0]

        self.opponent_pos = list(zip(*np.where(opponent_pos_map == 1)))
        self.opponent_pos.sort(key = lambda space: self.distance_map[space])

        self.crate_pos = list(zip(*np.where(crates == 1)))
        self.crate_pos.sort(key = lambda space: self.distance_map[space])

        self.coin_pos = list(zip(*np.where(coins == 1)))
        self.coin_pos.sort(key = lambda space: self.distance_map[space])

        self.obstacle_map = deepcopy(walls) + deepcopy(crates) + deepcopy(opponent_pos_map) + deepcopy(bombs)


        for idx, val in np.ndenumerate(deepcopy(self.distance_map)):
            distance = abs(self.own_pos[0]-idx[0]) + abs(self.own_pos[1]-idx[1])
            self.distance_map[idx] = distance

        self.current_state = state
        

        #Add new timers
        new_bombs = list(zip(*np.where(bombs > 0)))
        for pos in new_bombs:
            walls_in_way = [False, False, False, False, False]
            for i in range(1,4):
                for o, direction in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]):
                    field = (pos[0]-direction[0]*i, pos[1]-direction[1]*i)
                    size = walls.shape
                    if field[0] > (size[0] - 1) or field[0] < 0 or field[1] > (size[1] -1) or field[1] < 0:
                        continue
                    if walls[field] == 1:
                        walls_in_way[o] = True
                        continue
                    elif (not walls_in_way[o]) and walls[field] == 0:
                        self.prediction_explosion_timers[field] = bombs[pos]
                        if crates[field] == 1:
                            self.predicted_exploded_crates[field] = 1
        
        for idx, val in np.ndenumerate(deepcopy(self.obstacle_map)):
            explosions = self.current_state["explosions"]
            if explosions[idx] > 10 and self.distance_map[idx] <= explosions[idx]-10:
                self.obstacle_map[idx] = 1
            timer = self.prediction_explosion_timers[idx]
            distance = self.distance_map[idx]
            if timer > 0 and ((distance == 6-timer) or (distance == 5-timer)):
                self.obstacle_map[idx] = 1
            


    def position_is_dangerous(self, position):
        explosions = self.current_state["explosions"]
        if explosions[position] > 0 or self.prediction_explosion_timers[position] > 0:
            return True
        return False

    def get_path_to_best_crates(self):
        crates = self.current_state["crates"]
        walls = self.current_state["walls"]
        crate_paths = []
        for pos in deepcopy(self.crate_pos):
            mod_map = deepcopy(self.obstacle_map)
            mod_map[pos] = 0
            path = self.astar(mod_map, self.own_pos, pos)
            if path != None:
                crate_paths.append(path)
        max_crate_weight = -100000
        best_path = None
        for i, path in enumerate(crate_paths):
            exploding_crates = 0
            idx = path[-2]
            walls_in_way = [False, False, False, False, False]
            for i in range(1,4):
                for o, direction in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]):
                    field = (idx[0]-direction[0]*i, idx[1]-direction[1]*i)
                    size = self.obstacle_map.shape
                    if field[0] > (size[0] - 1) or field[0] < 0 or field[1] > (size[1] -1) or field[1] < 0:
                        continue
                    if walls[field] == 1:
                        walls_in_way[o] = True
                    elif (not walls_in_way[o]) and walls[field] == 0 and crates[field] == 1:
                        exploding_crates += 1
            if exploding_crates-len(path) > max_crate_weight:
                max_crate_weight = exploding_crates-len(path)
                best_path = path
        return best_path

    def get_reachable_coins_on_field(self):
        coin_positions = deepcopy(self.coin_pos)
        for coin_pos in deepcopy(coin_positions):
            own_distance = self.distance_map[coin_pos]
            for player_pos in self.opponent_pos:
                player_distance = abs(player_pos[0]-coin_pos[0]) + abs(player_pos[1]-coin_pos[1])
                distance_diff = player_distance - own_distance
                if distance_diff < -4 and coin_pos in coin_positions:
                    coin_positions.remove(coin_pos)
        return coin_positions
    
    def next_to_player(self):
        for pos in self.opponent_pos:
            #Manhattan distance because diagonals cant be reached via bomb
            distance = self.distance_map[pos]
            if distance <= 2: return True
        return False
    
    def get_safe_spaces(self, obstacle_map):
        safe_spaces = list(zip(*np.where(obstacle_map == 0)))
        safe_spaces.sort(key = lambda space: self.distance_map[space])
        return safe_spaces
    
    def followPath(self, path):
        if path == None: return 4
        next_field = path[1]
        if self.position_is_dangerous(next_field) and (not self.position_is_dangerous(self.own_pos)):
            return 4
        if self.current_state["crates"][next_field] == 1:
            if self.current_state["self_info"]["bombs_left"] > 0:
                return 5
            else:
                return 4
        movement = (next_field[0]-self.own_pos[0], next_field[1]-self.own_pos[1])
        return self.movement_mapping[movement]

    def find_fastest_unobstructed_path_to_object(self, object_positions, obstacle_map):
        path = None
        for pos in deepcopy(object_positions):
            tmp_map = deepcopy(obstacle_map)
            tmp_map[self.own_pos] = 0
            tmp_map[pos] = 0
            path = self.astar(tmp_map, self.own_pos, pos)
            object_positions.remove(pos)
            if path != None: break
        return path

    def astar(self, maze, start, end):
        class Node():

            def __init__(self, parent=None, position=None):
                self.parent = parent
                self.position = position

                self.g = 0
                self.h = 0
                self.f = 0

            def __eq__(self, other):
                return self.position == other.position

        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path


            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
                # Make sure within range
                size = maze.shape
                if node_position[0] > (size[0] - 1) or node_position[0] < 0 or node_position[1] > (size[1] -1) or node_position[1] < 0:
                    continue
                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] != 0:
                    continue
                # #Make sure safe field
                # #Explosion will still be dangerous when field is reached
                # if explosions[node_position] > 0 and self.distance_map[node_position] < explosions[node_position]-10:
                #     continue
                # #Predicted explosion will be dangerous when field is reached
                # timer = self.prediction_explosion_timers[node_position]
                # distance = self.distance_map[node_position]
                # if timer > 0 and ((distance == 6-timer) or (distance == 5-timer)):
                #     continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                flag = False
                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        flag = True
                        break
                if flag == True: continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        flag = True
                        break
                if flag == True: continue


                # Add the child to the open list
                open_list.append(child)