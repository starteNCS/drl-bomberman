import numpy as np

from bomberman_rl import Actions
import random

# AUTHORS: Daniel Hilfer, Moritz Grunert, Domenic Scholz

class Agent:
    """
    Stick to this interface to enable later competition.
    (Demonstration only - do not inherit)
    """
    def __init__(self):
        self.setup()
        # TODO: Maybe add config params as constructor arguments?
        self.bomb_blast_radius = 3 # excluding the bomb position itself
        self.bomb_age_threshold = 1 # bomb age threshold for the agent to not move into the blast area (so if bomb age is less than this number, the agent would still consider moving into the blast area)
        self.coin_interest_radius = 4 # if a coin is within this radius (not a whole circle, just up down right and left), the agent moves towards it (if no explosion or wall is in the way)
        self.last_non_evading_movement = None
        self.last_evading_movement = None
        self.evading = False # indicates whether the agent is currently trying to get out of an active blast area or not

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        pass

    def get_opposite_movement(self, movement : Actions) -> Actions:
        match movement:
            case Actions.DOWN:
                return Actions.UP
            case Actions.UP:
                return Actions.DOWN
            case Actions.LEFT:
                return Actions.RIGHT
            case Actions.RIGHT:
                return Actions.LEFT
            case _:
                return None
        return None

    def is_movement(self, action : Actions) -> bool:
        return action in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]

    def convert_action_to_pos(self, action : Actions, agent_pos : tuple, step_size = 1) -> tuple:
        y  = agent_pos[0]
        x = agent_pos[1]
        match action:
            case Actions.UP:
                return (y, x-step_size)
            case Actions.DOWN:
                return (y, x+step_size)
            case Actions.LEFT:
                return (y-step_size, x)
            case Actions.RIGHT:
                return (y+step_size, x)
            case _:
                return None

    def pos_out_of_bounds(self, play_area_size : int, pos : tuple) -> bool:
        y_out_of_bounds = pos[0] < 0 or pos[0] >= play_area_size
        x_out_of_bounds = pos[1] < 0 or pos[1] >= play_area_size
        return x_out_of_bounds or y_out_of_bounds

    def get_bomb_positions(self, bombs : np.ndarray) -> list:
        raw_pos = np.where(bombs >= 1)
        num_bombs = len(raw_pos[0])
        
        bomb_data = []
        
        for i in range(num_bombs):
            pos = (int(raw_pos[0][i]), int(raw_pos[1][i]))
            age = bombs[pos]

            bomb_data.append(pos + (age,))

        return bomb_data

    def get_blast_area(self, bomb_data : list, walls : np.ndarray) -> np.ndarray:
        
        blast_area = np.zeros_like(walls)

        for bx, by, b_age in bomb_data:
            blast_area[(bx, by)] = b_age
            
            # idea: loop once over the blast radius and extend the explosion into each direction. If there was a wall (blocked), then no longer extend into that direction
            # left, right, up, down
            blocked = [False, False, False, False]
            for i in range(1, self.bomb_blast_radius + 1):
                # left, right, up, down
                new_pos = [(bx - i, by), (bx + i, by), (bx, by - i), (bx, by + i)]
                for pos_idx in range(len(new_pos)):
                    
                    if blocked[pos_idx]:
                        continue

                    pos = new_pos[pos_idx]

                    if self.pos_out_of_bounds(play_area_size=walls.shape[0], pos=pos):
                        blocked[pos_idx] = True
                        continue

                    if walls[pos]:
                        blocked[pos_idx] = True
                        continue

                    blast_area[pos] = b_age

        return blast_area

    def get_possible_movements(self, walls : np.ndarray, crates : np.ndarray, agent_pos : tuple, opponents_positions : list[tuple], bombs : np.ndarray, explosions : np.ndarray) -> list[Actions]:
        
        possible_movements = []
        
        blocked_positions = walls | crates
        actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
        for action in actions:
            new_pos = self.convert_action_to_pos(agent_pos=agent_pos, action=action)
            
            if self.pos_out_of_bounds(pos=new_pos, play_area_size=walls.shape[0]):
                continue
            
            # real explosions have a value of 11 and 12, the remaining dust has a value of 1 and 2 but the agent can walk through that
            if not blocked_positions[new_pos] and not new_pos in opponents_positions and not bombs[new_pos] and explosions[new_pos] < 10:
                possible_movements.append(action)

        return possible_movements

    def reduce_movements_based_on_blast_area(self, movements : list[Actions], blast_area : np.ndarray, agent_pos : tuple) -> list[Actions]:
        new_pos = [self.convert_action_to_pos(action=action, agent_pos=agent_pos) for action in movements]

        non_exploding_movements = []
        for movement_idx in range(len(new_pos)):
            pos = new_pos[movement_idx]

            # should never happen because @param movements only allows movements into open spaces but you never know..
            if self.pos_out_of_bounds(pos=pos, play_area_size=blast_area.shape[0]):
                continue

            if blast_area[pos] < self.bomb_age_threshold:
                non_exploding_movements.append(movements[movement_idx])

        return non_exploding_movements

    def reduce_movements_based_on_current_explosions(self, movements : list[Actions], explosions : np.ndarray, agent_pos : tuple) -> list[Actions]:
        new_pos = [self.convert_action_to_pos(action=action, agent_pos=agent_pos) for action in movements]

        non_exploding_movements = []
        for movement_idx in range(len(new_pos)):
            pos = new_pos[movement_idx]

            # should never happen because @param movements only allows movements into open spaces but you never know..
            if self.pos_out_of_bounds(pos=pos, play_area_size=explosions.shape[0]):
                continue

            # the actual explosions have values of 11 and 12, the dust afterwards has values of 1 and 2 but the agent can actually walk through the dust.
            if explosions[pos] < 10:
                non_exploding_movements.append(movements[movement_idx])

        return non_exploding_movements

    def get_coin_movements(self, walls : np.ndarray, crates : np.ndarray, coins : np.ndarray, agent_pos : tuple) -> list[Actions]:
        '''
        Returns a list of actions which lead the agent in the direction of a coin. The list is sorted in ascending distance to the coin.
        '''
        movements = []

        blocked = walls | crates

        actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
        # left right up down
        blocked_view = [False, False, False, False]
        for i in range(1, self.coin_interest_radius):
            for action_idx in range(len(actions)):
                
                if blocked_view[action_idx]:
                    continue
                
                pos = self.convert_action_to_pos(action=actions[action_idx], agent_pos=agent_pos, step_size=i)

                if self.pos_out_of_bounds(pos=pos, play_area_size=walls.shape[0]):
                    blocked_view[action_idx] = True
                    continue

                if blocked[pos]:
                    blocked_view[action_idx] = True
                    continue

                if coins[pos]:
                    movements.append(actions[action_idx])

        return movements

    def opponent_in_blast_radius(self, opponents_pos : np.ndarray, agent_pos : tuple, walls : np.ndarray) -> bool:
        
        actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
        blocked_view = [False, False, False, False]

        for i in range(1, self.bomb_blast_radius + 1):
            for action_idx in range(len(actions)):
                
                if blocked_view[action_idx]:
                    continue

                action = actions[action_idx]
                new_pos = self.convert_action_to_pos(action=action, agent_pos=agent_pos, step_size=i)

                if self.pos_out_of_bounds(play_area_size=walls.shape[0], pos=new_pos):
                    blocked_view[action_idx] = True
                    continue

                if walls[new_pos]:
                    blocked_view[action_idx] = True
                    continue

                if opponents_pos[new_pos]:
                    return True

        return False

    def get_num_crates_nearby(self, agent_pos : tuple, crates : np.ndarray) -> int:
        
        actions = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
        num_crates_nearby = 0

        for action in actions:
            new_pos = self.convert_action_to_pos(action=action, agent_pos=agent_pos)

            if self.pos_out_of_bounds(pos=new_pos, play_area_size=crates.shape[0]):
                continue

            if crates[new_pos]:
                num_crates_nearby += 1

        return num_crates_nearby

    def remove_loop_movement(self, last_movement : Actions, possible_movements : list[Actions]) -> list[Actions]:
        
        if len(possible_movements) == 1:
            return possible_movements
        
        movements = [action for action in possible_movements if self.get_opposite_movement(action) != last_movement]

        return movements

    def get_opponents_positions(self, game_state : dict) -> list[tuple]:
        opponents_positions_raw = np.where(game_state["opponents_pos"] == 1)
        num_opponents = len(opponents_positions_raw[0])
        
        opponents_positions = []
        for opp_index in range(num_opponents):
            opponents_positions.append((opponents_positions_raw[0][opp_index], opponents_positions_raw[1][opp_index]))

        return opponents_positions

    def act(self, state: dict, **kwargs) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        opponents_positions = self.get_opponents_positions(state)

        agent_pos = np.where(state["self_pos"] == 1)
        agent_pos = (int(agent_pos[0][0]), int(agent_pos[1][0])) # agent_pos in form (x, y) where (0,0) is in the top left
        
        
        plausible_actions = self.get_possible_movements(walls=state["walls"], crates=state["crates"], agent_pos=agent_pos, opponents_positions=opponents_positions, bombs=state["bombs"], explosions=state["explosions"])

        bombs = self.get_bomb_positions(state["bombs"]) # bomb array needs to be transposed in order to fit the agent coordinate system
        blast_area = self.get_blast_area(bombs, state["walls"])

        
        # TOP PRIORITY: IF WE ARE IN AN AREA THAT IS ABOUT TO EXPLODE, TRY TO MOVE AWAY FROM IT!

        if blast_area[agent_pos] >= self.bomb_age_threshold:            
            self.evading = True
            vertical_blast =  blast_area[self.convert_action_to_pos(Actions.UP, agent_pos)] >= self.bomb_age_threshold or blast_area[self.convert_action_to_pos(Actions.DOWN, agent_pos)] >= self.bomb_age_threshold
            horizontal_blast = blast_area[self.convert_action_to_pos(Actions.RIGHT, agent_pos)] >= self.bomb_age_threshold or blast_area[self.convert_action_to_pos(Actions.LEFT, agent_pos)] >= self.bomb_age_threshold

            # no matter how great our escape strategy is, if there is no space where we could move, we simply can't escape for now. 
            if len(plausible_actions) == 0:
                return Actions.WAIT.value
            
            # if there is only one thing we can do and we have no choice, do that thing.
            if len(plausible_actions) == 1:
                self.last_evading_movement = plausible_actions[0]
                return self.last_evading_movement.value
            
            # meaning that we are standing right on the bomb (meaning we layed it)
            if horizontal_blast and vertical_blast:
                # retreat if possible, else continue in the same direction
                self.last_evading_movement = self.get_opposite_movement(self.last_non_evading_movement) if self.get_opposite_movement(self.last_non_evading_movement) in plausible_actions else self.last_non_evading_movement

                # if the tactics was nice but does not work, choose randomly
                if self.last_evading_movement not in plausible_actions:
                    self.last_evading_movement = random.choice(plausible_actions)
                
                return self.last_evading_movement.value

            # idea behind right/left: if we determined the blast direction and we can directly evade the explosion, do it!
            if vertical_blast:
                
                if Actions.RIGHT in plausible_actions:
                    self.last_evading_movement = Actions.RIGHT
                    return self.last_evading_movement.value
                if Actions.LEFT in plausible_actions:
                    self.last_evading_movement = Actions.LEFT
                    return self.last_evading_movement.value
                
            
            # idea behind up/down: if we determined the blast direction and we can directly evade the explosion, do it!
            if horizontal_blast:
                
                if Actions.UP in plausible_actions:
                    self.last_evading_movement = Actions.UP
                    return self.last_evading_movement.value
                if Actions.DOWN in plausible_actions:
                    self.last_evading_movement = Actions.DOWN
                    return self.last_evading_movement.value
                
            # idea: if we already started evading and that direction is still free, go on
            if self.last_evading_movement != None and self.last_evading_movement in plausible_actions:
                return self.last_evading_movement

            # idea: if we can retreat (so the opposite movement is possible) then do it. But otherwise run into the danger and try to search for another way out.
            opposite_direction = self.get_opposite_movement(self.last_non_evading_movement)
            self.last_evading_movement = opposite_direction if opposite_direction in plausible_actions else self.last_non_evading_movement

            # if for whatever reason after all our tactics, the movement still is not possible, act randomly --> PANIC!
            if not self.last_evading_movement in plausible_actions:
                return random.choice(plausible_actions)
            else:
                return self.last_evading_movement.value
        
        # we are safe again, start to track our movement again
        if self.last_evading_movement != None and self.evading:
            self.evading = False
            self.last_non_evading_movement = self.last_evading_movement
            self.last_evading_movement = None

        # DO NOT MOVE INTO THE BLAST RADIUS OF A BOMB THAT IS ABOUT TO EXPLODE

        plausible_actions = self.reduce_movements_based_on_blast_area(movements=plausible_actions, blast_area=blast_area, agent_pos=agent_pos)

        # DO NOT WALK INTO AN EXPLOSION THAT IS HAPPENING RIGHT NOW

        explosions = state["explosions"] # explosions array needs to be transposed in order to fit the agent coordinate system
        plausible_actions = self.reduce_movements_based_on_current_explosions(movements=plausible_actions, explosions=explosions, agent_pos=agent_pos)

        # if this is the case, every action would bring us closer to death because it leads us into an explosion
        if len(plausible_actions) == 0:
            return Actions.WAIT.value

        # IF THERE IS A COIN REACHABLE AND NEARBY, TRY TO COLLECT IT!
        
        coin_movements = self.get_coin_movements(coins=state["coins"], agent_pos=agent_pos, crates=state["crates"], walls=state["walls"])
        if len(coin_movements) != 0:
            for movement in coin_movements:
                if movement in plausible_actions:
                    self.last_non_evading_movement = movement
                    return movement.value

        # IF ANOTHER PLAYER IS IN SIGHT, LAY A BOMB

        bomb_available = state["self_info"]["bombs_left"] > 0
        if self.opponent_in_blast_radius(opponents_pos=state["opponents_pos"], agent_pos=agent_pos, walls=state["walls"]) and bomb_available:
            return Actions.BOMB.value

        # IF THERE IS AT LEAST A CRATE NEARBY, LAY A BOMB

        crate_nearby = self.get_num_crates_nearby(agent_pos=agent_pos, crates=state["crates"]) > 0
        # by checking that the lengh of the plausibe actions is 1, we assure that we are in a dead end. In that case it makes sense to place a bomb if there can be something destroyed. 
        if crate_nearby and len(plausible_actions) == 1 and bomb_available:
            return Actions.BOMB.value

        # IF THERE IS MORE THAN ONE CHOICE, REMOVE THE ACTIONS WHICH WOULD LEAD US BACK --> REDUCE LOOPS

        plausible_actions = self.remove_loop_movement(last_movement=self.last_non_evading_movement, possible_movements=plausible_actions)
        
        # IF WE ARE AT THIS POINT, IT DOES NOT MATTER WHERE WE MOVE SO JUST PICK A RANDOM DIRECTION

        random_choice = random.choice(plausible_actions)
        
        if self.is_movement(random_choice):
            self.last_non_evading_movement = random_choice
        
        return random_choice.value