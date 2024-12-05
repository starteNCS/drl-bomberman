import random

from bomberman_rl import Actions
from ...interface import RuleBaseAgent


'''
    Simply place this file in the scripts folder and run the main.py file to see the agent in action.

    Contributors:
    Alexander Hülsmann
    Philipp Honsel
    Johnathan Wrede
'''


class Agent(RuleBaseAgent):
    def __init__(self):
        super().__init__()
        self.direction = Actions.RIGHT.value
        self.escape_steps = 0
        self.avoid_coords = []
        self.rng = random.Random()

    def get_positive_coords(self, nd_array):
        positive_coords = []
        for x in range(nd_array.shape[0]):
            for y in range(nd_array.shape[1]):
                if nd_array[x, y] == 1:
                    positive_coords.append((x, y))
        return positive_coords

    def get_pos(self, state):
        x, y = self.get_positive_coords(state["self_info"]["position"])[0]
        return x, y

    def get_next_coords(self, state, direction):
        x, y = self.get_pos(state)
        if direction == Actions.UP.value:
            y -= 1
        elif direction == Actions.DOWN.value:
            y += 1
        elif direction == Actions.LEFT.value:
            x -= 1
        elif direction == Actions.RIGHT.value:
            x += 1
        return x, y

    def facing_object(self, state, direction):
        x, y = self.get_next_coords(state, direction)
        if state["walls"][x, y] == 1:
            return "wall"
        if state["crates"][x, y] == 1:
            return "crate"
        if state["bombs"][x, y] > 0:
            return "bomb"
        return None

    def setup(self):
        pass

    def act(self, state: dict, **kwargs) -> int:

        # Bomb was placed -> move away
        valid_move = None
        if self.escape_steps > 0:
            self.escape_steps -= 1
            for i in range(4):
                if self.facing_object(state, i) is None:
                    self.direction = i
                    if self.get_next_coords(state, i) not in self.avoid_coords:
                        self.avoid_coords.append(self.get_pos(state))
                        valid_move = i
                        break

            if self.escape_steps == 0:
                self.avoid_coords = []
                return Actions.WAIT.value
            if valid_move is not None:
                return valid_move
            else:
                return Actions.WAIT.value

        # Move to next free spot
        if self.facing_object(state, self.direction) == "wall" or self.facing_object(state, self.direction) == "bomb":
            self.direction = (self.direction + 1) % 4
        elif self.facing_object(state, self.direction) == "crate":
            self.escape_steps = 5
            self.avoid_coords.append(self.get_pos(state))
            return Actions.BOMB.value
        else:
            if state["step"] % 20 == 0:
                self.direction = self.rng.randint(0, 3)

        return self.direction