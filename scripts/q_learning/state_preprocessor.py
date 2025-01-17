import enum
import heapq
from collections import deque

import numpy as np

import torch
from torch import Tensor


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f'({self.x}, {self.y})'

    def manhattan(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)


class StatePreprocessor:

    class MapElements:
        WALL = 2
        CRATE = 4
        COIN = 8
        BOMB = 16
        EXPLOSION = 32
        OPPONENT = 64

    DIRECTIONS = [
        Position(0, 1),  # Up
        Position(0, -1),  # Down
        Position(-1, 0),  # Left
        Position(1, 0),  # Right
    ]

    V0_SIZE = 17 * 17 + 2

    V2_SIZE = 1 + 4 * 7 + 5

    @staticmethod
    def process_v2(state: dict) -> Tensor | None:
        """
        In general the same as in v1, but using one-hot encoding for each direction
        This results in each direction being a sept-tupel

        [0]: If a bomb is on the field of the agent
        4 times for each direction:
            [1-8]: What is in direction
        [30]: How many bombs are left for the agent
        [31]: What the score is for the agent
        [32]: If the agent sees a bomb in any direction
        [33]: The current episode step
        """
        if state is None:
            return None

        input_tensor = torch.zeros(StatePreprocessor.V2_SIZE, dtype=torch.float)

        player_pos = StatePreprocessor.self_position(state)

        input_tensor[0] = 1 if state["bombs"][player_pos.x][player_pos.y] != 0 else 0

        input_tensor_counter = 1
        see_bomb_in_any_dir = False

        # order is given in DIRECTIONS array, up -> down -> left -> right, just like the input tensor
        for direction in StatePreprocessor.DIRECTIONS:
            distance = 1
            current_see_pos = player_pos + direction

            while True:
                wall = StatePreprocessor.check_position_in_matrix(state["walls"], current_see_pos)
                if wall:
                    input_tensor, input_tensor_counter = StatePreprocessor.set_direction_information_in_tensor(input_tensor, input_tensor_counter, StatePreprocessor.MapElements.WALL, distance)
                    break

                crate = StatePreprocessor.check_position_in_matrix(state["crates"], current_see_pos)
                if crate:
                    input_tensor, input_tensor_counter = StatePreprocessor.set_direction_information_in_tensor(input_tensor, input_tensor_counter, StatePreprocessor.MapElements.CRATE, distance)
                    break

                coin = StatePreprocessor.check_position_in_matrix(state["coins"], current_see_pos)
                if coin:
                    input_tensor, input_tensor_counter = StatePreprocessor.set_direction_information_in_tensor(input_tensor, input_tensor_counter, StatePreprocessor.MapElements.COIN, distance)
                    break

                bomb = StatePreprocessor.check_position_in_matrix(state["bombs"], current_see_pos)
                if bomb:
                    input_tensor, input_tensor_counter = StatePreprocessor.set_direction_information_in_tensor(input_tensor, input_tensor_counter, StatePreprocessor.MapElements.BOMB, distance)
                    see_bomb_in_any_dir = True
                    break

                explosion = StatePreprocessor.check_position_in_matrix(state["explosions"], current_see_pos)
                if explosion:
                    input_tensor, input_tensor_counter = StatePreprocessor.set_direction_information_in_tensor(input_tensor, input_tensor_counter, StatePreprocessor.MapElements.EXPLOSION, distance)
                    break

                opponent = StatePreprocessor.check_position_in_matrix(state["opponents_pos"], current_see_pos)
                if opponent:
                    input_tensor, input_tensor_counter = StatePreprocessor.set_direction_information_in_tensor(input_tensor, input_tensor_counter, StatePreprocessor.MapElements.OPPONENT, distance)
                    break

                current_see_pos = current_see_pos + direction
                distance = distance + 1

        input_tensor[input_tensor_counter] = state["self_info"]["bombs_left"]
        input_tensor_counter = input_tensor_counter + 1
        input_tensor[input_tensor_counter] = state["self_info"]["score"]
        input_tensor_counter = input_tensor_counter + 1
        input_tensor[input_tensor_counter] = len(state["opponents_info"])
        input_tensor_counter = input_tensor_counter + 1
        input_tensor[input_tensor_counter] = 1 if see_bomb_in_any_dir else 0
        input_tensor_counter = input_tensor_counter + 1
        input_tensor[input_tensor_counter] = state["step"]

        return input_tensor

    @staticmethod
    def process_v1(state) -> Tensor | None:
        """
        Version 1 of state preprocessing

        Saves what it sees in a direction and the distance to it in a tuple for each direction
        """
        if state is None:
            return None

        input_tensor = torch.zeros(14, dtype=torch.float)

        self_pos_matrix = np.array(state["self_pos"])
        player_pos_y_list, player_pos_x_list = np.where(self_pos_matrix == 1)
        player_pos = Position(player_pos_x_list[0], player_pos_y_list[0])
        input_tensor[0] = player_pos.x
        input_tensor[1] = player_pos.y

        input_tensor[2] = state["bombs"][player_pos.y][player_pos.x]

        input_tensor_counter = 3

        # order is given in DIRECTIONS array, up -> down -> left -> right, just like the input tensor
        for direction in StatePreprocessor.DIRECTIONS:
            distance = 1
            current_see_pos = player_pos + direction

            while True:
                wall = StatePreprocessor.check_position_in_matrix(state["walls"], current_see_pos)
                if wall:
                    input_tensor[input_tensor_counter] = StatePreprocessor.MapElements.WALL
                    input_tensor_counter = input_tensor_counter + 1
                    input_tensor[input_tensor_counter] = distance
                    input_tensor_counter = input_tensor_counter + 1
                    break

                crate = StatePreprocessor.check_position_in_matrix(state["crates"], current_see_pos)
                if crate:
                    input_tensor[input_tensor_counter] = StatePreprocessor.MapElements.CRATE
                    input_tensor_counter = input_tensor_counter + 1
                    input_tensor[input_tensor_counter] = distance
                    input_tensor_counter = input_tensor_counter + 1
                    break

                coin = StatePreprocessor.check_position_in_matrix(state["coins"], current_see_pos)
                if coin:
                    input_tensor[input_tensor_counter] = StatePreprocessor.MapElements.COIN
                    input_tensor_counter = input_tensor_counter + 1
                    input_tensor[input_tensor_counter] = distance
                    input_tensor_counter = input_tensor_counter + 1
                    break

                bomb = StatePreprocessor.check_position_in_matrix(state["bombs"], current_see_pos)
                if bomb:
                    input_tensor[input_tensor_counter] = StatePreprocessor.MapElements.BOMB
                    input_tensor_counter = input_tensor_counter + 1
                    input_tensor[input_tensor_counter] = distance
                    input_tensor_counter = input_tensor_counter + 1
                    break

                explosion = StatePreprocessor.check_position_in_matrix(state["explosions"], current_see_pos)
                if explosion:
                    input_tensor[input_tensor_counter] = StatePreprocessor.MapElements.EXPLOSION
                    input_tensor_counter = input_tensor_counter + 1
                    input_tensor[input_tensor_counter] = distance
                    input_tensor_counter = input_tensor_counter + 1
                    break

                opponent = StatePreprocessor.check_position_in_matrix(state["opponents_pos"], current_see_pos)
                if opponent:
                    input_tensor[input_tensor_counter] = StatePreprocessor.MapElements.OPPONENT
                    input_tensor_counter = input_tensor_counter + 1
                    input_tensor[input_tensor_counter] = distance
                    input_tensor_counter = input_tensor_counter + 1
                    break

                current_see_pos = current_see_pos + direction
                distance = distance + 1

        input_tensor[input_tensor_counter] = state["self_info"]["bombs_left"]
        input_tensor_counter = input_tensor_counter + 1
        input_tensor[input_tensor_counter] = state["self_info"]["score"]
        input_tensor_counter = input_tensor_counter + 1
        input_tensor[input_tensor_counter] = len(state["opponents_info"])

        return input_tensor

    @staticmethod
    def process_v0(state):
        """
        Transforms the dict state into a tensor, that can be used by the neural network
        Each field of the map gets its own "state". The "id" (value in this field) denotes what is happening on the field

        Afterward the tensor is flattened into a one dimensional tensor, to conform with the input to the network

        Ids:
            1 : WALLS
            2 : CRATES
            4 : COINS
            8 : BOMBS
            16: EXPLOSIONS
            32: SELF_POS
            64: OPPONENT_POS

        Using powers of 2, there will always be a unique id for a state of a field, even if there is a player and a
        bomb on that field for example

        :param state: Current state
        :return: 1xINPUT_SIZE tensor, where len-2 is score and len-1 is bombs_left
        """
        def map_tensor(tensor_base, tensor_add, id):
            """
            Maps the 'tensor_add' onto the 'tensor_base' with a shift in its value to the id

            :param tensor_base: The tensor base
            :param tensor_add: The tensor add
            :param id: The id for the type that is added
            :return: The mapped tensor
            """
            return tensor_base + (tensor_add * id)

        if state is None:
            return None

        base_tensor = torch.from_numpy(np.array(state["walls"]))
        base_tensor = map_tensor(base_tensor, torch.from_numpy(np.array(state["crates"])), 2)
        base_tensor = map_tensor(base_tensor, torch.from_numpy(np.array(state["coins"])), 4)
        base_tensor = map_tensor(base_tensor, torch.from_numpy(np.array(state["bombs"])), 8)
        base_tensor = map_tensor(base_tensor, torch.from_numpy(np.array(state["explosions"])), 16)
        base_tensor = map_tensor(base_tensor, torch.from_numpy(np.array(state["self_pos"])), 32)
        base_tensor = map_tensor(base_tensor, torch.from_numpy(np.array(state["opponents_pos"])), 64)

        map_tensor = torch.from_numpy(np.array(base_tensor).flatten())

        # add the "non-map state" behind the map
        bombs_left_tensor = torch.tensor(np.array([state["self_info"]["bombs_left"]]))
        score_tensor = torch.tensor(np.array([state["self_info"]["score"]]))
        tensor = map_tensor
        tensor = torch.cat((tensor, score_tensor), dim=0)
        tensor = torch.cat((tensor, bombs_left_tensor), dim=0)

        # if tensor.shape[0] != StatePreprocessor.V0_SIZE:
        #     raise AssertionError("Tensor shape of state does not match the excepted shape of 1x{}, found {}".format(StatePreprocessor.V0_SIZE, tensor.shape))

        return tensor.float()

    @staticmethod
    def check_position_in_matrix(matrix, position):
        """
        Checks if _something_ is in the position of this matrix
        """
        return matrix[position.x][position.y] != 0

    @staticmethod
    def set_direction_information_in_tensor(tensor, tensor_counter, item, distance):
        """
        For version 2

        sets the information about one direction into the input tensor for the nn
        :param tensor: The input tensor for the nn
        :param tensor_counter: The tensor index counter
        :param item: The map element, that should be set in this direction
        :param distance: The distance between the current position and the next map element

        :return: the new input tensor and new counter value
        """
        tensor[tensor_counter] = 1 if item == StatePreprocessor.MapElements.WALL else 0
        tensor_counter = tensor_counter + 1
        tensor[tensor_counter] = 1 if item == StatePreprocessor.MapElements.CRATE else 0
        tensor_counter = tensor_counter + 1
        tensor[tensor_counter] = 1 if item == StatePreprocessor.MapElements.COIN else 0
        tensor_counter = tensor_counter + 1
        tensor[tensor_counter] = 1 if item == StatePreprocessor.MapElements.BOMB else 0
        tensor_counter = tensor_counter + 1
        tensor[tensor_counter] = 1 if item == StatePreprocessor.MapElements.EXPLOSION else 0
        tensor_counter = tensor_counter + 1
        tensor[tensor_counter] = 1 if item == StatePreprocessor.MapElements.OPPONENT else 0
        tensor_counter = tensor_counter + 1

        tensor[tensor_counter] = distance
        tensor_counter = tensor_counter + 1

        return tensor, tensor_counter

    @staticmethod
    def self_position(state):
        self_pos_matrix = np.array(state["self_pos"])
        row, col = np.where(self_pos_matrix == 1)
        return Position(row[0], col[0])
