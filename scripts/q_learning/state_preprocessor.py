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
        Position(0, 1),  # Right
    ]

    V2_SIZE = 3 + 4 * 7 + 3

    @staticmethod
    def process_v2(state) -> Tensor | None:
        """
        same as process_v1, but using one-hot encoding
        """
        if state is None:
            return None

        input_tensor = torch.zeros(StatePreprocessor.V2_SIZE, dtype=torch.float)

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

        return input_tensor

    @staticmethod
    def process_v1(state) -> Tensor | None:
        if state is None:
            return None

        input_tensor = torch.zeros(14, dtype=torch.float)

        self_pos_matrix = np.array(state["self_pos"])
        player_pos_y_list, player_pos_x_list = np.where(self_pos_matrix == 1)
        player_pos = Position(player_pos_x_list[0], player_pos_y_list[0])
        input_tensor[0] = player_pos.x
        input_tensor[1] = player_pos.y

        input_tensor[2] = state["bombs"][player_pos.y][player_pos.x]
        print(f"Is bomb on player pos: {state["bombs"][player_pos.y][player_pos.x] == 1}")

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

        if tensor.shape[0] != 17*17+2:
            raise AssertionError("Tensor shape of state does not match the excepted shape of 1x{}, found {}".format(17*17+2, tensor.shape))

        return tensor.float()

    @staticmethod
    def check_position_in_matrix(matrix, position):
        return matrix[position.y][position.x] == 1

    @staticmethod
    def set_direction_information_in_tensor(tensor, tensor_counter, item, distance):
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
