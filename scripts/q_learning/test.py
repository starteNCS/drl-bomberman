import unittest

import torch

from scripts.q_learning.state_preprocessor import StatePreprocessor


class StatePreprocessorTest(unittest.TestCase):
    test_map = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    test_map_self_pos = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    test_map_empty = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_v2_walls(self):
        state = {
            "round": 1,
            "step": 1,
            "walls":  self.test_map,
            "self_pos":  self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map_empty,
            "bombs": self.test_map_empty,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)

        self.assert_has(processed_state, StatePreprocessor.MapElements.WALL, "up", 4.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.WALL, "down", 2.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.WALL, "left", 1.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.WALL, "right", 3.0)
        self.assert_self_bomb(processed_state, 0)

    def test_v2_crates(self):
        state = {
            "round": 1,
            "step": 1,
            "walls": self.test_map_empty,
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map,
            "coins": self.test_map_empty,
            "bombs": self.test_map_empty,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)

        self.assert_has(processed_state, StatePreprocessor.MapElements.CRATE, "up", 4.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.CRATE, "down", 2.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.CRATE, "left", 1.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.CRATE, "right", 3.0)
        self.assert_self_bomb(processed_state, 0)

    def test_v2_coins(self):
        state = {
            "round": 1,
            "step": 1,
            "walls": self.test_map_empty,
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map,
            "bombs": self.test_map_empty,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)

        self.assert_has(processed_state, StatePreprocessor.MapElements.COIN, "up", 4.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.COIN, "down", 2.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.COIN, "left", 1.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.COIN, "right", 3.0)
        self.assert_self_bomb(processed_state, 0)

    def test_v2_bombs(self):
        state = {
            "round": 1,
            "step": 1,
            "walls": self.test_map_empty,
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map_empty,
            "bombs": self.test_map,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)

        self.assert_has(processed_state, StatePreprocessor.MapElements.BOMB, "up", 4.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.BOMB, "down", 2.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.BOMB, "left", 1.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.BOMB, "right", 3.0)
        self.assert_self_bomb(processed_state, 0)

    def test_v2_explosion(self):
        state = {
            "round": 1,
            "step": 1,
            "walls": self.test_map_empty,
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map_empty,
            "bombs": self.test_map_empty,
            "explosions": self.test_map,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)

        self.assert_has(processed_state, StatePreprocessor.MapElements.EXPLOSION, "up", 4.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.EXPLOSION, "down", 2.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.EXPLOSION, "left", 1.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.EXPLOSION, "right", 3.0)
        self.assert_self_bomb(processed_state, 0)

    def test_v2_opponent(self):
        state = {
            "round": 1,
            "step": 1,
            "walls": self.test_map_empty,
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map_empty,
            "bombs": self.test_map_empty,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)

        self.assert_has(processed_state, StatePreprocessor.MapElements.OPPONENT, "up", 4.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.OPPONENT, "down", 2.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.OPPONENT, "left", 1.0)
        self.assert_has(processed_state, StatePreprocessor.MapElements.OPPONENT, "right", 3.0)
        self.assert_self_bomb(processed_state, 0)

    def test_v2_self_bomb(self):
        state = {
            "round": 1,
            "step": 1,
            "walls": self.test_map,  # to not break the state preprocessor, but not tested here
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map_empty,
            "bombs": self.test_map_self_pos,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": 1,
                "score": 1,
            },
            "opponents_info": []
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)
        self.assert_self_bomb(processed_state, 1)

    def test_v2_meta(self):
        step_val = 303
        bomb_left_val = 404  # unique, but value does not really matter
        score_val = 505  # unique, but value does not really matter
        opponent_count = 606  # unique, but value does not really matter

        state = {
            "round": 1,
            "step": step_val,
            "walls": self.test_map,  # to not break the state preprocessor, but not tested here
            "self_pos": self.test_map_self_pos,
            "crates": self.test_map_empty,
            "coins": self.test_map_empty,
            "bombs": self.test_map_self_pos,
            "explosions": self.test_map_empty,
            "opponents_pos": self.test_map_empty,
            "self_info": {
                "bombs_left": bomb_left_val,
                "score": score_val,
            },
            "opponents_info": range(0, 606)
        }

        processed_state: torch.Tensor = StatePreprocessor.process_v2(state)
        self.assertEqual(bomb_left_val, processed_state[StatePreprocessor.V2_SIZE - 5].item())
        self.assertEqual(score_val, processed_state[StatePreprocessor.V2_SIZE - 4].item())
        self.assertEqual(opponent_count, processed_state[StatePreprocessor.V2_SIZE - 3].item())
        self.assertEqual(step_val, processed_state[StatePreprocessor.V2_SIZE - 1].item())

    def assert_self_bomb(self, state, should_have_bomb):
        self.assertEqual(should_have_bomb, state[0].item())

    def assert_has(self, state, type, direction, distance):
        direction_map = {
            "up": 0,
            "down": 1,
            "left": 2,
            "right": 3,
        }

        type_map = {
            StatePreprocessor.MapElements.WALL: 0,
            StatePreprocessor.MapElements.CRATE: 1,
            StatePreprocessor.MapElements.COIN: 2,
            StatePreprocessor.MapElements.BOMB: 3,
            StatePreprocessor.MapElements.EXPLOSION: 4,
            StatePreprocessor.MapElements.OPPONENT: 5
        }

        direction_value = direction_map[direction]
        type_value = type_map[type]

        #  1 for the state if on bomb
        # 7 for each direction (6 items + distance)
        # the type itself
        index_start = 1 + (direction_value * 7)
        index_map_element = index_start + type_value
        index_distance = index_start + 6

        self.assertEqual(1, state[index_map_element].item())
        self.assertEqual(distance, state[index_distance].item())

