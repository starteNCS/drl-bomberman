from bomberman_rl.envs.actions import Actions, ActionSpace
from bomberman_rl.envs.agent_code import LearningAgent
import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QLearningAgent(LearningAgent):

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.visualization = None
        self.total_steps = None
        self.gamma = None
        self.learning_rate = None
        self.q_net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_training()

    def setup(self):
        pass

    def act(self, state):
        return self.q_net.get_action(state)

    def setup_training(self):
        self.q_net = DQN()
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.total_steps = 20_000
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.visualization = {
            "dqn_all_rewards": [],
            "dqn_total_env_steps": [],  # Track environment steps
            "dqn_episode_reward": 0,
        }

    def game_events_occurred(self, old_state, self_action, new_state, events):
        pass


    def game_over(self, state):
        pass


    def end_episode(self, state):
        pass

class DQN(nn.Module):

    INPUT_SIZE = 291

    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, ActionSpace.n),
        )

    def forward(self, in_tensor):
        """
        "Runs the neural network"
        Expects the input to be 1xINPUT_SIZE

        :param in_tensor: The tensor for the input
        :return: The output of the neural network (q values for this state, is of shape 1xActionSpace.n).
        """
        return self.layers(in_tensor)

    def get_action(self, state):
        """
        Select an action using a greedy policy, sometimes.
        :param state: Current state (dict, observation space).
        :return: Selected action.
        """
        # TODO@JONATHAN: hier ne bessere epsilon decay funktion finden
        action = None
        if random.randint(1, 10) == 1:
            action = ActionSpace.sample()
            print("RND: {}".format(action))
        else:
            # Greedy action (exploitation)
            state_tensor = self.state_to_tensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
            else:
                state_tensor = state_tensor.cpu()

            q_values = self.forward(state_tensor)
            action = q_values.argmax(dim=1).item()
            print("DQN: {}".format(action))
        return action

    def state_to_tensor(self, state):
        """
        Transforms the dict state into a tensor, that can be used by the neural network
        Each field of the map gets its own "state". The "id" (value in this field) denotes what is happening on the field

        Afterward the tensor is flattened into a one dimensional tensor, to conform with the input to the network

        Ids:
            1: WALLS
            2: CRATES
            3: COINS
            4: BOMBS
            5: EXPLOSIONS
            6: SELF_POS
            7: OPPONENT_POS


        :param state: Current state
        :return: 1xINPUT_SIZE tensor, where len-2 is score and len-1 is bombs_left
        """
        base_tensor = torch.from_numpy(np.array(state["walls"]))
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["crates"])), 2)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["coins"])), 3)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["bombs"])), 4)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["explosions"])), 5)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["self_pos"])), 6)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["opponents_pos"])), 7)


        map_tensor = torch.from_numpy(np.array(base_tensor).flatten())

        # add the "non-map state" behind the map
        bombs_left_tensor = torch.tensor(np.array([state["self_info"]["bombs_left"]]))
        score_tensor = torch.tensor(np.array([state["self_info"]["score"]]))
        tensor = map_tensor
        tensor = torch.cat((tensor, score_tensor), dim=0)
        tensor = torch.cat((tensor, bombs_left_tensor), dim=0)

        if tensor.shape[0] != DQN.INPUT_SIZE:
            raise AssertionError("Tensor shape of state does not match the excepted shape of 1x{}, found {}".format(DQN.INPUT_SIZE, tensor.shape))

        return tensor.float()


    def map_tensor(self, tensor_base, tensor_add, id):
        return tensor_base + (tensor_add * id)