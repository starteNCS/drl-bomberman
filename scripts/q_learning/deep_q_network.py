from bomberman_rl.envs.actions import ActionSpace
import random

import torch
import torch.nn as nn

from scripts.q_learning.state_preprocessor import StatePreprocessor


class DQN(nn.Module):

    INPUT_SIZE = 13

    def __init__(self, gamma, learning_rate):
        super(DQN, self).__init__()

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.device = self.choose_device()
        print("Using device: {}".format(self.device))

        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 60),
            nn.ReLU(),
            nn.Linear(60, ActionSpace.n),
        ).to(self.device)

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
        if random.randint(1, 25) == 1:
            action = ActionSpace.sample()
            # print("RND: {}".format(action))
        else:
            state_tensor = StatePreprocessor.process_v1(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            q_values = self.forward(state_tensor.to(self.device))
            action = q_values.argmax(dim=1).item()
            # print("DQN: {}".format(action))
        return action



    @staticmethod
    def choose_device():
        """
        Chooses which device is available to run the nn on
        :return: string that represents the device
        """

        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"

        return "cpu"
