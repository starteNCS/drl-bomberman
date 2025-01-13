import math

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

        self.eps_start = 0.5  # self.eps_start is the starting value of epsilon
        self.eps_end = 0.05  # self.eps_end is the final value of epsilon
        self.eps_decay = 2500  # self.eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
        self.steps = 0

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

        self.steps = self.steps + 1

        action = None
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        if random.random() > epsilon:
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
