import math

from bomberman_rl.envs.actions import ActionSpace, Actions
import random

import torch
import torch.nn as nn

from scripts.q_learning.replay_buffer import ReplayBuffer
from scripts.q_learning.state_preprocessor import StatePreprocessor


class DQN(nn.Module):

    INPUT_SIZE = StatePreprocessor.V2_SIZE
    FILE_PATH = "/Users/philipp/Development/Master/DRL/bomberman_rl/trained_networks/next"

    def __init__(self, gamma, learning_rate):
        super(DQN, self).__init__()

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.device = self.choose_device()
        print("Using device: {}".format(self.device))

        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ActionSpace.n),
        ).to(self.device)

        self.replay_buffer = ReplayBuffer(10000)

        self.eps_start = 1  # self.eps_start is the starting value of epsilon
        self.eps_end = 0.05  # self.eps_end is the final value of epsilon
        self.eps_decay = 5000  # self.eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
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
        action = None

        self.steps = self.steps + 1
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        if random.random() < epsilon:
            action = ActionSpace.sample()
        else:
            state_tensor = StatePreprocessor.process_v2(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            with torch.no_grad():
                q_values = self.forward(state_tensor.to(self.device))
                action = q_values.argmax(dim=1).item()

        return action

    def save_network(self, filename):
        torch.save(self.state_dict(), f"{DQN.FILE_PATH}/{filename}.pt")

    def load_network(self, filename):
        network = torch.load(f"{DQN.FILE_PATH}/{filename}.pt", weights_only=True, map_location=self.device)
        if network is None:
            raise AssertionError(f"Expected to find {DQN.FILENAME} file")

        self.load_state_dict(network)

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
