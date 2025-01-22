import math
import os

from bomberman_rl.envs.actions import ActionSpace, Actions
import random

import torch
import torch.nn as nn

from q_learning.features import EPSILON_DECAY_ENABLED, COMPLEX_TRAINER, SIMPLE_TRAINER
from q_learning.replay_buffer import ReplayBuffer
from q_learning.state_preprocessor import StatePreprocessor, get_rule_based_action


class DQN(nn.Module):

    INPUT_SIZE = StatePreprocessor.V2_SIZE
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, gamma, learning_rate, training):
        super(DQN, self).__init__()

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.device = self.choose_device()
        self.training = training
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
        self.eps_decay = 5000  # a relatively high decay, to explore a lot in the begging
        self.steps = 0

    def forward(self, in_tensor):
        """
        "Runs the neural network"
        Expects the input to be 1xINPUT_SIZE

        :param in_tensor: The tensor for the input
        :return: The output of the neural network (q values for this state, is of shape 1xActionSpace.n).
        """
        return self.layers(in_tensor)

    def get_action(self, state, trainer):
        """
        Select an action using a greedy policy, sometimes.
        :param state: Current state (dict, observation space).
        :return: Selected action.
        """
        action = None

        self.steps = self.steps + 1
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        trainer_action = trainer.act(state)
        if self.training and (EPSILON_DECAY_ENABLED and random.random() < epsilon) or (not EPSILON_DECAY_ENABLED and random.random() < 0.1):
            if random.random() < 0.1:
                print("Random action")
                action = ActionSpace.sample()
            else:
                print("Trainer action")
                if trainer_action is not None and COMPLEX_TRAINER:
                    action = trainer_action
                elif SIMPLE_TRAINER:
                    action = get_rule_based_action(StatePreprocessor.process_v2(state))
                else:
                    action = Actions.WAIT.value
        else:
            print("DQN action")
            state_tensor = StatePreprocessor.process_v2(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            with torch.no_grad():
                q_values: torch.Tensor = self.forward(state_tensor.to(self.device))
                action = q_values.max(dim=1).indices.item()

        return action

    def get_q_value(self, state):
        state_tensor = StatePreprocessor.process_v2(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values: torch.Tensor = self.forward(state_tensor.to(self.device))

        return q_values.max(dim=1).values.item()

    def save_network(self, filename):
        """
        Saving the network weights to the disk
        """
        torch.save(self.state_dict(), f"{DQN.FILE_PATH}/{filename}.pt")

    def load_network(self, filename):
        """
        Loading the network weights from the disk
        """
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
