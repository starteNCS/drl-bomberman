import copy

import numpy as np
import torch
from torch import nn

from scripts.learning_agent.q_learning import DQN
from scripts.q_learning.replay_buffer import ReplayBuffer, Replay
from scripts.q_learning.state_preprocessor import StatePreprocessor


# from scripts.q_learning.state_preprocessor import StatePreprocessor


class Trainer:

    def __init__(self, q_net, replay_buffer: ReplayBuffer):
        self.policy_q_net = q_net

        self.target_q_net: DQN = copy.deepcopy(self.policy_q_net)
        self.target_q_net.eval()
        self.optimizer = torch.optim.Adam(self.target_q_net.parameters(), lr=self.policy_q_net.learning_rate)

        self.replay_buffer = replay_buffer

        self.tau = 0.01

    def optimize(self,  old_state, action, next_state, reward):
        """
        One iteration of Q learning (Bellman optimality equation for Q values)
        """

        # TODO@JONATHAN: Hier vielleicht double q learning?

        old_state_tensor = StatePreprocessor.process_v2(old_state).float().to(self.policy_q_net.device)
        next_state_tensor = StatePreprocessor.process_v2(next_state).float().to(self.policy_q_net.device)

        # Update the Q-network using the current step information
        with torch.no_grad():
            # 1. Action selection using the main network (q_net)
            next_action = self.policy_q_net(next_state_tensor.unsqueeze(0)).argmax(1)

            # 2. Evaluate Q-value of the selected action using the target network (q_target_net)
            next_q_value = self.target_q_net(next_state_tensor.unsqueeze(0))[0, next_action]

            # 3. Compute target Q-value
            target_q_value = reward + self.policy_q_net.gamma * next_q_value.item()

        # Convert the target Q-value to a PyTorch tensor and move it to the same device as the model
        target_tensor = torch.tensor([target_q_value]).to(self.policy_q_net.device)

        # Compute the Q-value for the current state-action pair
        q_value = self.policy_q_net(old_state_tensor.unsqueeze(0))[0, action]

        # Compute the loss
        loss = torch.nn.functional.mse_loss(q_value, target_tensor.squeeze(0))

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.policy_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def optimize_replay(self, old_state, action, next_state, reward):
        """
        One iteration of Q learning (Bellman optimality equation for Q values)

        Makes use of the Replay Buffer
        """

        self.replay_buffer.push(old_state, action, next_state, reward)
        sampled_replays = self.replay_buffer.get_sample(128)

        # Transposing the samples. List of replays becomes ONE replay with list of old_state, action, reward, next_state
        # batch = Replay(*zip(*sampled_replays))

        losses = []
        for state, action, next_state, reward in sampled_replays:
            loss = self.optimize(state, action, next_state, reward)
            losses.append(loss)

        return np.mean(losses)
