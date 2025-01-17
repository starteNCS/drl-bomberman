import copy

import numpy as np
import torch
from torch import nn

from scripts.q_learning.deep_q_network import DQN
from scripts.q_learning.features import REPLAY_BUFFER_ENABLED, DOUBLE_DQN_ENABLED
from scripts.q_learning.replay_buffer import ReplayBuffer, Replay
from scripts.q_learning.state_preprocessor import StatePreprocessor


# from scripts.q_learning.state_preprocessor import StatePreprocessor


class Trainer:

    def __init__(self, q_net, replay_buffer: ReplayBuffer):
        self.policy_q_net = q_net
        self.optimizer = torch.optim.Adam(self.policy_q_net.parameters(), lr=self.policy_q_net.learning_rate)

        self.target_q_net: DQN = copy.deepcopy(self.policy_q_net)
        self.target_q_net.eval()

        self.replay_buffer = replay_buffer
        self.replay_optimizer_starting = 512  # only start with replay buffer optimizer, if there are 512 replays
        self.replay_batch_size = 256  # batch size for replay learning

        self.optimize_steps = 0
        self.sync_every_steps = 128

    def optimize_single(self, old_state, action, reward, next_state, done):
        """
        One iteration of Q learning (Bellman optimality equation for Q values)

        Might use double Q learning, if enabled.
        If double Q learning is enabled, the target network and the policy network are hard synced every few steps

        :param old_state: the state the environment had
        :param action: the action that lead from old_state to next_state
        :param reward: the reward that the agent got for moving from old_state to next_state
        :param next_state: the state the environment had
        :param done: whether the episode is done or not
        """

        old_state_tensor = StatePreprocessor.process_v2(old_state).float().to(self.policy_q_net.device)
        next_state_tensor = StatePreprocessor.process_v2(next_state).float().to(self.policy_q_net.device)

        # Update the Q-network using the current step information
        with torch.no_grad():
            # 1. Action selection using the main network (q_net)
            next_action = self.policy_q_net(next_state_tensor.unsqueeze(0)).argmax(1)

            next_q_value = None
            # 2. Evaluate Q-value of the selected action using the target network (q_target_net)
            if DOUBLE_DQN_ENABLED:
                next_q_value = self.target_q_net(next_state_tensor.unsqueeze(0))[0, next_action]
            else:
                next_q_value = self.policy_q_net(next_state_tensor.unsqueeze(0))[0, next_action]

            # 3. Compute target Q-value
            target_q_value = reward if done else reward + self.policy_q_net.gamma * next_q_value.item()

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

        if DOUBLE_DQN_ENABLED and self.optimize_steps % self.sync_every_steps == 0:
            self.target_q_net.load_state_dict(self.policy_q_net.state_dict())



    def optimize_replay(self):
        """
        Replay buffer of Double Q-Learning with Replay Buffer

        Uses double Q-Learning, if enabled.
        If double Q learning is enabled, the target network and the policy network are hard synced every few steps
        """
        if len(self.replay_buffer.memory) < self.replay_optimizer_starting:
            return  # Not enough samples to perform optimization

        # Sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.get_sample(self.replay_batch_size)
        batch = Replay(*zip(*batch))  # Unpack batch into namedtuple fields

        # Convert batch data into tensors
        old_states = torch.stack([StatePreprocessor.process_v2(s).float() for s in batch.old_state]).to(self.policy_q_net.device)
        actions = torch.tensor(batch.action, dtype=torch.long).to(self.policy_q_net.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.policy_q_net.device)
        next_states = torch.stack([StatePreprocessor.process_v2(s).float() for s in batch.next_state]).to(self.policy_q_net.device)
        done = torch.tensor(batch.done, dtype=torch.long).to(self.policy_q_net.device)

        # Compute Q-values for current states and actions
        q_values = self.policy_q_net(old_states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Double Q-Learning
        with torch.no_grad():
            # Select actions using the main network
            next_actions = self.policy_q_net(next_states).argmax(1)
            # Evaluate selected actions using the target network
            next_q_values = None
            if DOUBLE_DQN_ENABLED:
                next_q_values = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.policy_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Compute target Q-values
            target_q_values = rewards + (1.0 - done) * self.policy_q_net.gamma * next_q_values

        # Compute the loss
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if DOUBLE_DQN_ENABLED and self.optimize_steps % self.sync_every_steps == 0:
            self.target_q_net.load_state_dict(self.policy_q_net.state_dict())

    def optimize(self, old_state, action, reward, next_state, done):
        if not REPLAY_BUFFER_ENABLED:
            self.optimize_single(old_state, action, reward, next_state, done)
            return

        self.optimize_steps = self.optimize_steps + 1
        self.replay_buffer.push(old_state, action, reward, next_state, done)

        if len(self.replay_buffer.memory) < self.replay_optimizer_starting:
            self.optimize_single(old_state, action, reward, next_state, done)
        else:
            self.optimize_replay()
