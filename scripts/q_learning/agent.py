from bomberman_rl.envs.agent_code import LearningAgent
import bomberman_rl.envs.events as ev
import numpy as np

import torch

import matplotlib.pyplot as plt

from io import BytesIO

from scripts.q_learning.deep_q_network import DQN
from scripts.q_learning.trainer import Trainer


class QLearningAgent(LearningAgent):

    def __init__(self):
        super().__init__()
        self.visualization: {} = None
        self.gamma: float = 0.99
        self.learning_rate: float = 0.05

        self.q_net: DQN = None
        self.trainer: Trainer = None

    def setup(self):
        pass

    def act(self, state):
        return self.q_net.get_action(state)

    def setup_training(self):
        self.gamma: float = 0.99
        self.learning_rate: float = 0.001
        self.q_net = DQN(self.gamma, self.learning_rate)
        self.q_net.train()
        self.reset_visualization()

    def game_events_occurred(self, old_state, self_action, new_state, events):
        if old_state is None or new_state is None:
            return

        step_reward = QLearningAgent.calculate_reward(events)
        self.visualization["dqn_episode_reward"] += step_reward
        self.visualization["dqn_episode_steps"] += 1

        if self.trainer is None:
            self.trainer = Trainer(self.q_net)

        loss = self.trainer.optimize(old_state, self_action, new_state, step_reward)

        self.visualization["dqn_episode_losses"].append(loss)

        pass

    def game_over(self, state):
        pass

    def end_of_round(self):
        self.visualization["dqn_total_rewards"].append(self.visualization["dqn_episode_reward"])
        self.visualization["dqn_total_env_steps"].append(self.visualization["dqn_episode_steps"])
        self.visualization["dqn_episode_reward"] = 0
        self.visualization["dqn_episode_steps"] = 0
        current_episode = self.visualization["dqn_total_episode_number"]
        self.visualization["dqn_total_episode_number"].append(1 if len(current_episode) == 0 else current_episode[-1] + 1)
        self.visualization["dqn_total_mean_losses"].append(np.mean(self.visualization["dqn_episode_losses"]))
        self.visualization["dqn_episode_losses"] = []

        print(f"completed episode {current_episode[-1]}")

        self.plot_dqn_learning()

    def reset_visualization(self):
        self.visualization = {
            "dqn_total_rewards": [],
            "dqn_total_env_steps": [],
            "dqn_total_episode_number": [],  # used to display time correctly in plot
            "dqn_total_mean_losses": [],
            "dqn_episode_reward": 0,
            "dqn_episode_steps": 0,
            "dqn_episode_losses": []
        }

    def plot_dqn_learning(self):
        steps = self.visualization["dqn_total_episode_number"]
        rewards = self.visualization["dqn_total_rewards"]
        loss = self.visualization["dqn_total_mean_losses"]

        if len(steps) % 500 == 0:
            print(f"Episode {len(steps)} completed")

        if len(steps) % 100 != 0:
            return

        plt.figure(figsize=(12, 6))
        plt.title('Environment Steps: %s. - Reward: %s' % (steps[-1], np.mean(rewards[-10:])))
        plt.plot(rewards[::25], label="Rewards")
        # plt.plot(loss, label="Average loss of episode")
        plt.xlabel("Environment Steps")
        plt.ylabel("Rewards")
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        with open("base.png", "wb") as f:
            f.write(buf.getvalue())

        plt.close()

    @staticmethod
    def calculate_reward(events):
        """
        Calculates the step rewards given all the events

        :param events: Events that occured in this step
        :return: step reward
        """
        reward_mapping = {
            ev.MOVED_LEFT: -1.,
            ev.MOVED_RIGHT: -1.,
            ev.MOVED_UP: -1.,
            ev.MOVED_DOWN: -1.,
            ev.WAITED: -1.,
            ev.INVALID_ACTION: -10.,

            ev.BOMB_DROPPED: -1.,
            ev.BOMB_EXPLODED: 0.,

            ev.CRATE_DESTROYED: 0.,
            ev.COIN_FOUND: 0.,
            ev.COIN_COLLECTED: 100.,

            ev.KILLED_OPPONENT: 500.,
            ev.KILLED_SELF: 0.,

            ev.GOT_KILLED: -700.,
            ev.OPPONENT_ELIMINATED: 0.,
            ev.SURVIVED_ROUND: 0.
        }

        step_reward = 0
        for event in events:
            step_reward += reward_mapping[event]

        return step_reward
