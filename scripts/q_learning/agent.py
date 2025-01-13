from bomberman_rl.envs.actions import Actions
from bomberman_rl.envs.agent_code import LearningAgent
import bomberman_rl.envs.events as ev
import numpy as np

import torch

import matplotlib.pyplot as plt

from io import BytesIO

from scripts.q_learning.deep_q_network import DQN
from scripts.q_learning.replay_buffer import ReplayBuffer
from scripts.q_learning.trainer import Trainer


class QLearningAgent(LearningAgent):

    def __init__(self):
        super().__init__()
        self.visualization: {} = None
        self.gamma: float = 0.99
        self.learning_rate: float = 0.001

        self.q_net = None

        self.replay_buffer: ReplayBuffer = None
        self.trainer: Trainer = None

    def setup(self):
        pass

    def act(self, state, **kwargs):
        return self.q_net.get_action(state)

    def setup_training(self):
        self.gamma: float = 0.99
        self.learning_rate: float = 0.001
        self.q_net = DQN(self.gamma, self.learning_rate)
        self.q_net.train()

        self.replay_buffer = ReplayBuffer(10_000)

        self.reset_visualization()

    def game_events_occurred(self, old_state, self_action, new_state, events):
        if new_state is None:
            new_state = old_state

        step_reward = QLearningAgent.calculate_reward(events)

        self.visualization["dqn_episode_reward"] += step_reward
        self.visualization["dqn_episode_steps"] += 1

        # print(f"Action selected: {Actions(self_action)}, reward: ({step_reward} / {self.visualization['dqn_episode_reward']}), event: {events}")

        if self.trainer is None:
            self.trainer = Trainer(self.q_net, self.replay_buffer)

        self.trainer.optimize(old_state, self_action, new_state, step_reward)

        # if new_state != old_state:
        #     self.trainer.replay_buffer.push(old_state, self_action, new_state, step_reward)
        # self.trainer.optimize_replay()

        pass

    def game_over(self, state):
        pass

    def end_of_round(self):
        self.visualization["dqn_total_rewards"].append(self.visualization["dqn_episode_reward"])
        self.visualization["dqn_total_env_steps"].append(self.visualization["dqn_episode_steps"])
        self.visualization["dqn_episode_reward"] = 0
        self.visualization["dqn_episode_steps"] = 0
        current_episode = self.visualization["dqn_episode_number"]
        self.visualization["dqn_episode_number"] = self.visualization["dqn_episode_number"] + 1

        self.plot_dqn_learning()

        if current_episode % 1000 == 0:
            self.q_net.save_network(f"episode_{self.visualization["dqn_episode_number"]}")
            print(f"Saved network to disk at episode {self.visualization['dqn_episode_number']}")

    def reset_visualization(self):
        self.visualization = {
            "dqn_total_rewards": [],
            "dqn_total_env_steps": [],
            "dqn_episode_reward": 0,
            "dqn_episode_steps": 0,
            "dqn_episode_number": 1
        }

    def plot_dqn_learning(self):
        rewards = self.visualization["dqn_total_rewards"]
        episode_number = self.visualization["dqn_episode_number"]

        if episode_number % 50 == 0:
            print(f"Episode {episode_number} completed")
            print(f"Replay buffer size: {len(self.replay_buffer)}")

        if episode_number % 100 != 0:
            return

        plt.figure(figsize=(12, 6))
        plt.title('Environment Steps: %s. - Reward: %s' % (episode_number, np.mean(rewards[-10:])))
        plt.plot(rewards, label="Rewards")
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

            ev.CRATE_DESTROYED: 5.,
            ev.COIN_FOUND: 25.,
            ev.COIN_COLLECTED: 10.,

            ev.KILLED_OPPONENT: 50.,
            ev.KILLED_SELF: -20.,

            ev.GOT_KILLED: -15.,
            ev.OPPONENT_ELIMINATED: 0.,  # somebody killed somebody, not really of importance here
            ev.SURVIVED_ROUND: 35.
        }

        step_reward = 0
        for event in events:
            step_reward += reward_mapping[event]

        return step_reward
