from bomberman_rl.envs.agent_code import LearningAgent
import bomberman_rl.envs.events as ev
import numpy as np

import matplotlib.pyplot as plt

from io import BytesIO

from scripts.q_learning.deep_q_network import DQN
from scripts.q_learning.replay_buffer import ReplayBuffer
from scripts.q_learning.trainer import Trainer


class QLearningAgent(LearningAgent):

    def __init__(self):
        super().__init__()
        self.visualization: {} = None
        self.visualization_moving_average_window = 50

        self.gamma: float = 0.99
        self.learning_rate: float = 0.001

        self.q_net = None

        self.replay_buffer: ReplayBuffer = None
        self.trainer: Trainer = None

    def setup(self):
        pass

    def act(self, state, **kwargs):
        """
        Agent acts on the given state
        """
        return self.q_net.get_action(state)

    def setup_training(self):
        """
        Setup the agent for training. This initializes the DQN in training mode
        """
        self.gamma: float = 0.99
        self.learning_rate: float = 0.001
        self.q_net = DQN(self.gamma, self.learning_rate)
        self.q_net.train()

        self.replay_buffer = ReplayBuffer(10_000)

        self.reset_visualization()

    def game_events_occurred(self, old_state, self_action, new_state, events):
        """
        On every game event (evey step in an episode) this method is called.
        Here we keep track of some values for visualization purposes

        Also on every step we calculate the reward (see calculate_reward method) and
        train the agent on the given (S_t-1, A_t-1, R_t-1, S_t)

        :param old_state: The previous state
        :param self_action: The action that lead from old_state to new_state
        :param new_state: The new state
        :param events: The events that occured during this step
        """
        if new_state is None:
            new_state = old_state

        step_reward = QLearningAgent.calculate_reward(events, new_state)

        self.visualization["dqn_episode_reward"] += step_reward
        self.visualization["dqn_episode_steps"] += 1

        if self.trainer is None:
            self.trainer = Trainer(self.q_net, self.replay_buffer)

        done = ev.GOT_KILLED in events

        self.trainer.optimize(old_state, self_action, step_reward, new_state, done)

        pass

    def game_over(self, state):
        pass

    def end_of_round(self):
        """
        This method is called every time an episode ended

        Here we only do some housekeeping for visualization purposes, but do nothing in regard of the network
        """
        self.visualization["dqn_episode_number"] = self.visualization["dqn_episode_number"] + 1
        current_episode = self.visualization["dqn_episode_number"]

        self.visualization["dqn_total_rewards"].append(self.visualization["dqn_episode_reward"])
        self.visualization["dqn_episode_reward"] = 0
        self.visualization["dqn_episode_steps"] = 0
        self.add_moving_average(self.visualization["dqn_total_rewards"])

        self.plot_dqn_learning()

        if current_episode % 1000 == 0:
            self.q_net.save_network(f"episode_{self.visualization["dqn_episode_number"]}")
            print(f"Saved network to disk at episode {self.visualization['dqn_episode_number']}")

    @staticmethod
    def calculate_reward(events, next_state):
        """
        Calculates the step rewards given all the events

        :param events: Events that occured in this step
        :return: step reward
        """
        reward_mapping = {
            ev.MOVED_LEFT: -2.,  # moves just side to side with no punishment
            ev.MOVED_RIGHT: -2.,  # moves just side to side with no punishment
            ev.MOVED_UP: -2.,  # moves just side to side with no punishment
            ev.MOVED_DOWN: -2.,  # moves just side to side with no punishment
            ev.WAITED: -5.,

            ev.INVALID_ACTION: -10.,

            ev.BOMB_DROPPED: 15.,  # Small reward to encourage bombing
            ev.BOMB_EXPLODED: 5.,

            ev.CRATE_DESTROYED: 10.,
            ev.COIN_FOUND: 15.,
            ev.COIN_COLLECTED: 100.,  # very strong incentive to collect coins (because agent ignores coins currently)

            ev.KILLED_OPPONENT: 500.,  # very strong incentive to kill opponents (matching the reward structure of the game)
            ev.KILLED_SELF: -50.,

            ev.GOT_KILLED: -30.,
            ev.OPPONENT_ELIMINATED: 0.,

            ev.SURVIVED_ROUND: 50.,
        }

        step_reward = 0
        for event in events:
            step_reward += reward_mapping[event]

        step_reward += next_state["step"] * 0.0025  # Incentive to stay alive, 0.0025 is an educated guess that leads to
                                                    # a cumulated reward of approx 100 for staying alive 400 episodes

        return step_reward

    def plot_dqn_learning(self):
        """
        Plots a curve of all rewards and the moving average reward over the
        last "self.visualization_moving_average_window" steps. The plot is saved to the root of the project
        """
        rewards = self.visualization["dqn_total_rewards"]
        moving_average = self.visualization["dqn_total_rewards_moving_average"]
        episode_number = self.visualization["dqn_episode_number"]

        print(f"Episode {episode_number} completed")
        print(f"Replay buffer size: {len(self.replay_buffer)}")

        plt.figure(figsize=(12, 6))
        plt.title('Environment Steps: %s. - Reward: %s' % (episode_number, moving_average[-1]))
        plt.plot(rewards, label="Rewards")
        plt.plot(moving_average, label=f"Moving average of last {self.visualization_moving_average_window} Rewards")
        plt.xlabel("Environment Steps")
        plt.ylabel("Rewards")
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        with open("base.png", "wb") as f:
            f.write(buf.getvalue())

        plt.close()

    def add_moving_average(self, rewards):
        """
        Calculates the moving average of rewards array

        :param rewards: array of all rewards over the course of trainig, even though technically the last
                        "visualization_moving_average_window"-elements are enough
        """
        if len(rewards) < self.visualization_moving_average_window:
            self.visualization["dqn_total_rewards_moving_average"].append(np.mean(rewards))
            return

        window = rewards[-self.visualization_moving_average_window:]
        self.visualization["dqn_total_rewards_moving_average"].append(np.mean(window))

    def reset_visualization(self):
        """
        Setups up the visualization map
        """
        self.visualization = {
            "dqn_total_rewards": [],
            "dqn_total_rewards_moving_average": [],
            "dqn_episode_reward": 0,
            "dqn_episode_steps": 0,
            "dqn_episode_number": 0
        }
