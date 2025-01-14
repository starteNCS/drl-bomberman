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

        step_reward = QLearningAgent.calculate_reward(events, new_state)

        self.visualization["dqn_episode_reward"] += step_reward
        self.visualization["dqn_episode_steps"] += 1

        # print(f"Action selected: {Actions(self_action)}, reward: ({step_reward} / {self.visualization['dqn_episode_reward']}), event: {events}")

        if self.trainer is None:
            self.trainer = Trainer(self.q_net, self.replay_buffer)

        done = ev.GOT_KILLED in events

        self.trainer.optimize(old_state, self_action, step_reward, new_state, done)

        pass

    def game_over(self, state):
        pass

    def end_of_round(self):
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

    def reset_visualization(self):
        self.visualization = {
            "dqn_total_rewards": [],
            "dqn_total_rewards_moving_average": [],
            "dqn_episode_reward": 0,
            "dqn_episode_steps": 0,
            "dqn_episode_number": 0
        }

    def plot_dqn_learning(self):
        rewards = self.visualization["dqn_total_rewards"]
        moving_average = self.visualization["dqn_total_rewards_moving_average"]
        episode_number = self.visualization["dqn_episode_number"]

        print(f"Episode {episode_number} completed")
        print(f"Replay buffer size: {len(self.replay_buffer)}")

        plt.figure(figsize=(12, 6))
        plt.title('Environment Steps: %s. - Reward: %s' % (episode_number, np.mean(rewards[-10:])))
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
        if len(rewards) < self.visualization_moving_average_window:
            self.visualization["dqn_total_rewards_moving_average"].append(np.mean(rewards))
            return

        window = rewards[-self.visualization_moving_average_window:]
        self.visualization["dqn_total_rewards_moving_average"].append(np.mean(window))

    @staticmethod
    def calculate_reward(events, next_state):
        """
        Calculates the step rewards given all the events

        :param events: Events that occured in this step
        :return: step reward
        """
        reward_mapping = {
            ev.MOVED_LEFT: -0.1,
            ev.MOVED_RIGHT: -0.1,
            ev.MOVED_UP: -0.1,
            ev.MOVED_DOWN: -0.1,
            ev.WAITED: -1.,

            ev.INVALID_ACTION: -10.,

            ev.BOMB_DROPPED: 5.,  # Small reward to encourage bombing
            ev.BOMB_EXPLODED: 1.,  # Encourage the agent to drop bombs in effective places

            ev.CRATE_DESTROYED: 10.,  # Increased to emphasize importance
            ev.COIN_FOUND: 15.,  # Reduced to align better with COIN_COLLECTED
            ev.COIN_COLLECTED: 25.,  # Increased to prioritize collection over finding

            ev.KILLED_OPPONENT: 50.,  # Keep as is
            ev.KILLED_SELF: -50.,  # Stronger penalty to discourage self-destruction

            ev.GOT_KILLED: -30.,  # Stronger penalty to emphasize survival
            ev.OPPONENT_ELIMINATED: 0.,  # Neutral

            ev.SURVIVED_ROUND: 50.,  # Increased to encourage long-term planning
        }

        step_reward = 0
        for event in events:
            step_reward += reward_mapping[event]

        step_reward += next_state["step"] * 0.01

        return step_reward
