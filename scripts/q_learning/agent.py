from bomberman_rl.envs.actions import ActionSpace
from bomberman_rl.envs.agent_code import LearningAgent
import random
import numpy as np

import torch
import torch.nn as nn
import bomberman_rl.envs.events as ev

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from io import BytesIO


class QLearningAgent(LearningAgent):

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.visualization: {} = None
        self.total_steps: int = 0
        self.gamma: float = 0
        self.learning_rate: float = 0
        self.q_net: DQN = None
        self.setup_training()

    def setup(self):
        pass

    def act(self, state):
        return self.q_net.get_action(state)

    def setup_training(self):
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.q_net = DQN(self.gamma, self.learning_rate)
        self.q_net.train()
        self.reset_visualization()

    def game_events_occurred(self, old_state, self_action, new_state, events):
        if old_state is None or new_state is None:
            return

        step_reward = QLearningAgent.calculate_reward(events)
        self.visualization["dqn_episode_reward"] += step_reward
        self.visualization["dqn_episode_steps"] += 1

        self.q_net.optimize_incremental(old_state, self_action, new_state, step_reward)

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

        self.plot_dqn_learning()

    def reset_visualization(self):
        self.visualization = {
            "dqn_total_rewards": [],
            "dqn_total_env_steps": [],
            "dqn_total_episode_number": [],  # used to display time correctly in plot
            "dqn_episode_reward": 0,
            "dqn_episode_steps": 0
        }

    def plot_dqn_learning(self):
        steps = self.visualization["dqn_total_episode_number"]
        rewards = self.visualization["dqn_total_rewards"]

        if len(steps) % 500 == 0:
            print(f"Episode {len(steps)} completed")

        if len(steps) != 10_000:
            return

        plt.figure(figsize=(12, 6))
        plt.title('Environment Steps: %s. - Reward: %s' % (steps[-1], np.mean(rewards[-10:])))
        plt.plot(steps, rewards, label="Rewards")
        plt.xlabel("Environment Steps")
        plt.ylabel("Rewards")
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        with open("my_plot.png", "wb") as f:
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


class DQN(nn.Module):

    INPUT_SIZE = 291

    def __init__(self, gamma, learning_rate):
        super(DQN, self).__init__()

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.device = self.choose_device()
        print("Using device: {}".format(self.device))

        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, ActionSpace.n),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
        if random.randint(1, 10) == 1:
            action = ActionSpace.sample()
            # print("RND: {}".format(action))
        else:
            # Greedy action (exploitation)
            state_tensor = self.state_to_tensor(state)
            state_tensor = state_tensor.unsqueeze(0)

            q_values = self.forward(state_tensor.to(self.device))
            action = q_values.argmax(dim=1).item()
            # print("DQN: {}".format(action))
        return action

    def optimize_incremental(self, old_state, action, next_state, reward):
        """
        One iteration of Q learning (Bellman optimality equation for Q values) on a random batch of past experience
        """

        # TODO@JONATHAN: Hier replay buffer (ist so wie in der gegebenen file dann vermutlich)
        # TODO@JONATHAN: Hier vielleicht double q learning?

        old_state_tensor = self.state_to_tensor(old_state).float().to(self.device)
        next_state_tensor = self.state_to_tensor(next_state).float().to(self.device)

        # Update the Q-network using the current step information
        with torch.no_grad():
            next_q_value = self(next_state_tensor.unsqueeze(0)).max(1)[0].item()
            target_q_value = reward + self.gamma * next_q_value

        # Convert state to a PyTorch tensor, add batch dimension, and move to device for neural network input
        state_tensor = self.forward(old_state_tensor).unsqueeze(0).to(self.device)
        # Convert the target Q-value to a PyTorch tensor and move it to the same device as the model
        target_tensor = torch.FloatTensor([target_q_value]).squeeze(0).to(self.device)

        # Compute the Q-value for the current state-action pair
        q_value = state_tensor[0, action]

        # Compute the loss
        loss = torch.nn.functional.mse_loss(q_value, target_tensor)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_tensor(self, state):
        """
        Transforms the dict state into a tensor, that can be used by the neural network
        Each field of the map gets its own "state". The "id" (value in this field) denotes what is happening on the field

        Afterward the tensor is flattened into a one dimensional tensor, to conform with the input to the network

        Ids:
            1 : WALLS
            2 : CRATES
            4 : COINS
            8 : BOMBS
            16: EXPLOSIONS
            32: SELF_POS
            64: OPPONENT_POS

        Using powers of 2, there will always be a unique id for a state of a field, even if there is a player and a
        bomb on that field for example

        :param state: Current state
        :return: 1xINPUT_SIZE tensor, where len-2 is score and len-1 is bombs_left
        """
        if state is None:
            return None

        base_tensor = torch.from_numpy(np.array(state["walls"]))
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["crates"])), 2)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["coins"])), 4)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["bombs"])), 8)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["explosions"])), 16)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["self_pos"])), 32)
        base_tensor = self.map_tensor(base_tensor, torch.from_numpy(np.array(state["opponents_pos"])), 64)

        map_tensor = torch.from_numpy(np.array(base_tensor).flatten())

        # add the "non-map state" behind the map
        bombs_left_tensor = torch.tensor(np.array([state["self_info"]["bombs_left"]]))
        score_tensor = torch.tensor(np.array([state["self_info"]["score"]]))
        tensor = map_tensor
        tensor = torch.cat((tensor, score_tensor), dim=0)
        tensor = torch.cat((tensor, bombs_left_tensor), dim=0)

        if tensor.shape[0] != DQN.INPUT_SIZE:
            raise AssertionError("Tensor shape of state does not match the excepted shape of 1x{}, found {}".format(DQN.INPUT_SIZE, tensor.shape))

        tensor.to(device=self.device)
        return tensor.float()

    @staticmethod
    def map_tensor(tensor_base, tensor_add, id):
        """
        Maps the 'tensor_add' onto the 'tensor_base' with a shift in its value to the id

        :param tensor_base: The tensor base
        :param tensor_add: The tensor add
        :param id: The id for the type that is added
        :return: The mapped tensor
        """
        return tensor_base + (tensor_add * id)

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
