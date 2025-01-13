import torch

from scripts.q_learning.state_preprocessor import StatePreprocessor


class Trainer:

    def __init__(self, q_net):
        self.q_net = q_net
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.q_net.learning_rate)

    def optimize(self,  old_state, action, next_state, reward):
        """
        One iteration of Q learning (Bellman optimality equation for Q values)
        """

        # TODO@JONATHAN: Hier replay buffer (ist so wie in der gegebenen file dann vermutlich)
        # TODO@JONATHAN: Hier vielleicht double q learning?

        old_state_tensor = StatePreprocessor.process_v1(old_state).float().to(self.q_net.device)
        next_state_tensor = StatePreprocessor.process_v1(next_state).float().to(self.q_net.device)

        # Update the Q-network using the current step information
        with torch.no_grad():
            next_q_value = self.q_net(next_state_tensor.unsqueeze(0)).max(1)[0].item()
            target_q_value = reward + self.q_net.gamma * next_q_value

        # Convert the target Q-value to a PyTorch tensor and move it to the same device as the model
        target_tensor = torch.tensor([target_q_value]).to(self.q_net.device)

        # Compute the Q-value for the current state-action pair
        q_value = self.q_net(old_state_tensor)[action]

        # Compute the loss
        loss = torch.nn.functional.mse_loss(q_value, target_tensor.squeeze(0))

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
