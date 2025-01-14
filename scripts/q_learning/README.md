# DQN Agent

This directory contains an deep reinforcement learning agent for the bomberman game. The agent makes use of:
- Deep q networks
- Epsilon decay
- Replay Buffers
- Double Q Learning

## File structure

In the next few paragraphs, the files and their contents are being explained.

The code for the agent is split among a few files for a better separation of concerns:
- agent.py 
- deep_q_network.py
- replay_buffer.py
- state_preprocessor.py
- trainer.py

Additionally, there are another two files, that are indirectly being used:
- features.py
- test.py

### agent.py

This file contains the `QLearningAgent` class, which inherits from the `Agent` class. Therefore, this is the interface
with which the game itself communicates.

This class also holds the deep q network itself

### deep_q_network.py

This file contains the implementation of the deep q network. The class `DQN` inherits from `nn.Module` and has two
important methods:

- get_action: this method returns an action, that the network thinks would work best for the given state
- forward: this method forwards an input tensor to the layers

The layers of the network are defined in the `__init__` method. Right now it consists of the input and output layer
as well as two hidden layers.

### replay_buffer.py

This class is a wrapper arround the python `deque`, which is a "double sided queue".
The wrapper just has two methods for appending and sampling.

### state_preprocessor.py

Since you cannot just input the state dictionary of this environment to the neural network, we need to preprocess the
state. This is done here. There are multiple versions, but as of now v2 is used.

### trainer.py

This file contains the training of the neural network. It has three methods:

- `optimize_single`: Does a single optimization step using the bellman function
- `optimize_replay`: Does mulitple optimization steps using the replay buffer
- `optimize`: The interface for the "outside" of trainer class. This method appends to the replay buffer and decides wether to use `optimize_single` or `optimize_replay

### features.py

To better evaluate the changes to the neural network, features.py contains three feature toggles, where features
(Epsilon decay, Replay Buffers, Double Q Learning) can be toggled on and off.