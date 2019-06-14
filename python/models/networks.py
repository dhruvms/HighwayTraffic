import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super().__init__()

        self.net = nn.Sequential(
                    nn.Linear(state_dim, 400),
                    nn.ReLU(),
                    nn.Linear(400, 300),
                    nn.ReLU(),
                    nn.Linear(300, action_dim)
                    )
        self.action_lim = action_lim

    def forward(self, state):
        action = torch.tanh(self.net(state)) * self.action_lim
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 400),
                    nn.ReLU(),
                    nn.Linear(400, 300),
                    nn.ReLU(),
                    nn.Linear(300, 1)
                    )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = self.net(state_action)
        return q
