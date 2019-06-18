import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")

        self.net = nn.Sequential(
                    nn.Linear(state_dim[0], 400),
                    nn.ReLU(),
                    nn.Linear(400, 300),
                    nn.ReLU(),
                    nn.Linear(300, action_dim[0])
                    )
        self.action_lim = torch.FloatTensor(action_lim).to(device)

    def forward(self, state):
        action = torch.tanh(self.net(state)) * self.action_lim
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
                    nn.Linear(state_dim[0] + action_dim[0], 400),
                    nn.ReLU(),
                    nn.Linear(400, 300),
                    nn.ReLU(),
                    nn.Linear(300, 1)
                    )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = self.net(state_action)
        return q
