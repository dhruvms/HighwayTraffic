import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim,
                    other_cars=False, ego_dim=None):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")

        self.other_cars = other_cars
        if self.other_cars:
            assert ego_dim is not None
            self.ego_dim = ego_dim
            self.others_net = nn.Sequential(
                        nn.Linear(state_dim[0]-self.ego_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU()
                        )
            self.net = nn.Sequential(
                        nn.Linear(self.ego_dim + 64, 400),
                        nn.ReLU(),
                        nn.Linear(400, 300),
                        nn.ReLU(),
                        nn.Linear(300, action_dim[0])
                        )
        else:
            self.net = nn.Sequential(
                        nn.Linear(state_dim[0], 400),
                        nn.ReLU(),
                        nn.Linear(400, 300),
                        nn.ReLU(),
                        nn.Linear(300, action_dim[0])
                        )
        self.action_lim = torch.FloatTensor(action_lim).to(device)

    def forward(self, state):
        if self.other_cars:
            other_cars = state[:, self.ego_dim:]
            ego = state[:, :self.ego_dim]

            other_feats = self.others_net(other_cars)
            state_cat = torch.cat([ego, other_feats], 1)
            action = torch.tanh(self.net(state_cat)) * self.action_lim
        else:
            action = torch.tanh(self.net(state)) * self.action_lim
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,
                    other_cars=False, ego_dim=None):
        super().__init__()

        self.other_cars = other_cars
        if self.other_cars:
            assert ego_dim is not None
            self.ego_dim = ego_dim
            self.others_net = nn.Sequential(
                        nn.Linear(state_dim[0]-self.ego_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU()
                        )
            self.net = nn.Sequential(
                        nn.Linear(self.ego_dim + 64 + action_dim[0], 400),
                        nn.ReLU(),
                        nn.Linear(400, 300),
                        nn.ReLU(),
                        nn.Linear(300, 1)
                        )
        else:
            self.net = nn.Sequential(
                        nn.Linear(state_dim[0] + action_dim[0], 400),
                        nn.ReLU(),
                        nn.Linear(400, 300),
                        nn.ReLU(),
                        nn.Linear(300, 1)
                        )

    def forward(self, state, action):
        if self.other_cars:
            other_cars = state[:, self.ego_dim:]
            ego = state[:, :self.ego_dim]

            other_feats = self.others_net(other_cars)
            state_action = torch.cat([ego, other_feats, action], 1)
            return self.net(state_action)
        else:
            state_action = torch.cat([state, action], 1)
            return self.net(state_action)
