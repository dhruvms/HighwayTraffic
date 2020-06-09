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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ActorCriticCNN(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim, hidden_size=64):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")

        num_inputs = state_dim[0]
        self.net = nn.Sequential(
                nn.Conv2d(num_inputs, 32, (4, 3), stride=(2, 1)), nn.ReLU(),
                nn.Conv2d(32, 64, (3, 1), stride=(2, 1)), nn.ReLU(), Flatten(),
                nn.Linear(64 * 24 * 1, hidden_size), nn.ReLU()
                )
        self.actor = nn.Linear(hidden_size, action_dim[0])
        self.action_lim = torch.FloatTensor(action_lim).to(device)

        self.critic = nn.Linear(hidden_size + action_dim[0], 1)

    def forward(self, obs, target_actions=None):
        feats = self.net(obs)
        action = torch.tanh(self.actor(feats)) * self.action_lim
        if target_actions is None:
            feats_action = torch.cat([feats, action], 1)
        else:
            feats_action = torch.cat([feats, target_actions], 1)
        return action, self.critic(feats_action)
