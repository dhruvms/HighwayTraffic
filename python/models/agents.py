import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from networks import Actor, Critic
from structs import Memory
from utils import *

class A2C():
    def __init__(self, state_dim, action_dim, action_lim, update_type='soft',
                lr_actor=1e-4, lr_critic=1e-3, tau=1e-3,
                mem_size=1e6, batch_size=64, gamma=0.99):
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")

        self.actor = Actor(state_dim, action_dim, action_lim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.target_actor = Actor(state_dim, action_dim, action_lim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic = Critic(state_dim, action_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.action_lim = action_lim
        self.tau = tau # hard update if tau is None
        self.update_type = update_type
        self.batch_size = batch_size
        self.gamma = gamma

        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

        self.memory = Memory(int(mem_size), action_dim, state_dim)

        mu = np.zeros(action_dim)
        sigma = 1.0
        self.noise = OrnsteinUhlenbeckActionNoise(mu, 0.2)
        self.target_noise = OrnsteinUhlenbeckActionNoise(mu, 0.2)

        self.initialised = True
        self.training = False

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).data.cpu().numpy().flatten()
        if self.training:
            action += self.noise()
            return action
        else:
            return action

    def append(self, obs0, action, reward, obs1, terminal1):
        self.memory.append(obs0, action, reward, obs1, terminal1)

    def reset_noise(self):
        self.noise.reset()
        self.target_noise.reset()

    def train(self):
        self.actor.train()
        self.target_actor.train()
        self.critic.train()
        self.target_critic.train()

        self.training = True

    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

        self.training = False

    def save(self, folder, episode, solved=False):
        filename = lambda type : folder + '%s' % type + \
                                    (not solved) * ('_ep%d' % (episode)) + \
                                    (solved * '_solved') + '.pth'

        torch.save(self.actor.state_dict(), filename('actor'))
        torch.save(self.target_actor.state_dict(), filename('target_actor'))

        torch.save(self.critic.state_dict(), filename('critic'))
        torch.save(self.target_critic.state_dict(), filename('target_critic'))

    def load_actor(self, actor_filepath):
        qualifier = '_' + actor_filepath.split("_")[-1]
        folder = actor_filepath[:actor_filepath.rfind("/")+1]
        filename = lambda type : folder + '%s' % type + qualifier

        self.actor.load_state_dict(torch.load(filename('actor'),
                                                    map_location=self.device))
        self.target_actor.load_state_dict(torch.load(filename('target_actor'),
                                                    map_location=self.device))

    def update(self, target_noise=True):
        try:
            minibatch = self.memory.sample(self.batch_size) # dict of ndarrays
        except ValueError as e:
            print('Replay memory not big enough. Continue.')
            return None, None

        states = Variable(torch.FloatTensor(minibatch['obs0'])).to(self.device)
        actions = Variable(torch.FloatTensor(minibatch['actions'])).to(self.device)
        rewards = Variable(torch.FloatTensor(minibatch['rewards'])).to(self.device)
        next_states = Variable(torch.FloatTensor(minibatch['obs1'])).to(self.device)
        terminals = Variable(torch.FloatTensor(minibatch['terminals1'])).to(self.device)

        target_actions = self.target_actor(next_states)
        if target_noise:
            for sample in range(target_actions.shape[0]):
                target_actions[sample] += self.target_noise()
                target_actions[sample].clamp(-self.action_lim, self.action_lim)
        target_critic_qvals = self.target_critic(next_states, target_actions)
        y = rewards + self.gamma * (1 - terminals) * target_critic_qvals

        # optimise critic
        critic_qvals = self.critic(states, actions)
        critic_loss = F.mse_loss(y, critic_qvals)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # optimise actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # optimise target networks
        if self.update_type == 'soft':
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)
        else:
            hard_update(self.target_actor, self.actor)
            hard_update(self.target_critic, self.critic)

        return actor_loss.item(), critic_loss.item()
