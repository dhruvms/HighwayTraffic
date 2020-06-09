import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import numpy as np

from networks import Actor, Critic, ActorCriticCNN
from structs import Memory
from utils import *

class A2C():
    def __init__(self, state_dim, action_dim, action_lim, update_type='soft',
                lr_actor=1e-4, lr_critic=1e-3, tau=1e-3,
                mem_size=1e6, batch_size=256, gamma=0.99,
                other_cars=False, ego_dim=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")

        self.joint_model = False
        if len(state_dim) == 3:
            self.model = ActorCriticCNN(state_dim, action_dim, action_lim)
            self.model_optim = optim.Adam(self.model.parameters(), lr=lr_actor)

            self.target_model = ActorCriticCNN(state_dim, action_dim, action_lim)
            self.target_model.load_state_dict(self.model.state_dict())

            self.model.to(self.device)
            self.target_model.to(self.device)

            self.joint_model = True
        else:
            self.actor = Actor(state_dim, action_dim, action_lim, other_cars=other_cars, ego_dim=ego_dim)
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
            self.target_actor = Actor(state_dim, action_dim, action_lim, other_cars=other_cars, ego_dim=ego_dim)
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_actor.eval()

            self.critic = Critic(state_dim, action_dim, other_cars=other_cars, ego_dim=ego_dim)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2)
            self.target_critic = Critic(state_dim, action_dim, other_cars=other_cars, ego_dim=ego_dim)
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_critic.eval()

            self.actor.to(self.device)
            self.target_actor.to(self.device)
            self.critic.to(self.device)
            self.target_critic.to(self.device)

        self.action_lim = action_lim
        self.tau = tau # hard update if tau is None
        self.update_type = update_type
        self.batch_size = batch_size
        self.gamma = gamma

        if self.joint_model:
            mem_size = mem_size//100
        self.memory = Memory(int(mem_size), action_dim, state_dim)

        mu = np.zeros(action_dim)
        sigma = np.array([0.5, 0.05])
        self.noise = OrnsteinUhlenbeckActionNoise(mu, sigma)
        self.target_noise = OrnsteinUhlenbeckActionNoise(mu, sigma)

        self.initialised = True
        self.training = False

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(np.expand_dims(obs, axis=0)).to(self.device)
            if self.joint_model:
                action, _ = self.model(obs)
                action = action.data.cpu().numpy().flatten()
            else:
                action = self.actor(obs).data.cpu().numpy().flatten()

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
        if self.joint_model:
            self.model.train()
            self.target_model.train()
        else:
            self.actor.train()
            self.target_actor.train()
            self.critic.train()
            self.target_critic.train()

        self.training = True

    def eval(self):
        if self.joint_model:
            self.model.eval()
            self.target_model.eval()
        else:
            self.actor.eval()
            self.target_actor.eval()
            self.critic.eval()
            self.target_critic.eval()

        self.training = False

    def save(self, folder, episode, previous=None, solved=False):
        filename = lambda type, ep : folder + '%s' % type + \
                                    (not solved) * ('_ep%d' % (ep)) + \
                                    (solved * '_solved') + '.pth'

        if self.joint_model:
            torch.save(self.model.state_dict(), filename('model', episode))
            torch.save(self.target_model.state_dict(), filename('target_model', episode))
        else:
            torch.save(self.actor.state_dict(), filename('actor', episode))
            torch.save(self.target_actor.state_dict(), filename('target_actor', episode))

            torch.save(self.critic.state_dict(), filename('critic', episode))
            torch.save(self.target_critic.state_dict(), filename('target_critic', episode))

        if previous is not None and previous > 0:
            if self.joint_model:
                os.remove(filename('model', previous))
                os.remove(filename('target_model', previous))
            else:
                os.remove(filename('actor', previous))
                os.remove(filename('target_actor', previous))
                os.remove(filename('critic', previous))
                os.remove(filename('target_critic', previous))

    def load_actor(self, actor_filepath):
        qualifier = '_' + actor_filepath.split("_")[-1]
        folder = actor_filepath[:actor_filepath.rfind("/")+1]
        filename = lambda type : folder + '%s' % type + qualifier

        if self.joint_model:
            self.model.load_state_dict(torch.load(filename('model'),
                                                    map_location=self.device))
            self.target_model.load_state_dict(torch.load(filename('target_model'),
                                                    map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(filename('actor'),
                                                    map_location=self.device))
            self.target_actor.load_state_dict(torch.load(filename('target_actor'),
                                                    map_location=self.device))

    def load_all(self, actor_filepath):
        self.load_actor(actor_filepath)
        qualifier = '_' + actor_filepath.split("_")[-1]
        folder = actor_filepath[:actor_filepath.rfind("/")+1]
        filename = lambda type : folder + '%s' % type + qualifier

        if not self.joint_model:
            self.critic.load_state_dict(torch.load(filename('critic'),
                                                    map_location=self.device))
            self.target_critic.load_state_dict(torch.load(filename('target_critic'),
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

        if self.joint_model:
            target_actions, _ = self.target_model(next_states)
            if target_noise:
                for sample in range(target_actions.shape[0]):
                    target_actions[sample] += self.target_noise()
                    target_actions[sample].clamp(-self.action_lim, self.action_lim)
            _, target_qvals = self.target_model(next_states, target_actions=target_actions)
            y = rewards + self.gamma * (1 - terminals) * target_qvals

            _, model_qvals = self.model(states, target_actions=actions)
            value_loss = F.mse_loss(y, model_qvals)
            model_actions, _ = self.model(states)
            _, model_qvals = self.model(states, target_actions=model_actions)
            action_loss = -model_qvals.mean()

            self.model_optim.zero_grad()
            (value_loss + action_loss).backward()
            self.model_optim.step()
        else:
            target_actions = self.target_actor(next_states)
            if target_noise:
                for sample in range(target_actions.shape[0]):
                    target_actions[sample] += self.target_noise()
                    target_actions[sample].clamp(-self.action_lim, self.action_lim)
            target_critic_qvals = self.target_critic(next_states, target_actions)
            y = rewards + self.gamma * (1 - terminals) * target_critic_qvals

            # optimise critic
            critic_qvals = self.critic(states, actions)
            value_loss = F.mse_loss(y, critic_qvals)
            self.critic_optim.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            # optimise actor
            action_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            action_loss.backward()
            self.actor_optim.step()

        # optimise target networks
        if self.update_type == 'soft':
            if self.joint_model:
                soft_update(self.target_model, self.model, self.tau)
            else:
                soft_update(self.target_actor, self.actor, self.tau)
                soft_update(self.target_critic, self.critic, self.tau)
        else:
            if self.joint_model:
                hard_update(self.target_model, self.model)
            else:
                hard_update(self.target_actor, self.actor)
                hard_update(self.target_critic, self.critic)

        return action_loss.item(), value_loss.item()
