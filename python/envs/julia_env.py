import julia
import numpy as np
import gym
from gym.spaces import Box

from config import JULIA_ENV_DICT

class JuliaEnv(gym.Env):
    def __init__(self,
                 env_name,  # name of the environment to load
                 # dictionary of parameters Dict{String,Any} passed to the env
                 # initialization
                 param_dict,
                 ):
        # Load in functions
        self.j = julia.Julia()
        self.j.eval("include(\"" + JULIA_ENV_DICT[env_name] + "\")")
        self.j.using('.' + env_name)

        self.j_env_params = None
        self.j_env_obj = None
        self._action_space = None
        self._observation_space = None

        self.param_dict = param_dict

    def reset(self, render=False):
        env, obs, params = self.j.reset(self.param_dict)
        self.j_env_obj = env
        self.j_env_params = params

        lo, hi = self.j.action_space(self.j_env_params)
        self._action_space = Box(np.array(lo), np.array(hi))
        lo, hi = self.j.observation_space(self.j_env_params)
        self._observation_space = Box(np.array(lo), np.array(hi))

        return obs

    def render(self):
        raise NotImplementedError

    def save_gif(self, actions, filename):
        raise NotImplementedError

    def step(self, action):
        obs, reward, done, info, env = self.j.step(self.j_env_obj, action)
        self.j_env_obj = env

        return obs, reward, done, info

    @property
    def action_space(self):
        if self._action_space is None:
            print("InitialisationError: Must reset() environment first.")

        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            print("InitialisationError: Must reset() environment first.")

        return self._observation_space

    @property
    def reward_mech(self):
        """
        If your step function returns multiple rewards for different agents
        """
        return 'global'
