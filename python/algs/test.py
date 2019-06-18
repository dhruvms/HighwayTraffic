#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import gym
import envs

env = gym.make('LaneFollow-v0')
o = env.reset()
print(o)
