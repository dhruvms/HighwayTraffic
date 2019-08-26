#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(__file__))

from gym.envs.registration import register

register(
    id='LaneFollow-v0',
    entry_point='envs.julia_env:JuliaEnv',
    kwargs={
    			'env_name': 'LaneFollow',
    			'param_dict': {},
    		}
)

register(
    id='LaneFollow-v1',
    entry_point='envs.zmq_env:ZMQEnv',
    kwargs={
    			'env_name': 'LaneFollow',
    			'param_dict': {},
    		}
)

register(
    id='LaneChange-v0',
    entry_point='envs.julia_env:JuliaEnv',
    kwargs={
    			'env_name': 'LaneChange',
    			'param_dict': {},
    		}
)

# from julia_env import JuliaEnv
