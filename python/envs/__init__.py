#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(__file__))

from gym.envs.registration import register

register(
    id='HighwayTraffic-v0',
    entry_point='envs.julia_env:JuliaEnv',
    kwargs={
    			'env_name': 'HighwayTraffic',
    			'param_dict': {},
    		}
)

register(
    id='HighwayTraffic-v1',
    entry_point='envs.zmq_env:ZMQEnv',
    kwargs={
    			'env_name': 'HighwayTraffic',
    			'param_dict': {},
    		}
)
