#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import argparse
import numpy as np
import torch
import time

from external import *
import envs # registers the environment
from utils import get_model_name

args = get_args()
if args.model_name is None:
    args.model_name = [0, 1, 2, 3, 4, 5, 6, 7]

model_name = get_model_name(args)
expdir = args.save_dir + model_name + '/'
args.log_dir = expdir + 'logs/'
args.model_dir = expdir + 'model/'

args.det = not args.non_det
args.eval = True
device = torch.device("cuda:0" if args.cuda else "cpu")
env = make_vec_envs(
    args,
    device,
    allow_early_resets=True)
action_space_hi = env.action_space.high
action_space_lo = env.action_space.low

# Get a render function
render_func = get_render_func(env)
timestr = time.strftime("%Y%m%d-%H%M%S")
viddir = expdir + '/vids/'
if not os.path.exists(viddir):
    os.makedirs(viddir)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.model_dir, args.algo + ".pt"), map_location=device)
actor_critic.eval()

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

eval_reward = 0.0
solved = 0
for episode in range(1, args.eval_episodes+1):
    filename = viddir + 'test_%d.mp4' % (episode)
    obs = env.reset()
    ep_reward = 0.0
    for t in range(1, args.max_steps+1):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)

        # # Clamp action to limits
        # torch.clamp_(action[:, 0], action_space_lo[0], action_space_hi[0])
        # torch.clamp_(action[:, 1], action_space_lo[1], action_space_hi[1])
        # Observe reward and next obs
        obs, reward, terminal, debug = env.step(action)
        masks.fill_(0.0 if terminal else 1.0)
        ep_reward += reward

        if terminal or t == args.max_steps:
            if terminal:
                solved += 1
            default_filename = "eval_ep.mp4"
            try:
                os.rename(default_filename, filename)
            except FileNotFoundError:
                render_func(filename)

            break

    print('EVAL: Episode reward = %f' % (ep_reward))
    eval_reward += ep_reward

avg_reward = eval_reward / args.eval_episodes
print('EVAL: Avg reward = %f | Unsolved = %d' % (avg_reward, solved))
