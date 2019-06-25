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

args = get_args()

args.det = not args.non_det
args.eval = True
device = torch.device("cuda:0" if args.cuda else "cpu")
env = make_vec_envs(
    args,
    device,
    allow_early_resets=True)

# Get a render function
render_func = get_render_func(env)
timestr = time.strftime("%Y%m%d-%H%M%S")
gifdir = args.load_dir + args.algo + '/gifs/' + args.env_name.lower() + '-' + timestr + '/'
if not os.path.exists(gifdir):
    os.makedirs(gifdir)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.algo, args.env_name + ".pt"))
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
    obs = env.reset()
    ep_reward = 0.0
    for t in range(1, args.max_steps+1):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)

        print(action)
        next_state, reward, terminal, debug = env.step(action)
        masks.fill_(0.0 if terminal else 1.0)
        ep_reward += reward

        if terminal or t == args.max_steps:
            if terminal:
                solved += 1
            break

    eval_reward += ep_reward
    filename = gifdir + 'test_%d.gif' % (episode)
    render_func(filename)

avg_reward = eval_reward / args.eval_episodes
print('EVAL: Avg reward = %f | Solved = %d\n' % (avg_reward, solved))

# if render_func is not None:
#     render_func('human')
#
# if args.env_name.find('Bullet') > -1:
#     import pybullet as p
#
#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if (p.getBodyInfo(i)[0].decode() == "torso"):
#             torsoId = i
#
# while True:
#     with torch.no_grad():
#         value, action, _, recurrent_hidden_states = actor_critic.act(
#             obs, recurrent_hidden_states, masks, deterministic=args.det)
#
#     # Obser reward and next obs
#     obs, reward, done, _ = env.step(action)
#
#     masks.fill_(0.0 if done else 1.0)
#
#     if args.env_name.find('Bullet') > -1:
#         if torsoId > -1:
#             distance = 5
#             yaw = 0
#             humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
#             p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
#
#     if render_func is not None:
#         render_func('human')
