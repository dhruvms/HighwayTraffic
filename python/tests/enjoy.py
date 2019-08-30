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

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 4, sharey=True)

args = get_args()
if args.model_name is None:
    args.model_name = [0, 1, 2, 3, 4, 5, 6, 7, 8]

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
if args.eval_folder:
    viddir = expdir + 'vids/' + args.eval_folder + '/'
else:
    viddir = expdir + 'vids/' + timestr + '/'
if not os.path.exists(viddir):
    os.makedirs(viddir)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.model_dir, args.algo + ".pt"), map_location=device)
actor_critic.eval()
# print(actor_critic)

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
    filename = viddir + 'test_%d' % (episode)
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
        # ax[0].imshow(obs.data.cpu().numpy()[0, -5, :, :])
        # ax[1].imshow(obs.data.cpu().numpy()[0, -4, :, :])
        # ax[2].imshow(obs.data.cpu().numpy()[0, -3, :, :])
        # ax[3].imshow(obs.data.cpu().numpy()[0, -2, :, :])
        # ax[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # ax[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # ax[2].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # ax[3].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        # plt.suptitle("Action: ({:.3f}, {:.3f}), Reward: {:.3f}"
        #                 .format(action[0][0], action[0][1], reward[0][0]))
        # plt.savefig('temp.png', dpi=300, bbox_inches='tight')
        masks.fill_(0.0 if terminal else 1.0)
        ep_reward += reward

        if terminal or t == args.max_steps:
            if terminal:
                solved += 1
            default_filename = 'eval_ep'
            try:
                if args.video:
                    os.rename(default_filename + '.mp4', filename + '.mp4')

                os.rename(default_filename + '.dat', filename + '.dat')
            except FileNotFoundError:
                render_func(filename + '.mp4')

            break

    # print('Eval: Episode reward = %f' % (ep_reward))
    eval_reward += ep_reward

avg_reward = eval_reward / args.eval_episodes
# print('Eval: Avg reward = %f | Unsolved = %d' % (avg_reward, solved))
