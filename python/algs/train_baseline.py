#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from external import *
import envs # registers the environment

import multiprocessing as mp
PYTHON_EXEC = '/home/dsaxena/work/code/python/venvs/p36ws/bin/python'
mp.set_executable(PYTHON_EXEC)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 4, sharey=True)

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    envs = make_vec_envs(args, device, False)
    # action_space_hi = envs.action_space.high
    # action_space_lo = envs.action_space.low

    other_cars = args.cars > 1
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        other_cars=other_cars, ego_dim=args.ego_dim,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    print(actor_critic)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    best_median = 0.0

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # # Clamp action to limits
            # torch.clamp_(action[:, 0], action_space_lo[0], action_space_hi[0])
            # torch.clamp_(action[:, 1], action_space_lo[1], action_space_hi[1])
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            # ax[0].imshow(obs.data.cpu().numpy()[0, -4, :, :])
            # ax[1].imshow(obs.data.cpu().numpy()[0, -3, :, :])
            # ax[2].imshow(obs.data.cpu().numpy()[0, -2, :, :])
            # ax[3].imshow(obs.data.cpu().numpy()[0, -1, :, :])
            # plt.pause(0.00001)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        median_ep_reward = np.median(episode_rewards)
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            change = args.change * "-Change" + (not args.change) * "-Follow"
            cars = "-{}cars".format(args.cars)
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + change + cars + ".pt"))

            # if median_ep_reward > best_median:
            #     best_median = median_ep_reward
            #     torch.save([
            #         actor_critic,
            #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            #     ], os.path.join(save_path, args.env_name + change + cars + "-best" + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n\tDist Entropy: {:.3f}, Value Loss: {:.3f}, Action Loss: {:.3f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    envs.close()


if __name__ == "__main__":
    main()
