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
from utils import Logger, get_model_name

import multiprocessing as mp
PYTHON_EXEC = '/home/dsaxena/work/code/python/venvs/p36ws/bin/python'
mp.set_executable(PYTHON_EXEC)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 4, sharey=True)

def map_to_range(input, in_start, in_end, out_start, out_end):
    scale = ((out_end - out_start) / (in_end - in_start))
    output = out_start + scale * (input - in_start)
    return output

def main():
    args = get_args()

    if args.model_name is None:
        args.model_name = [0, 1, 2, 3, 4, 5, 6, 7]
    model_name = get_model_name(args)

    expdir = args.save_dir + model_name + '/'
    args.log_dir = expdir + 'logs/'
    args.model_dir = expdir + 'model/'
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    LOGGER = Logger(args.log_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    envs = make_vec_envs(args, device, False)
    action_high = envs.action_space.high
    action_low = envs.action_space.low

    other_cars = args.cars > 1
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        other_cars=other_cars, ego_dim=args.ego_dim,
        beta_dist=args.beta_dist,
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
            args.action_loss_coef,
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
            lr = utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
            LOGGER.scalar_summary('stats/lr', lr, j + 1)

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
            # ax[0].imshow(obs.data.cpu().numpy()[0, -5, :, :])
            # ax[1].imshow(obs.data.cpu().numpy()[0, -4, :, :])
            # ax[2].imshow(obs.data.cpu().numpy()[0, -3, :, :])
            # ax[3].imshow(obs.data.cpu().numpy()[0, -2, :, :])
            # plt.suptitle("Action: ({:.3f}, {:.3f}), Reward: {:.3f}"
            #                 .format(action[0][0], action[0][1], reward[0][0]))
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
        total_loss = value_loss + action_loss - dist_entropy
        weighted_loss = (value_loss * args.value_loss_coef +
                            action_loss * args.action_loss_coef -
                            dist_entropy * args.entropy_coef)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.model_dir != "":
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(args.model_dir, args.algo + ".pt"))

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

            LOGGER.scalar_summary('losses/value_loss', value_loss, total_num_steps)
            LOGGER.scalar_summary('losses/action_loss', action_loss, total_num_steps)
            LOGGER.scalar_summary('losses/dist_entropy', dist_entropy, total_num_steps)
            LOGGER.scalar_summary('losses/total_loss', total_loss, total_num_steps)
            LOGGER.scalar_summary('losses/weighted_loss', weighted_loss, total_num_steps)

            LOGGER.scalar_summary('rewards/mean', np.mean(episode_rewards), total_num_steps)
            LOGGER.scalar_summary('rewards/median', np.median(episode_rewards), total_num_steps)
            LOGGER.scalar_summary('rewards/min', np.min(episode_rewards), total_num_steps)
            LOGGER.scalar_summary('rewards/max', np.max(episode_rewards), total_num_steps)

            # for tag, value in actor_critic.named_parameters():
            #     tag = tag.replace('.', '/')
            #     LOGGER.histo_summary(tag, value.data.cpu().numpy(), total_num_steps)
            #     LOGGER.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), total_num_steps)

            action_np = action.data.cpu().numpy()

            if args.beta_dist:
                action_np = map_to_range(action_np, 0.0, 1.0, action_low, action_high)

            jerk = action_np[:, 0]
            steering_rate = action_np[:, 1]

            LOGGER.scalar_summary('actions/jerk_mean', np.mean(jerk), total_num_steps)
            LOGGER.scalar_summary('actions/jerk_median', np.median(jerk), total_num_steps)
            LOGGER.scalar_summary('actions/jerk_min', np.min(jerk), total_num_steps)
            LOGGER.scalar_summary('actions/jerk_max', np.max(jerk), total_num_steps)
            LOGGER.scalar_summary('actions/jerk_0', jerk[0], total_num_steps)

            LOGGER.scalar_summary('actions/steering_rate_mean', np.mean(steering_rate), total_num_steps)
            LOGGER.scalar_summary('actions/steering_rate_median', np.median(steering_rate), total_num_steps)
            LOGGER.scalar_summary('actions/steering_rate_min', np.min(steering_rate), total_num_steps)
            LOGGER.scalar_summary('actions/steering_rate_max', np.max(steering_rate), total_num_steps)
            LOGGER.scalar_summary('actions/steering_rate_0', steering_rate[0], total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args, device, LOGGER, total_num_steps)

    envs.close()


if __name__ == "__main__":
    main()
