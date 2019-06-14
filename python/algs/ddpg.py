#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import time
import argparse
import numpy as np
import gym
import torch

from models import A2C

def evaluate(agent, env, args, logfile, render=False, log=True):
    agent.eval()

    eval_reward = 0.0
    solved = 0
    for episode in range(1, args.eval_episodes+1):
        state = env.reset()
        ep_reward = 0.0
        for t in range(args.max_steps):
            if render:
                env.render()

            action = agent.select_action(state)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminal, debug = env.step(action)
            state = next_state
            ep_reward += reward

            if terminal or t == args.max_steps-1:
                if terminal:
                    solved += 1
                break

        eval_reward += ep_reward

    if log:
        avg_reward = eval_reward / args.eval_episodes
        logfile.write('EVAL: Avg reward = %f | Solved = %d\n' % (avg_reward, solved))
        logfile.flush()

def train_ddpg(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    savedir = args.logdir + 'ddpg/' + args.env + '/' + timestr + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    env = gym.make(args.env)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_lim = float(env.action_space.high[0])

    agent = A2C(state_dim, action_dim, action_lim,
                update_type=args.update, batch_size=args.batch_size)
    agent.train()

    if args.seed:
        print("Random Seed: {}".format(args.random_seed))
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    avg_reward = 0.0
    logfile = open(savedir + 'log.txt', 'w+')
    for episode in range(1, args.episodes+1):
        ep_reward = 0.0
        state = env.reset()
        agent.reset_noise()
        for t in range(args.max_steps):
            action = agent.select_action(state)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminal, debug = env.step(action)
            agent.append(state, action, reward, next_state, float(terminal))
            state = next_state
            ep_reward += reward
            avg_reward += reward

            if args.update_always:
                junk = np.random.normal(np.random.randint(-10, 10), np.random.random() + 5.0)
                tot_actor_loss = junk
                tot_critic_loss = junk
                for b in range(args.update_batches):
                    actor_loss, critic_loss = agent.update(target_noise=False)
                    if (actor_loss is not None) and (critic_loss is not None):
                        tot_actor_loss += actor_loss
                        tot_critic_loss += critic_loss
                if (tot_actor_loss != junk) and (tot_critic_loss != junk):
                    tot_actor_loss -= junk
                    tot_critic_loss -= junk
                    tot_actor_loss /= args.update_batches
                    tot_critic_loss /= args.update_batches
                    logfile.write('LOSS: %d,%f,%f\n' % (episode, tot_actor_loss, tot_critic_loss))
                    logfile.flush()

            if terminal or t == args.max_steps-1:
                junk = np.random.normal(np.random.randint(-10, 10), np.random.random() + 5.0)
                tot_actor_loss = junk
                tot_critic_loss = junk
                for b in range(args.update_batches):
                    actor_loss, critic_loss = agent.update(target_noise=False)
                    if (actor_loss is not None) and (critic_loss is not None):
                        tot_actor_loss += actor_loss
                        tot_critic_loss += critic_loss
                if (tot_actor_loss != junk) and (tot_critic_loss != junk):
                    tot_actor_loss -= junk
                    tot_critic_loss -= junk
                    tot_actor_loss /= args.update_batches
                    tot_critic_loss /= args.update_batches
                    logfile.write('LOSS: %d,%f,%f\n' % (episode, tot_actor_loss, tot_critic_loss))
                    logfile.flush()
                break

        logfile.write('%d,%f\n' % (episode, ep_reward))
        logfile.flush()

        if (avg_reward/args.done_window) > args.done_reward:
            print("Task Solved!")
            args.test_folder = savedir
            agent.save(savedir, episode, solved=True)
            break

        if (episode % args.save_every) == 0:
            agent.save(savedir, episode)

        if (episode % args.done_window) == 0:
            avg_reward /= args.done_window
            ep_start = episode - args.done_window + 1
            print('Episodes %d - %d | Average reward = %f' % (ep_start, episode,
                                                                    avg_reward))
            avg_reward = 0.0

        if (episode % args.eval_every) == 0:
            # print("Evaluating!")
            evaluate(agent, env, args, logfile)
            agent.train()

    print("Testing!")
    evaluate(agent, env, args, logfile)
    logfile.close()

def test_ddpg(args):
    if not args.test_folder:
        print('ERROR: Need trained model folder.')
        return

    env = gym.make(args.env)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_lim = float(env.action_space.high[0])

    agent = A2C(state_dim, action_dim, action_lim,
                update_type=args.update, batch_size=args.batch_size)
    agent.load_actor(args.test_folder)

    evaluate(agent, env, args, None, render=True, log=False)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--env', type=str, default='LunarLanderContinuous-v2',
                        help='Gym environment')
    parser.add_argument('--logdir', type=str, default='../logs/',
                        help='Log data folder')
    parser.add_argument('--update', type=str, default='soft',
                        help='Soft vs hard update')
    parser.add_argument('--seed', action='store_true', help='Manually seed')
    parser.add_argument('--random_seed', type=int, default=68845,
                        help='Random seed')
    parser.add_argument('--episodes', type=int, default=1500,
                        help='Training episodes')
    parser.add_argument('--max-steps', type=int, default=2000,
                        help='Max steps per episode')
    parser.add_argument('--update-always', action='store_true',
                        help='Update network every timestep')
    parser.add_argument('--update-batches', type=int, default=100,
                        help='Number of minibatches sampled during each update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training minibatch size')
    parser.add_argument('--done-window', type=int, default=50,
                        help='Number of episodes horizon')
    parser.add_argument('--done-reward', type=float, default=200.0,
                        help='Solved average reward score')
    parser.add_argument('--save-every', type=int, default=100,
                        help='Save network frequency')

    parser.add_argument('--eval-every', type=int, default=100,
                        help='Network eval frequency')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Evaluation episodes')

    parser.add_argument('--test-folder', type=str, default='',
                        help='Folder with trained model data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # train_ddpg(args)
    test_ddpg(args)
