import os
#!/usr/bin/env python
import sys
sys.path.append(os.path.abspath('..'))

import time
import argparse
import numpy as np
import gym
import torch

from models import A2C
import envs # registers the environment

def evaluate(agent, env, args, logfile, render_episode=False, log=True):
    agent.eval()

    eval_reward = 0.0
    solved = 0
    for episode in range(1, args.eval_episodes+1):
        state = env.reset()
        ep_reward = 0.0
        for t in range(1, args.max_steps+1):

            action = agent.select_action(np.array(state))
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminal, debug = env.step(action)

            if args.debug:
                s_f, t_f, phi_f, v_ego = debug[:4]
                print("EVAL | (s, t, phi, v) = (%3.2f, %3.2f, %3.2f, %3.2f)" % (s_f, t_f, phi_f, v_ego))

            state = next_state
            ep_reward += reward

            if terminal or t == args.max_steps:
                if terminal:
                    solved += 1
                break

        eval_reward += ep_reward

        if render_episode:
            try:
                model = args.actor_model.split("_")[-1]
                model = model.split(".")[0]
                gifdir = args.actor_model[:args.actor_model.rfind("/")+1] + 'gifs_%s/' % model
                if not os.path.exists(gifdir):
                    os.makedirs(gifdir)
                filename = gifdir + 'test_%d.gif' % (episode)
            except:
                filename = 'test_%d.gif' % (episode)
            env.render(filename=os.path.abspath(filename))

    if log:
        avg_reward = eval_reward / args.eval_episodes
        logfile.write('EVAL: Avg reward = %f | Solved = %d\n' % (avg_reward, solved))
        logfile.flush()

def train_ddpg(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    change = args.change * "-change" + (not args.change) * "-follow"
    savedir = args.logdir + 'julia-sim/' + args.env.lower() + change + '/' + timestr + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    env = gym.make(args.env)
    env.reset(args)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    action_lim = env.action_space.high
    other_cars = args.cars > 1

    agent = A2C(state_dim, action_dim, action_lim,
                update_type=args.update, batch_size=args.batch_size,
                other_cars=other_cars, ego_dim=args.ego_dim)
    ep_start = 1
    if args.resume_train:
        if not args.actor_model:
            print('ERROR: Need trained model folder.')
            return
        agent.load_all(args.actor_model)
        ep_start = int(args.actor_model.split('_')[-1].split('.')[0][2:])
    agent.train()

    if args.seed:
        print("Random Seed: {}".format(args.random_seed))
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    avg_reward = 0.0
    logfile = open(savedir + 'log.txt', 'w+')
    for episode in range(ep_start, args.episodes+1):
        ep_reward = 0.0
        state = env.reset()
        agent.reset_noise()
        for t in range(1, args.max_steps+1):
            action = agent.select_action(np.array(state))
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminal, debug = env.step(action)
            if args.debug:
                s_f, t_f, phi_f, v_ego = debug[:4]
                print("(s, t, phi, v) = (%3.2f, %3.2f, %3.2f, %3.2f)" % (s_f, t_f, phi_f, v_ego))
                logfile.write("(Episode, Step): (%d, %d) | (s, t, phi, v) = (%3.2f, %3.2f, %3.2f, %3.2f)\n" % (episode, t, s_f, t_f, phi_f, v_ego))
                logfile.flush()

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

            if terminal or t == args.max_steps:
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

        if (episode % args.save_every) == 0:
            agent.save(savedir, episode)

        if (episode % args.eval_every) == 0:
            avg_reward /= args.eval_every
            ep_start = episode - args.eval_every + 1
            print('Episodes %d - %d | Average reward = %f' % (ep_start, episode,
                                                                    avg_reward))
            avg_reward = 0.0

            # print("Evaluating!")
            evaluate(agent, env, args, logfile)
            agent.train()

    print("Testing!")
    evaluate(agent, env, args, logfile)
    logfile.close()

def test_ddpg(args):
    if not args.actor_model:
        print('ERROR: Need trained model folder.')
        return

    env = gym.make(args.env)
    env.reset(args)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    action_lim = env.action_space.high
    other_cars = args.cars > 1

    agent = A2C(state_dim, action_dim, action_lim,
                update_type=args.update, batch_size=args.batch_size,
                other_cars=other_cars, ego_dim=args.ego_dim)
    agent.load_actor(args.actor_model)

    evaluate(agent, env, args, None, render_episode=True, log=False)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--env', type=str, default='LaneFollow-v0',
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
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--update-always', action='store_true',
                        help='Update network every timestep')
    parser.add_argument('--update-batches', type=int, default=100,
                        help='Number of minibatches sampled during each update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training minibatch size')
    parser.add_argument('--save-every', type=int, default=100,
                        help='Save network frequency')

    parser.add_argument('--eval-every', type=int, default=100,
                        help='Network eval frequency')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Evaluation episodes')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='in eval mode or not')

    parser.add_argument('--actor-model', type=str, default='',
                        help='Path to trained actor model')

    parser.add_argument('--debug', action='store_true',
        help='Print debug info')
    parser.add_argument('--resume-train', action='store_true',
            help='Print debug info')

    # HighwayTraffic simulation related parameters
    parser.add_argument('--length', default=1000.0, type=float,
        help='Roadway length')
    parser.add_argument('--lanes', default=3, type=int,
        help='Number of lanes on roadway')
    parser.add_argument('--cars', default=30, type=int,
        help='Number of cars on roadway')
    parser.add_argument('--stadium', action='store_true', default=False,
        help='stadium roadway')
    parser.add_argument('--change', action='store_true', default=False,
        help='change lanes')
    parser.add_argument('--v-des', default=15.0, type=float,
        help='Max desired velocity')
    parser.add_argument('--dt', default=0.2, type=float,
        help='Simulation timestep')
    parser.add_argument('--ego-dim', default=8, type=int,
        help='Egovehicle feature dimension')
    parser.add_argument('--other-dim', default=7, type=int,
        help='Other vehicle feature dimension')
    parser.add_argument('--j-cost', default=0.01, type=float,
        help='Jerk cost')
    parser.add_argument('--d-cost', default=0.02, type=float,
        help='Steering rate cost')
    parser.add_argument('--a-cost', default=0.01, type=float,
        help='Acceleration cost')
    parser.add_argument('--v-cost', default=0.5, type=float,
        help='Desired velocity deviation cost')
    parser.add_argument('--phi-cost', default=1.0, type=float,
        help='Lane heading deviation cost')
    parser.add_argument('--t-cost', default=2.0, type=float,
        help='Lane lateral displacement cost')

    parser.add_argument('--ip', type=str, default="127.0.0.1",
            help='ZMQ Server IP address')
    parser.add_argument('--port', type=int, default=9393,
            help='ZMQ Server port number')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # train_ddpg(args)
    test_ddpg(args)
