import os
#!/usr/bin/env python
import sys
sys.path.append(os.path.abspath('..'))

import argparse
import numpy as np
import time

from envs import ZMQEnvTCN

def gendata(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    savedir = args.logdir + 'tcn-data/' + timestr + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    env = ZMQEnvTCN()
    env.reset(args)

    DATA = None

    for episode in range(ep_start, args.episodes+1):
        obs = env.reset()
        print(obs)
        for t in range(1, args.max_steps+1):
            obs, reward, terminal, debug = env.step()
            print(obs)

            if DATA is None:
                DATA = np.array(obs)
            else:
                DATA = np.vstack([DATA, np.array(obs)])

            if terminal or t == args.max_steps:
                break

    np.savetxt(savedir + "DATA.csv", DATA, delimiter=",")
    params = np.array([args.sampled, args.features, args.neighbours])
    np.savetxt(savedir + "params.csv", params, delimiter=",")

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--episodes', type=int, default=1000,
                        help='Training episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--logdir', type=str, default='../logs/',
                        help='Log data folder')

    parser.add_argument('--length', default=100.0, type=float,
        help='Roadway length')
    parser.add_argument('--lanes', default=3, type=int,
        help='Number of lanes on roadway')
    parser.add_argument('--cars', default=20, type=int,
        help='Number of cars on roadway')
    parser.add_argument('--dt', default=0.2, type=float,
        help='Simulation timestep')
    parser.add_argument('--stadium', action='store_true', default=False,
        help='stadium roadway')
    parser.add_argument('--v-des', default=15.0, type=float,
        help='Max desired velocity')
    parser.add_argument('--sampled', default=10, type=int,
        help='Number of cars to sample for neighbour features')
    parser.add_argument('--features', default=7, type=int,
        help='Neighbour featurevec dimension')
    parser.add_argument('--neighbours', default=10, type=int,
        help='Max number of neighbours allowed')
    parser.add_argument('--radius', default=50.0, type=float,
        help='Neighbour distance check radius')

    parser.add_argument('--ip', type=str, default="127.0.0.1",
            help='ZMQ Server IP address')
    parser.add_argument('--port', type=int, default=9393,
            help='ZMQ Server port number')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gendata(args)
