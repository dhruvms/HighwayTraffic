import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='discount factor for rewards (default: 0.9)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--action-loss-coef',
        type=float,
        default=1.0,
        help='action loss coefficient (default: 1.0)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=68845, help='random seed (default: 68845)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='HighwayTraffic-v1',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--save-dir',
        default='../data/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    parser.add_argument(
        '--load-dir',
        default='../algs/trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')

    # HighwayTraffic simulation related parameters
    parser.add_argument('--length', default=1000.0, type=float,
        help='Roadway length')
    parser.add_argument('--lanes', default=3, type=int,
        help='Number of lanes on roadway')
    parser.add_argument('--cars', default=30, type=int,
        help='Number of cars on roadway')
    parser.add_argument('--testcars', default=30, type=int,
        help='Number of cars on roadway')
    parser.add_argument('--stadium', action='store_true', default=False,
        help='stadium roadway')
    parser.add_argument('--change', action='store_true', default=False,
        help='change lanes')
    parser.add_argument('--both', action='store_true', default=False,
        help='change or follow')
    parser.add_argument('--v-des', default=5.0, type=float,
        help='Max desired velocity')
    parser.add_argument('--dt', default=0.2, type=float,
        help='Simulation timestep')
    parser.add_argument('--ego-dim', default=9, type=int,
        help='Egovehicle feature dimension')
    parser.add_argument('--other-dim', default=7, type=int,
        help='Other vehicle feature dimension')
    parser.add_argument('--occupancy', action='store_true', default=False,
        help='occupancy grid observation')
    parser.add_argument('--fov', default=50, type=int,
        help='Field of view')
    parser.add_argument('--j-cost', default=0.001, type=float,
        help='Jerk cost')
    parser.add_argument('--d-cost', default=0.01, type=float,
        help='Steering rate cost')
    parser.add_argument('--a-cost', default=0.1, type=float,
        help='Acceleration cost')
    parser.add_argument('--v-cost', default=2.5, type=float,
        help='Desired velocity deviation cost')
    parser.add_argument('--phi-cost', default=1.0, type=float,
        help='Lane heading deviation cost')
    parser.add_argument('--t-cost', default=10.0, type=float,
        help='Lane lateral displacement cost')
    parser.add_argument('--end-cost', default=1.0, type=float,
        help='Deadend cost')
    parser.add_argument('--beta-dist', action='store_true', default=False,
        help='use beta distribution policy')
    parser.add_argument('--clamp-in-sim', action='store_true', default=False,
        help='clamp action inside simulator')
    parser.add_argument('--extra-deadends', action='store_true', default=False,
        help='add deadends in other lanes')
    parser.add_argument('--gap', default=1.1, type=float,
        help='Gap between cars')

    parser.add_argument('--max-steps', type=int, default=256,
                        help='Max steps per episode')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Evaluation episodes')
    parser.add_argument('--eval', action='store_true', default=False,
        help='in eval mode or not')
    parser.add_argument('--norm-obs', action='store_true', default=True,
        help='in eval mode or not')
    parser.add_argument('--hri', action='store_true', default=False,
        help='HRI specific test case')
    parser.add_argument('--curriculum', action='store_true', default=False,
        help='use randomised curriculum')
    parser.add_argument('--stopgo', action='store_true', default=False,
        help='add stop and go behaviour')

    parser.add_argument('--log', action='store_true', default=False,
        help='tensorboard logging')
    parser.add_argument('--ip', type=str, default="127.0.0.1",
            help='ZMQ Server IP address')
    parser.add_argument('--base-port', type=int, default=9394,
            help='ZMQ Server port number')
    parser.add_argument('--model-name', nargs='+', type=int,
            help='Terms to include in model name')

    parser.add_argument('--eval-mode', type=str, default="mixed",
            help='types of other vehicles (mixed/cooperative/aggressive)')
    parser.add_argument('--video', action='store_true', default=False,
        help='save video')
    parser.add_argument('--write-data', action='store_true', default=False,
        help='save data file')
    parser.add_argument('--eval-folder', type=str, default='',
            help='eval save folder name')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
