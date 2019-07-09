#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

# wrapper for a POMDPs.jl environment in python using ZMQ for python/julia interop
# example from https://github.com/JuliaPOMDP/RLInterface.jl/issues/2

import subprocess
import zmq
import numpy as np
import gym
from gym.spaces import Box

from config import JULIA_ENV_DICT

FNULL = open(os.devnull, 'w')

class ZMQConnection:
    """
        Initialize a ZMQ connection given an IP address and a port.
    """

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://{}:{}".format(ip, port))

    @property
    def socket(self):
        return self._socket

    def sendreq(self, msg):
        self.socket.send_json(msg)
        respmsg = self.socket.recv_json()
        return respmsg

class ZMQEnv(gym.Env):
    def __init__(self,
                 env_name,  # name of the environment to load
                 # dictionary of parameters Dict{String,Any} passed to the env
                 # initialization
                 param_dict):
        self._action_space = None
        self._observation_space = None
        self.ep_count = 0

        if type(param_dict) is dict:
            self.params = param_dict
        else:
            self.params = vars(param_dict)

        self.zmq = False

    def setup_zmq(self, ip='127.0.0.1', port=9393):
        self._conn = ZMQConnection(ip, port)

        try:
            pid = subprocess.check_output(["pgrep", "-f", str(port)]).decode()
        except subprocess.CalledProcessError as e:
            print("[Py-INFO] pgrep failed because reasons. ({}):".format(e.returncode) , e.output.decode())
        else:
            try:
                os.kill(int(pid), 9)
                print("[Py-INFO] Killed existing ZMQ server at %s:%d!" % (ip, port))
            except ProcessLookupError as e:
                print("[Py-INFO] Tried to kill an old entry.")
            except ValueError as e:
                print("[Py-INFO] No existing ZMQ server at %s:%d. Moving on!" % (ip, port))

        self.julia = subprocess.Popen(["julia", "../../julia/scripts/zmq_server.jl",
                            "--port", str(port), "--ip", str(ip)])
        self.zmq = True
        self.ip = ip
        self.port = port
        print("[Py-INFO] Starting Julia subprocess and ZMQ Connection at %s:%d."
                    % (self.ip, self.port))

    def reset(self, args_dict=None, render=False):
        if args_dict is not None:
            self.params = vars(args_dict)

        if not self.zmq:
            self.setup_zmq(ip=self.params["ip"], port=self.params["port"])

        assert self.zmq
        # reset the environment
        data = self._conn.sendreq({"cmd": "reset", "params": self.params})
        assert "obs" in data
        obs = data["obs"]

        # get observation space information
        data = self._conn.sendreq({"cmd": "observation_space"})
        lo, hi = np.array(data["lo"]), np.array(data["hi"])
        if np.all(lo == None):
            lo.fill(-np.inf)
        if np.all(hi == None):
            hi.fill(-np.inf)
        self._observation_space = Box(lo, hi)

        # get action space information
        data = self._conn.sendreq({"cmd": "action_space"})
        lo, hi = data["lo"], data["hi"]
        self._action_space = Box(np.array(lo), np.array(hi))

        return obs

    def render(self, filename="default.gif"):
        data = self._conn.sendreq({"cmd": "render", "filename": filename})

    def step(self, action):
        data = self._conn.sendreq({"cmd": "step", "a": str(action[0]),
                                                    "delta": str(action[1])})
        assert "obs" in data
        assert "rew" in data
        assert "done" in data
        assert "info" in data

        infos = dict()
        infos['egostate'] = data["info"]

        if data["done"]:
            self.ep_count += 1

        return data["obs"], data["rew"], data["done"], infos

    def kill(self):
        print("[Py-INFO] Kill Julia subprocess and close ZMQ Connection at %s:%d."
                    % (self.ip, self.port))
        self.julia.terminate()
        self.zmq = False

    def close(self):
        self.kill()

    @property
    def action_space(self):
        if self._action_space is None:
            print("InitialisationError: Must reset() environment first.")

        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            print("InitialisationError: Must reset() environment first.")

        return self._observation_space

if __name__ == '__main__':
    from external import *
    import envs # registers the environment

    args = get_args()
    env = gym.make(args.env_name)
    obs = env.reset(args)
    while True:
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)

        print("s ->{}".format(obs))
        print("a ->{}".format(action))
        print("sp->{}".format(ob))
        print("r ->{}".format(reward))

        obs = ob
        if done:
            break

    env.close()
    env.kill()
