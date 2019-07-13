#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import subprocess
import zmq
import numpy as np

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

class ZMQEnvTCN():
    def __init__(self, param_dict=None):
        if param_dict is not None:
            if type(param_dict) is dict:
                self.params = param_dict
            else:
                self.params = vars(param_dict)

        self.zmq = False

    def setup_zmq(self, ip='127.0.0.1', port=9393):
        self._conn = ZMQConnection(ip, port)

        self.julia = subprocess.Popen(["julia",
                            "../../julia/scripts/zmq_server_tcn.jl",
                            "--port", str(port), "--ip", str(ip)])
        self.zmq = True
        self.ip = ip
        self.port = port
        print("[Py-INFO] Starting Julia subprocess and ZMQ Connection at %s:%d."
                    % (self.ip, self.port))

    def reset(self, args_dict=None):
        if args_dict is not None:
            print("[Py-INFO] Reset with dictionary!")
            self.params = vars(args_dict)

        if not self.zmq:
            self.setup_zmq(ip=self.params["ip"], port=self.params["port"])

        assert self.zmq
        # reset the environment
        data = self._conn.sendreq({"cmd": "reset", "params": self.params})
        assert "obs" in data
        obs = data["obs"]

        return obs

    def render(self, filename="default.gif"):
        data = self._conn.sendreq({"cmd": "render", "filename": filename})

    def step(self):
        data = self._conn.sendreq({"cmd": "step"})
        assert "obs" in data
        assert "rew" in data
        assert "done" in data
        assert "info" in data

        infos = dict()
        infos['info'] = data["info"]

        if data["done"] and self.params["eval"]:
            filename = "eval_ep.gif"
            self.render(filename)

        return data["obs"], data["rew"], data["done"], infos

    def kill(self):
        print("[Py-INFO] Kill and close ZMQ Connection at %s:%d."
                    % (self.ip, self.port))
        self.julia.terminate()
        self.zmq = False

    def close(self):
        self.kill()
