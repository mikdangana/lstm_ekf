import tensorflow as tf
import os, re, sys, traceback
import yaml, logging, logging.handlers
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random
from utils import *

logger = logging.getLogger("Kalman_Filter")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='lstm_ekf.log', 
    format='%(levelname)s %(asctime)s in %(funcName)s() ' +
        '%(filename)s-%(lineno)s: %(message)s \n', level=logging.INFO)

config = None
initialized = False


def do_action(ekf, msmts):
    stateinfo = {"1": "nWebServers", "2": "nDb", "3": "nSearch"} # TODO fix keys
    window_size = 2 # TODO: should be based on variance of a state variable
    state = list(map(lambda s: s[0], ekf.x_prior))  # state prediction
    to_lqn = lambda lst: reduce(lambda a,b: "["+str(a)+","+str(b)+"]", lst)
    logger.debug("state = " + str(state))

    for k,v in stateinfo.items():
        util = utilization(k, state, stateinfo)
        bounds = (int(util*100 - window_size), int(util*100 + window_size))
        window = range(1 if bounds[0] < 1 else bounds[0], bounds[1])
        logger.debug("window = " + str(window))
        os_run(get_config(['model-update-cmd'], [v, to_lqn(window)]))

    out = os_run(get_config('model-solve-cmd'))
    logger.debug("out = " + str(out) + ", rows = " + str(len(out.split("\n"))))
    util = lambda row: reduce(lambda a,b: a+b, list(map(float, row[3:])), 1)
    low = lambda a,b: a if util(a) < util(b) else b
    best = reduce(low, list(map(lambda l: l.split(", "), out.split("\n")[1:])))
    return run_action(best)


def run_action(action):
    run_state = load_state()
    run_state = {"app": 1, "db": 1, "solr": 1} if not run_state else run_state
    action = {k:v for k,v in zip(run_state.keys(), action[0:3])}

    for res,count in action:
        if count > run_state[res]:
            for step in get_steps("provision-cmds", res):
                os_run(step)
            run_state[res] = run_state[res] + 1
        elif count < run_state[res]:
            for step in get_steps("deprovision-cmds", res):
                os_run(step)
            run_state[res] = run_state[res] - 1
    save_state(run_state)
    return True


def get_steps(*cmd_path):
    steps = []
    variables = get_config("variables")
    cfg = get_config(cmd_path)

    for step in cfg if cfg else []:
        for cmd in step if isinstance(step, list) else [step]: 
            for k,v in variables.items():
                cmd = re.sub(r'<' + k + '>', v, cmd)
                steps.append(cmd)
    return steps


def load_config():
    global config
    with open("lstm_ekf.yaml", 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.error("Unable to load config", ex)
    return config


def get_config(path, params = []):
    global config
    if not config:
        load_config()
    val = get(config, path)
    for (i, param) in zip(range(0, len(params)), params):
        val = re.sub(r'<param' + str(i) + '>', str(param), str(val))
    if 'variables' in config:
        for k, v in config['variables'].items():
            k = "<" + str(k) + ">"
            logger.debug("k = " + str(k) + ", v = " + str(v))
            if isinstance(v, type("")):
                val = re.sub(r'' + k, v, str(val))
            elif isinstance(v, list) and (k in val):
                return [re.sub(r'' + k, vi, str(val)) for vi in v]
    return val


