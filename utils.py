import os, re, sys, traceback
import yaml, logging, logging.handlers
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random

logger = logging.getLogger("Kalman_Filter")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='lstm_ekf.log', 
    format='%(levelname)s %(asctime)s in %(funcName)s() ' +
        '%(filename)s-%(lineno)s: %(message)s \n', level=logging.INFO)

state_file = "lstm_ekf.state"

def os_run(cmd):

    res = None
    try:
        res = os.popen(cmd).read()
        logger.debug("Ran cmd = " + cmd + ", output = " + str(res))
    except:
        ex = traceback.format_exc()
        logger.error("Error running '"+ str(cmd) +"': " + str(ex))
    return res 


def utilization(key, states, stateinfo): 
    maxval = get(stateinfo, key, 'max')
    return states[int(key)] / maxval if maxval else states[int(key)]


def get(o, *keys):
    for k in keys:
        if isinstance(k, list):
            o = get(o[k[0]], k[1:]) if len(k) else o
        elif not k in o:
            return None
        else:
            o = o[k]
    return o



def load_state():
    with open(state_file, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.error("Unable to load state", ex)
    return None


def save_state(runtime_state):
    with open(state_file, 'w') as out:
        try:
            out.write(yaml.dump(runtime_state))
        except yaml.YAMLError as ex:
            logger.error("Unable to save state", ex)
            return False
    return True



def repeat(v, n):
    if isinstance(v, type(lambda i:i)):
        return map(v, range(0,n))
    else:
        return map(lambda i: v, range(0,n))


