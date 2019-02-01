import os, re, sys, traceback
import yaml, logging, logging.handlers
import pickle
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract, size, append
from numpy.linalg import inv
from functools import reduce
from random import random
from time import time

logger = logging.getLogger("Kalman_Filter")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='lstm_ekf.log', 
    format='%(levelname)s %(asctime)s in %(funcName)s() ' +
        '%(filename)s-%(lineno)s: %(message)s \n', level=logging.DEBUG)

state_file = "lstm_ekf.state"

def os_run(cmd):

    res = None
    try:
        logger.debug("Running cmd " + str(cmd))
        res = os.popen(cmd).read()
        logger.debug("Ran cmd = " + cmd + ", output = " + str(len(res)))
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
            o = get(o[k[0]], k[1:]) if len(k) and o else o
        elif not k in o:
            return None
        else:
            o = o[k]
    return o



def load_state():
    try:
        with open(state_file, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as ex:
                logger.error("Unable to load state", ex)
    except FileNotFoundError:
        logger.error("State file not found.")
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


def avg(seq):
    size = len(list(seq))
    return sum(seq) / (size if size else size+1)


def flatlist(matrix):
    v = array(matrix);
    v.resize(size(matrix))
    return list(v)


def pickleconc(filename, value):
    history = pickleload(filename) or []
    pickledump(filename, history + value)


def pickleadd(filename, value):
    history = pickleload(filename) or []
    pickledump(filename, history + [value])


def pickledump(filename, value):
    try:
        with open(filename, 'wb') as f:
            return pickle.dump(value, f)
    except FileNotFoundError:
        logger.error(str(filename) + " not found.")
    return None


def pickleload(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(str(filename) + " not found.")
    return None


def twod(lst):
    data = array([[]])
    for row in lst:
        data = append(data, row)
    data.resize(len(lst), len(lst[0]))
    return data
