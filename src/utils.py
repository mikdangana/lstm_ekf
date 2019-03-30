import os, re, sys, traceback
import yaml, logging, logging.handlers
import pickle, subprocess
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract, size, append, transpose
from numpy import gradient, mean, std, outer, vstack, concatenate
from numpy.linalg import inv, lstsq
from scipy import stats
from functools import reduce
from random import random
from time import time

logger = logging.getLogger("Utils")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='lstm_ekf.log', 
    format='%(levelname)s %(asctime)s in %(funcName)s() ' +
        '%(filename)s-%(lineno)s: %(message)s \n', level=logging.DEBUG)

state_file = "lstm_ekf.state"
procs = []


def os_run(cmd):

    res = None
    if isinstance(cmd, list):
        for c in cmd:
           res = res + os_run(c) + "\n"
        return res

    try:
        logger.debug("Running cmd " + str(cmd) + ", inst = " + str((isinstance(cmd, list), isinstance(cmd, type("")))))
        res = os.popen(cmd + " 2>>lstm_ekf.log").read()
        logger.debug("Ran cmd,out = " + str((cmd, len(res))))
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


def merge_state(delta):
    state = load_state() or {}
    logger.info("state, delta = " + str((state, delta)))
    state.update(delta)
    save_state(state)
    return state


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


def pickleconc(filename, values):
    history = pickleload(filename) or []
    pickledump(filename, history + values)


def pickleadd(filename, value):
    history = pickleload(filename) or []
    pickledump(filename, history + [value])


def pickledump(filename, value):
    try:
        with open(filename, 'wb') as f:
            return pickle.dump(value, f)
    except(FileNotFoundError, pickle.PicklingError):
        logger.error(str(filename) + " erro " + str(sys.exec_info()[0]))
    return None


def pickleload(filename):
    try:
        if os.path.getsize(filename) > 0:
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception: #(FileNotFoundError, EOFError, pickle.UnpicklingError):
        logger.error(str(filename) + " error " + str(sys.exc_info()[0]))
    return []


def find(lst, val):
    ids = [i if v==val or not val else -1 for i,v in zip(range(len(lst)), lst)]
    return list(filter(lambda v: v>=0, ids))


def sublist(lst, ids):
    return [v for i,v in filter(lambda v:v[0] in ids, zip(range(len(lst)),lst))]


# Approximates A for A*x = y
def solve_linear(x, ys, m_c = None):
    y_avg = array(list(map(sum, array(ys).T))).T / len(ys)
    def project(x1): 
        if len(shape(x1)) > 1 and shape(x1)[1]>1:
            x1 = array(list(map(sum, x1.T))).T / len(x1) 
        elif len(x1) < len(y_avg):
            x1 = concatenate((array(x1), ones(len(y_avg) - len(x1))))
        x1 = x1.reshape(len(x1), 1)
        logger.debug("project.x1 = " + str(x1) + ", shape = " + str(x1.shape))
        return x1
    if m_c:
        return (m_c[0], m_c[1], project)
    hx = project(x).T[0]
    A = vstack([array(hx), ones(len(hx))]).T
    logger.debug("A=" + str(A) + ", y=" + str(ys) + ", y_avg = " + str(y_avg))
    m, c = lstsq(A, y_avg, rcond=None)[0]
    logger.debug("m,c = " + str((m,c)))
    return (m, c, project)


def rotate_right(m):
    return transpose(concatenate([transpose(m[:,1:]), transpose(m[:,0:1])]))


def isconverged(data, confidence=0.95):
    return len(list(filter(lambda c: c>=confidence, convergence(data))))>0


def convergence(data):
    (stat, stds, window, step, tolerance) = ([], [], 3, int(len(data)/10), 0.03)
    for i in range(int(len(data)/step)):
        stds.append(std(data[i*step:(i+1) * step]))
        win = stds[-window:]
        m = mean(win) 
        (l, u) = (m-tolerance, m+tolerance) 
        confidence = len(list(filter(lambda d: d>=l and d<=u, win)))/len(win)
        stat.append(confidence)
    logger.info("convergence.confidences = " + str(stat[1:]) + ", data = " + str(shape(data)) + ", stds = " + str(stds))
    return stat[1:] # Ignore 1st window, its always 100% by definition


def twod(lst):
    data = array([[]])
    for row in lst:
        data = append(data, row)
    data.resize(len(lst), len(lst[0]) if len(shape(lst))>1 else 1)
    return data


# size of diag is row_count
# size of half is ( row_count^2 - row_count ) / 2
def symmetric(diag, half = None):
    (data, dmatrix, offset, rows) = ([], [], 0, len(diag))
    for i in range(rows):
        data.append([])
        dmatrix.append([])
        for j in range(rows):
            if i==j:
                dmatrix[-1].append(diag[i])
                data[-1].append(0)
            elif j > i:
                data[-1].append(half[offset] if half else 0)
                offset = offset + 1
                dmatrix[-1].append(0)
            else:
                data[-1].append(0)
                dmatrix[-1].append(0)
    upper = twod(data)
    return upper + transpose(upper) + twod(dmatrix)


if __name__ == "__main__":
    print(symmetric([1,2,3], [4,5,6]))
    print(symmetric([1,2,3]))
    print(symmetric([1,2,3,4], [5,6,7,8,9,10]))
    (x, y) = (array([5,3,8]), array([15,4,23]))
    (m, c, _) = solve_linear(x, [y])
    print("x = " + str(x) + ", y = " + str(y) + ", sol = " + str(m*x+c))
    print(os_run("wine lqns testbed.lqn"))
    print(sublist([1,2,3,4,5], [1,3]))
    print(merge_state({"abc": {"def": 1, "ghi": 2}}))
