from config import *
from utils import *
from plot import *
import config as cfg
import math
import yaml, logging, logging.handlers
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random


logger = logging.getLogger("Kalman_Filter")


# Test the accuracy of an EKF using the provide measurement data
def ekf_accuracy(ekf, msmt, indices=None, label="", predict=True, host=None):
    return ekf_accuracies(ekf, msmt, indices, label, predict, host)[-1]


# Test the accuracies of an EKF per measurement metric
def ekf_accuracies(ekf, msmt, indices=None, label="", predict=True, host=None):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    _ = ekf_predict(ekf) if predict else None
    ids = cfg.find(get_config("lqn-hosts"), host)
    state = list(map(lambda t: list(t.values())[0], tasks[ids[0]:ids[-1]+1]))
    (state, n_state) = (array(state), len(ids))
    logger.info("state = " + str(state) + ", msmt = " + str(len(msmt)))
    state.resize(n_state, 1)
    [accuracies, mean, state_ns, prior] = ekf_mean(ekfs[0], indices, state)
    max_mean = [mean, 0]
    means = [mean]
    for i in range(len(ekfs[1:])):
        _, mean, _, _ = ekf_mean(ekfs[i+1], indices, state)
        max_mean = [mean, i+1] if mean > max_mean[0] else max_mean
        means.append(mean)
    swap(ekfs, max_mean[1])
    mean = max_mean[0]
    # accuracy is average of 1 - 'point-wise scaled delta'
    logger.info(label + " x_prior = " + str(shape(ekfs[0].x_prior)) + 
        ", zip(prior,state,accuracies) = " + 
        str(list(zip(prior, state_ns, accuracies))) + 
        ", means = " + str(means) + ", accuracy = " + str(mean))
    return [[state_ns, accuracies], mean]



def ekf_predict(ekf):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    for ekf in ekfs:
        ekf.predict()


def ekf_mean(ekf, indices, state):
    nums = lambda ns : list(map(lambda n: n[0], ns))
    accuracy = lambda pt: -abs(pt[1]-pt[0]) #max(1 - abs((pt[1]-pt[0])/max(pt[0],1e-1)), 0)
    accuracies = list(map(accuracy, zip(nums(state), nums(ekf.x_prior), range(len(state)))))
    mean = avg([accuracies[i] for i in indices] if indices else accuracies) 
    return [accuracies, mean, nums(state), nums(ekf.x_prior)]


def swap(lst, i):
    tmp = lst[0]
    lst[0] = lst[i]
    lst[i] = tmp


def read2d(coeffs, width, start, end):
    vals = array(coeffs[start:end])
    vals.resize(width, width)
    return vals



# Build and update an EKF using the provided measurement data
def build_ekf(coeffs, z_data): 
    ekf = ExtendedKalmanFilter(dim_x = dimx, dim_z = n_msmt)
    if len(coeffs):
        coeffs = array(coeffs).flatten()
        if n_coeff == dimx * 2 + n_msmt:
            ekf.Q = symmetric(array(coeffs[0:dimx]))
            ekf.F = symmetric(array(coeffs[dimx:dimx*2]))
            r = symmetric(array(coeffs[-n_msmt:]))
        else:
            ekf.Q = read2d(coeffs, dimx, 0, dimx*dimx)
            ekf.F = read2d(coeffs, dimx, dimx*dimx, dimx*dimx*2)
            r = read2d(coeffs, n_msmt, -n_msmt*n_msmt, n_coeff)
        logger.info("ekf.Q="+str(ekf.Q) + ", ekf.F = " + str(ekf.F) + ", r = " + str(r))
        return update_ekf(ekf, z_data, r)
    return update_ekf(ekf, z_data)



def update_ekf(ekf, z_data, R = None):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    for i,z in zip(range(len(z_data)), z_data):
        z = array(z)
        z.resize(n_msmt, 1)
        def h(x):
            (m, c, project) = solve_linear(x, [z_data[i]])
            return m * project(x) + c
        def hjacobian(x):
            xs = array(list(map(lambda i: x + 1e-5*random(), range(len(x)))))
            zs = array(list(map(lambda xi: h(xi), xs)))
            logger.info("Hj = " + str(zs.T))
            return zs.T
        for ekf in ekfs:
            ekf.update(z, hjacobian, h, R)
    return ekf


def ekf_track(coeffs, z_data):
    points = []
    for i in range(len(z_data)):
        ekf = build_ekf(coeffs, z_data[0 : i])
        ekf.predict()
        points.append(ekf.x_prior)
    return list(zip(z_data, points))


def plot_ekf():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    logger.info("plot_ekf() done")


# Testbed to unit test EKF using hand-crafted data
def test_ekf():
    fns = [lambda x: 0 if x<50 else 1, math.exp, math.sin, math.erf]
    fns = [lambda x: 0 if x<50 else 1]
    coeffs = test_coeffs()
    logger.info("coeffs = " + str(len(coeffs)) + 
        ", size = " + str(size(coeffs)) + ", n_coeff = " + str(n_coeff))
    accuracies = [] 
    for n in range(len(fns)):
        (train, test) = test_zdata(fns, n)
        logger.info("train=" + str(len(train)) + ", train[2]=" + str(train[2]))
        ekf = build_ekf(coeffs, train)
        logger.info("test=" + str(len(test)) + ", test[0]=" + str(test[0]) + 
            ", fn = " + str(n) + " of " + str(len(fns)))
        means = list(map(lambda t: ekf_accuracy(ekf, t), test))
        accuracies.append(avg(means)) 
        logger.info("accuracy = " + str(accuracies[-1]) + ", fn = " + str(n) +
            " of " + str(len(fns)))
        predictions = ekf_track(coeffs, concatenate([train, test]))
        pickledump("predictions" + str(n) + ".pickle", predictions)
    logger.info("accuracies = " + str(accuracies))
    return


def test_coeffs():
    if n_coeff == dimx * 2 + n_msmt:
        return ones(dimx * 2 + n_msmt)
    else:
        return flatlist(identity(dimx)) + flatlist(identity(dimx)) + \
               flatlist(identity(n_msmt))

def test_zdata(fns, n):
    z_data = []
    for v in (array(range(300))/30 if n==1 else range(100)):
        msmt = map(lambda m: fns[n](v) if m<=n else random(), range(n_msmt))
        z_data.append(list(msmt))
    split = int(0.75 * len(z_data))
    return (z_data[0 : split], z_data[split : ])


if __name__ == "__main__":
    test_ekf()
