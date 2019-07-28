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
from datetime import *


logger = logging.getLogger("Kalman_Filter")


# Test the accuracy of an EKF using the provide measurement data
def ekf_accuracy(ekf, msmt, indices=None, label="", predict=True, host=None):
    return ekf_accuracies(ekf, msmt, indices, label, predict, host)[-1]


# Test the accuracies of an EKF per measurement metric
def ekf_accuracies(ekf, msmt, indices=None, label="", predict=True, host=None):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    _ = ekf_predict(ekf) if predict else None
    ids = cfg.find(get_config("lqn-hosts"), host)
    state = [list(t.values())[0] for t in tasks[ids[0]:ids[-1]+1]]
    (state, n_state) = (array(state), len(ids))
    logger.info("state = " + str(state) + ", msmt = " + str(len(msmt)))
    [accuracies, mean, state_ns, prior] = mean_accuracy(ekfs[0], indices, msmt)
    (max_mean, means) = ([mean, 0], [mean])
    for i in range(len(ekfs[1:])):
        _, mean, _, _ = mean_accuracy(ekfs[i+1], indices, msmt)
        max_mean = [mean, i+1] if mean > max_mean[0] else max_mean
        means.append(mean)
    swap(ekfs, max_mean[1])
    mean = max_mean[0]
    logger.info(label + " x_prior = " + str(shape(get_ekf(ekfs).x_prior)) + 
        ", zip(prior,state,accuracies) = " + 
        str(list(zip(prior, state_ns, accuracies))) + 
        ", means = " + str(means) + ", accuracy = " + str(mean))
    return [[state_ns, accuracies, prior], mean]


def ekf_predict(ekf):
    ekfs = ekf if isinstance(ekf, list) else [ekf]
    for ekf in ekfs:
        get_ekf(ekf).predict()
    return get_ekf(ekf).x_prior


def get_ekf(ekf):
    while isinstance(ekf, list) or isinstance(ekf, tuple):
        ekf = ekf[0]
    return ekf if isinstance(ekf, ExtendedKalmanFilter) else ekf['ekf']


def mean_accuracy(ekf, indices, state):
    (ekf,(m,c)) =(ekf['ekf'],ekf['mc']) if isinstance(ekf,dict) else (ekf,(1,0))
    nums = lambda ns : [n[0] for n in ns]
    prior = lambda kf: nums(m * get_ekf(kf).x_prior + c)
    acc = lambda pt: 1 - abs(pt[1] - pt[0]) #/abs(pt[0]+1e-9) #norms[pt[2] % len(norms)]
    accuracies = [acc(p) for p in zip(state, prior(ekf), range(len(state)))]
    logger.info("accuracies = " + str(accuracies) + \
        ", state = " + str(state) + ", prior = " + str(prior(ekf)))
    mean = avg([accuracies[i] for i in indices] if indices else accuracies) 
    return [accuracies, mean, state, prior(ekf)]


def swap(lst, i):
    tmp = lst[0]
    lst[0] = lst[i]
    lst[i] = tmp


def read2d(coeffs, width, start, end):
    vals = array(coeffs[start:end])
    vals.resize(width, width)
    return vals



# Build and update an EKF using the provided measurement data
def build_ekf(coeffs, z_data, linear_consts=None): 
    ekf = ExtendedKalmanFilter(dim_x = dimx, dim_z = n_msmt)
    #ekf.__init__(dimx, n_msmt)
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
        return update_ekf(ekf, z_data, r, linear_consts)
    return update_ekf(ekf, z_data)


def update_ekf(ekf, z_data, R=None, m_c = None, Hj=None, H=None):
    (ekfs, start) = (ekf if isinstance(ekf, list) else [ekf], datetime.now())
    priors = [[] for i in ekfs]
    for i,z in zip(range(len(z_data)), z_data):
        z = array(z)
        z.resize(n_msmt, 1)
        h = lambda x: H(x) if H else m_c[0]*x if m_c else x
        def hjacobian(x):
            m = m_c[0] if m_c else 1
            return Hj(x) if Hj else m * identity(len(x)) 
        for j,ekf in zip(range(len(ekfs)), ekfs):
            ekf = get_ekf(ekf)
            ekf.predict()
            priors[j].append(ekf.x_prior)
            ekf.update(z, hjacobian, h, R if len(shape(R)) else ekf.R)
    logger.info("priors,z_data,ekfs = " + str((priors, z_data,len(ekfs))))
    return (ekf, priors)


def ekf_track(coeffs, z_data):
    ekf, points = build_ekf(coeffs, z_data)
    return list(zip(z_data, points[0]))


def plot_ekf():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    logger.info("plot_ekf() done")


def test_coeffs(generators):
    if n_coeff == dimx * 2 + n_msmt:
        return concatenate([ones(dimx), ones(dimx), ones(n_msmt)])
    else:
        return flatlist(ones((dimx,dimx))) + flatlist(identity(dimx)) + \
               flatlist(ones((n_msmt,n_msmt)))


def get_generators():
    generators = [lambda x: 0 if x<50 else 1, #math.pow(1.01, x), 
                  math.exp, 
                  math.sin, #lambda x: random(),
                  math.erf, #lambda x: math.sin((x-10)/10) + random()*0.25, 
                  math.erf,
                  lambda x: 1 if int(x/50) % 2==1 else 0]
    return generators


# Testbed to unit test EKF using hand-crafted data
def test_ekf(generate_coeffs = test_coeffs):
    generators = get_generators()
    m_cs = [(10.0, 0.0) for i in range(len(generators))]
    (coeffs, accuracies, predictions) = (generate_coeffs(generators), [], [])
    for n in range(len(generators)):
        (train, test) = test_zdata(generators, n)
        logger.info("train[0:1] = " + str(train[0:1]))
        ekf, _ = build_ekf(coeffs, train, m_cs[n])
        ekf = {"ekf": ekf, "mc": m_cs[n]}
        accuracies.append(avg([ekf_accuracy(ekf, t) for t in test])) 
        logger.info("train=" + str(len(train)) + ", test = " + str(len(test)) + 
            ", accuracy = " + str(accuracies[-1]) + ", fn = " + str(n) + 
            " of " + str(len(generators)))
        predictions.append(ekf_track(coeffs, concatenate([train, test])))
        pickledump("predictions" + str(n) + ".pickle", predictions[-1])
    logger.info("accuracies = " + str(accuracies))
    return predictions


def test_zdata(generators, n):
    z_data = []
    g = generators[n]
    for v in (array(range(300))/30 if g in [math.exp,math.sin] else range(100)):
        msmt = [g(v) for m in range(n_msmt)]
        z_data.append(msmt)
    split = int(0.75 * len(z_data))
    return (z_data[0 : split], z_data[split : ])


if __name__ == "__main__":
    test_ekf()
    print("Output in lstm_ekf.log")
