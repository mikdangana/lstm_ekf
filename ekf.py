from config import *
from utils import *
from plot import *
import math
import yaml, logging, logging.handlers
import matplotlib.pyplot as plt
import pickle
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random


logger = logging.getLogger("Kalman_Filter")


# Test the accuracy of an EKF using the provide measurement data
def ekf_accuracy(ekf, msmt):
    ekf.predict()
    logger.info("state_ids = " + str(state_ids) + ", msmt = " + str(len(msmt)))
    (state, n_state) = ([msmt[i] for i in state_ids], len(state_ids))
    state = array(state)
    state.resize(n_state, 1)
    logger.info("state = " + str(state) + ", n_state = " + str(n_state) + ", prior = " + str(ekf.x_prior))
    # accuracy is average of 1 - 'point-wise scaled delta'
    acc = lambda p: max(1 - abs(p[0]-p[1])/max(p[1], 1e-9), 0)
    nums = lambda ns : map(lambda n: n[0], ns)
    mean = avg(list(map(acc, zip(nums(ekf.x_prior), nums(state)))))
    logger.info("x_prior = " + str(shape(ekf.x_prior)) + ", accuracy = " + str(mean))
    return mean



# Build and update an EKF using the provided measurement data
def build_ekf(coeffs, z_data): 
    dimx = int(math.sqrt(n_coeff - n_msmt*n_msmt))
    ekf = ExtendedKalmanFilter(dim_x = dimx, dim_z = n_msmt)
    q = array(coeffs[0:dimx*dimx])
    q.resize(dimx, dimx) # TODO need to determine size
    ekf.Q = q
    r = array(coeffs[-n_msmt*n_msmt:])
    r = r.resize(n_msmt, n_msmt) # TODO need to determine size
    #ekf.R = r # R assignment done using ekf.update() below
    #logger.debug("coeffs = " + str(size(coeffs)) + ", n_coeff = " + str(n_coeff) + ", ekf.Q = " + str(ekf.Q) + ", ekf.R = " + str(ekf.R) + ", r = " + str(r))
    return update_ekf(ekf, z_data, r)


def update_ekf(ekf, z_data, R = None):
    hjacobian = lambda x: identity(len(x))
    hx = lambda x: x
    #logging.debug("EKF.x.shape = " + str(shape(ekf.x)) + ", q.shape = " + str(shape(ekf.Q)) + ", q.type = " + str(type(ekf.Q)) + ", z_data = " + str(shape(z_data)))
    for z in z_data:
        z = array(z)
        z.resize(n_msmt, 1)
        #logger.info("update.z = " + str(shape(z)) + ", x_prior = " + str(shape(ekf.x_prior)) + ", hjacobian = " + str(hjacobian([1, 2])))
        ekf.update(z, hjacobian, hx, R)
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
    coeffs = flatlist(identity(int(sqrt(n_coeff-n_msmt*n_msmt)))) + flatlist(identity(n_msmt));
    logger.info("coeffs = " + str(len(coeffs)) + ", size = " + str(size(coeffs)) + ", n_coeff = " + str(n_coeff))
    accuracies = []
    for n in range(len(fns)):
        z_data = []
        for v in range(300):
            msmt = map(lambda m: fns[m](v) if m<=n else random(), range(n_msmt))
            z_data.append(list(msmt))
        split = int(0.75 * len(z_data))
        (train, test) = (z_data[0 : split], z_data[split : ])
        logger.info("train = " + str(len(train)) + ", train[0] = " + str(train[0]))
        ekf = build_ekf(coeffs, train)
        logger.info("test = " + str(len(test)) + ", test[0] = " + str(test[0]) + ", fn = " + str(n) + " of " + str(len(fns)))
        means = list(map(lambda t: ekf_accuracy(ekf, t), test))
        accuracies.append(avg(means)) 
        logger.info("accuracy = " + str(accuracies[-1]) + ", fn = " + str(n) + " of " + str(len(fns)))
        predictions = ekf_track(coeffs, z_data)
        with open("predictions" + str(n) + ".pickle", 'wb') as f:
            pickle.dump(predictions, f)
        # plotmetric(predictions, n)
    logger.info("accuracies = " + str(accuracies))
    return



if __name__ == "__main__":
    test_ekf()
