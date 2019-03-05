from config import *
from utils import *
from plot import *
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
def ekf_accuracy(ekf, msmt, indices=None, label=""):
    return ekf_accuracies(ekf, msmt, indices, label)[-1]


# Test the accuracies of an EKF per measurement metric
def ekf_accuracies(ekf, msmt, indices=None, label=""):
    ekf.predict()
    logger.info("state_ids = " + str(state_ids) + ", msmt = " + str(len(msmt)))
    (state, n_state) = ([msmt[i] for i in state_ids], len(state_ids))
    state = array(state)
    state.resize(n_state, 1)
    # accuracy is average of 1 - 'point-wise scaled delta'
    accuracy = lambda pt: max(1 - abs((pt[1]-pt[0])/max(pt[0],1e-9)), 0)
    nums = lambda ns : map(lambda n: n[0], ns)
    accuracies = list(map(accuracy, zip(nums(state), nums(ekf.x_prior))))
    mean = avg([accuracies[i] for i in indices] if indices else accuracies) 
    logger.info(label + " x_prior = " + str(shape(ekf.x_prior)) + 
        ", zip(prior,state,accuracies) = " + 
        str(list(zip(nums(ekf.x_prior), nums(state), accuracies))) + 
        ", accuracy = " + str(mean))
    return [[accuracies, list(nums(state))], mean]



# Build and update an EKF using the provided measurement data
def build_ekf(coeffs, z_data): 
    dimx = n_msmt #int(math.sqrt(n_coeff - n_msmt*n_msmt))
    ekf = ExtendedKalmanFilter(dim_x = dimx, dim_z = n_msmt)
    if coeffs:
        #q = symmetric(array(coeffs[0:dimx])) #array(coeffs[0:dimx*dimx])
        q = array(coeffs[0:dimx*dimx])
        q.resize(dimx, dimx) # TODO need to determine size
        ekf.Q = q
        #dg = array(coeffs[n_msmt*n_msmt:n_msmt*(n_msmt+1)])
        #dg = array(coeffs[n_msmt:n_msmt*2])
        #half = array(coeffs[n_msmt*(n_msmt+1):pow(n_msmt*(n_msmt+1), 2)])
        f = array(coeffs[n_msmt*n_msmt:n_msmt*n_msmt*2])
        f.resize(n_msmt, n_msmt) # TODO need to determine size
        ekf.F = f #symmetric(dg)
        #logger.info("ekf.F.dg=" + str(dg)+ ", coeffs.size=" + str(len(coeffs)))
        #r = symmetric(array(coeffs[-n_msmt:]))
        r = array(coeffs[-n_msmt*n_msmt:])
        r = r.resize(n_msmt, n_msmt) # TODO need to determine size
        return update_ekf(ekf, z_data, r)
    return update_ekf(ekf, z_data)


def update_ekf(ekf, z_data, R = None):
    hjacobian = lambda x: identity(len(x))
    hx = lambda x: x
    for z in z_data:
        z = array(z)
        z.resize(n_msmt, 1)
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
    coeffs = flatlist(identity(int(sqrt(n_coeff-n_msmt*n_msmt)))) + flatlist(identity(n_msmt))
    logger.info("coeffs = " + str(len(coeffs)) + 
        ", size = " + str(size(coeffs)) + ", n_coeff = " + str(n_coeff))
    accuracies = [] 
    for n in range(len(fns)):
        z_data = []
        for v in (array(range(300))/30 if n==2 else range(100)):
            msmt = map(lambda m: fns[m](v) if m<=n else random(), range(n_msmt))
            z_data.append(list(msmt))
        split = int(0.75 * len(z_data))
        (train, test) = (z_data[0 : split], z_data[split : ])
        logger.info("train=" + str(len(train)) + ", train[2]=" + str(train[2]))
        ekf = build_ekf(coeffs, train)
        logger.info("test=" + str(len(test)) + ", test[0]=" + str(test[0]) + 
            ", fn = " + str(n) + " of " + str(len(fns)))
        means = list(map(lambda t: ekf_accuracy(ekf, t), test))
        accuracies.append(avg(means)) 
        logger.info("accuracy = " + str(accuracies[-1]) + ", fn = " + str(n) +
            " of " + str(len(fns)))
        predictions =ekf_track(coeffs,list(map(lambda d:repeat(d[2],4),z_data)))
        pickledump("predictions" + str(n) + ".pickle", predictions)
    logger.info("accuracies = " + str(accuracies))
    return



if __name__ == "__main__":
    test_ekf()
