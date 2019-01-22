import tensorflow as tf
from config.py import *
import os, re, sys, traceback
import yaml, logging, logging.handlers
from filterpy.kalman import ExtendedKalmanFilter
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

n_msmt = 8 # Kalman z
n_param = 8 # Kalman x
n_entries = 250
# number of units in RNN cell
n_hidden = 512
learn_rate = 0.00001
default_n_epochs = 1000
state_ids = range(0, n_msmt)
config = None
state_file = "lstm_ekf.state"
initialized = False



def ekf_accuracy(ekf, msmts):

    ekf.predict()
    (state, n_state) = ([msmts[i] for i in state_ids], len(state_ids))
    state = array(state)
    state.resize(n_state, 1)
    logger.info("state = " + str(state) + ", n_state = " + str(n_state) + ", prior = " + str(ekf.x_prior))
    accuracy= lambda p: max(1 - pow(pow(p[0]-p[1], 2), 0.5)/max(p[1], 1e-9), 0)
    nums = lambda ns : map(lambda n: n[0], ns)
    mean = sum(map(accuracy, zip(nums(ekf.x_prior), nums(state))))/len(state)
    logger.info("x_prior = " + str(shape(ekf.x_prior)) + ", accuracy = " + str(mean))
    return mean



# Build and update an EKF using the provided measurments data
def build_ekf(coeffs, z_data): 
    ekf = ExtendedKalmanFilter(dim_x=n_param, dim_z= n_msmt)
    q = array(coeffs)
    q.resize(n_param, n_param) # TODO need to determine size
    ekf.Q = q
    r = array(coeffs)
    r = r.resize(n_msmt, n_msmt) # TODO need to determine size
    #ekf.R = r
    return update_ekf(ekf, z_data)


def update_ekf(ekf, z_data):
    hjacobian = lambda x: identity(len(x))
    hx = lambda x: x
    logging.info("ekf.x = " + str(ekf.x) + ", shape = " + str(shape(ekf.x)) + ", q.shape = " + str(shape(ekf.Q)) + ", q.type = " + str(type(ekf.Q)) + ", z_data = " + str(shape(z_data)))
    for z in z_data:
        z = array(z)
        z.resize(n_msmt, 1)
        logging.info("update.z = " + str(shape(z)) + ", x_prior = " + str(shape(ekf.x_prior)) + ", hjacobian = " + str(hjacobian([1, 2])))
        ekf.update(z, hjacobian, hx)
    return ekf




