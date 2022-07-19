from config import *
from lstm import Lstm
from utils import *
from plot import *
import config as cfg
import math
import yaml, logging, logging.handlers
import matplotlib.pyplot as plt
import tensorflow as tf
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter, CubatureKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract, eye, reshape
from numpy.linalg import inv
from functools import reduce
from random import random
from datetime import *


logger = logging.getLogger("Kalman_Filter")
kf_type = "EKF"


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
    if kf_type == "UKF":
        return ekf if isinstance(ekf, UnscentedKalmanFilter) else ekf['ekf']
    elif kf_type == "KF":
        return ekf if isinstance(ekf, KalmanFilter) else ekf['ekf']
    else:
        return ekf if isinstance(ekf, ExtendedKalmanFilter) else ekf['ekf']


def mean_accuracy(ekf, indices, state):
    (ekf,(m,c)) =(ekf['ekf'],ekf['mc']) if isinstance(ekf,dict) else (ekf,(1,0))
    nums = lambda ns : [n[0] for n in ns]
    prior = lambda kf: nums(m * get_ekf(kf).x_prior + c)
    acc = lambda pt: 1 - abs(pt[1] - pt[0]) 
                         #/abs(pt[0]+1e-9) #norms[pt[2] % len(norms)]
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
def build_ekf(coeffs, z_data, linear_consts=None, nmsmt = n_msmt, dx =dimx): 
    global n_msmt
    global dimx
    (dimx, n_msmt) = (dx, nmsmt)
    if kf_type == "UKF":
        ekf = build_unscented_ekf()
    elif kf_type == "KF":
        ekf = KalmanFilter(dim_x=4,dim_z=2)
        ekf.x = eye(4)
        #ekf = KalmanFilter(dim_x = dimx, dim_z = n_msmt)
        #(ekf.P, ekf.R, ekf.Q) = (eye(dimx), eye(dimx)*.1, eye(n_msmt)*.1)
    else:
        ekf = ExtendedKalmanFilter(dim_x = dimx, dim_z = n_msmt)
    if len(coeffs):
        r = update_ekf_coeffs(ekf, coeffs)
        return update_ekf(ekf, z_data, r, linear_consts)
    return update_ekf(ekf, z_data)


def build_unscented_ekf():
        #pts = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
        pts = JulierSigmaPoints(4)
        def fx(x, dt):
            # state transition function - predict next state based
            # on constant velocity model x = vt + x_0
            F = np.array([[1, dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]], dtype=float)
            return np.dot(F, x)
   
        def h(x):
            # measurement function - convert state into a measurement
            # where measurements are [x_pos, y_pos]
            return np.array([x[0], x[2]])
   
        ekf = UnscentedKalmanFilter(dim_x=4,dim_z=2,dt=.1,hx=h,fx=fx,points=pts)
        return ekf


def update_ekf_coeffs(ekf, coeffs):
        coeffs = array(coeffs).flatten()
        if n_coeff == dimx * 2 + n_msmt:
            ekf.Q = symmetric(array(coeffs[0:dimx]))
            ekf.F = symmetric(array(coeffs[dimx:dimx*2]))
            r = symmetric(array(coeffs[-n_msmt:]))
        else:
            ekf.Q = read2d(coeffs, dimx, 0, dimx*dimx)
            ekf.F = read2d(coeffs, dimx, dimx*dimx, dimx*dimx*2)
            r = read2d(coeffs, n_msmt, -n_msmt*n_msmt, n_coeff)
        logger.info("ekf.Q={}, F = {}, r = {}".format(ekf.Q, ekf.F, r))
        return r


def update_ekf(ekf, z_data, R=None, m_c = None, Hj=None, H=None):
    logger.info("z_data = " + str((array(z_data).shape)))
    (ekfs, start) = (ekf if isinstance(ekf, list) else [ekf], datetime.now())
    priors = [[] for i in ekfs]
    for i,z in zip(range(len(z_data)), z_data):
        z = reshape(z, (array(z).size, 1))
        logger.info("z = " + str((len(z), z.size, array(z).shape)))
        h = lambda x: H(x) if H else m_c[0]*x if m_c else x
        def hjacobian(x):
            m = m_c[0] if m_c else 1
            return Hj(x) if Hj else m * identity(len(x)) 
        for j,ekf in zip(range(len(ekfs)), ekfs):
            ekf = get_ekf(ekf)
            ekf.predict()
            priors[j].append(ekf.x_prior)
            if kf_type == "UKF":
                z = [z[0][0], z[1][0]]
                ekf.update(z)
            elif kf_type == "KF":
                z = [z[0][0], z[1][0]]
                ekf.update(z)
            else:
                ekf.update(z, hjacobian, h, R if len(shape(R)) else ekf.R)
    logger.info("priors,z_data,ekfs = " + str((array(priors).shape, 
                                               array(z_data).shape,len(ekfs))))
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
    kf_type = sys.argv[sys.argv.index("-t")+1] if "-t" in sys.argv else kf_type
    if "--testpcacsv" in sys.argv:
        (pca, f) = ("true", sys.path[0]+"/../data/mackey_glass_time_series.csv")
        f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
        pca = sys.argv[sys.argv.index("-pc")+1] if "-pc" in sys.argv else pca
        xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'P'
        ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'P'
        ekf = build_ekf([], [])
        def predfn(msmts, lqn_ps = None): 
            priors = update_ekf(ekf, msmts)[1]
            return to_size(priors[-1], msmts.shape[1], msmts.shape[0])
        test_pca_csv(f, xcol, ycol, None, predfn, dopca=pca.lower() == "true")
    elif "--testlstm" in sys.argv:
        test_lstm_ekf()
    print("Output in lstm_ekf.log")
