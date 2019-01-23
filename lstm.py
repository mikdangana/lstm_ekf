import tensorflow as tf
import math
from config import *
from utils import *
import os, re, sys, traceback
import yaml, logging, logging.handlers
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random, uniform
from scipy.misc import derivative

logger = logging.getLogger("Lstm_Kalman_Tuner")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='lstm_ekf.log', 
    format='%(levelname)s %(asctime)s in %(funcName)s() ' +
        '%(filename)s-%(lineno)s: %(message)s \n', level=logging.INFO)

n_msmt = 8 # Kalman z
n_param = 8 # Kalman x
n_entries = 10
# number of units in RNN cell
n_hidden = 1
learn_rate = 0.00001
default_n_epochs = 1000
state_ids = range(0, n_msmt)
config = None
state_file = "lstm_ekf.state"
initialized = False


def measurements():
    cmd = "top -b -n 2 | grep -v Tasks | grep -v top | grep -v %Cpu | " + \
        "grep -v KiB | grep -v PID | grep [0-9] | " + \
        "awk '{print $1/100000,$3/140,$4/20,$5/1000000,$6/1000000,$7/1000000,$9,$10}'"
    pstats = os_run(cmd)
    logger.info(len(pstats.split()))
    pstatsf = list(map(lambda i: float32(i) if i!="rt" else 0, pstats.split()))
    logger.info("type = " + str(type(pstatsf[0])))
    return pstatsf


def RNN(x, weights, biases):

    # 1-layer LSTM with n_hidden units.
    cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    #rnn_cell.zero_state(1, dtype=tf.float32)
    logger.info("RNN.cell = " + str(cell))

    # generate prediction
    outputs,states = tf.contrib.rnn.static_rnn(cell,inputs=[x],dtype=tf.float32)
    logger.info("inputs = " + str(x) + "outputs = " + str(outputs) + ", outputs.trunced = " + str(outputs[-1]) + ", states = " + str(states) + ", weights = " + str(weights['out']) + ", biases = " + str(biases['out']))

    # there are n_entries outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def predict_coeffs(model, newdata):
    global initialized
    logger.info("initialized = " + str(initialized))
    if not initialized:
        return list(map(lambda n: random(), range(0,2300)))
    input = array(newdata)
    input.resize(n_entries, n_msmt)
    output = tf_run(model, feed_dict={X:input})
    logger.debug("Coeff predict = " + str(output[-1]) + ", input = " + str(shape(newdata)))
    return output[-1]


def repeat(x, n):
    return list(map(lambda i: x, range(n)))


# Generates test training labels
def test_labels(model, sample_size = 5):
    (labels, noise) = ([], 1.0) 
    symbols = [lambda x: 0 if x<50 else 1, math.factorial, math.exp, math.log, math.sqrt, math.cos, math.sin, math.tan, math.erf]
    for i in range(sample_size):
        for s in range(len(symbols)):
            for k in range(100):
                def measure(msmt):
                    #res = (1 - uniform(0.0, noise)) * symbols[s](k+1)
                    val = symbols[s](k+1) #if msmt==0 else derivative(symbols[s], k+1, msmt)
                    return (1 - uniform(0.0, noise)) * val
                labels.append([repeat(list(map(measure, range(n_msmt))), n_entries), repeat(s, n_entries)])
            logger.debug("test_labels.s = " + str(s))
    logger.debug("test_labels.labels = " + str(labels))
    return labels


def tf_run(*args, **kwargs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess.run(*args, **kwargs)


# Returns a trained LSTM model for R & Q Kalman Filter coefficient prediction
def tune_model(n_epochs = default_n_epochs):

    X = tf.placeholder("float", [n_entries, n_msmt])
    Y = tf.placeholder("float", [n_entries, 1])

    # RNN output node weights and biases
    weights = {'out':tf.Variable(tf.ones([1,n_entries]),name='w')}
    biases = {'out': tf.Variable(tf.zeros([1,n_entries]), name='b')}

    model = RNN(X, weights, biases)
    logger.debug("model = " + str(model))

    cost = tf.reduce_mean(tf.square(model - Y))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(cost)
    return train_and_test(model, X, Y, train_op, cost, n_epochs)


def train_and_test(model, X, Y, train_op, cost, n_epochs):
    test_data = []

    # Training
    for epoch in range(0, n_epochs):
        batch_data = test_labels(model)
        train_data = batch_data[0 : int(len(batch_data)*0.75)]
        test_data = test_data + batch_data[int(len(batch_data)*0.75) : ]
        for (i, (batch_x, batch_y)) in zip(range(len(train_data)), train_data):
            # Remember 'cost' contains the model
            _, total_cost = tf_run([train_op, cost], feed_dict =
                {X: to_size(batch_x,n_msmt,n_entries), 
                 Y: to_size(batch_y,1, n_entries)})
            logger.debug("batchx = " + str(shape(batch_x)) +  ", batchy = " + 
               str(shape(batch_y)) + ", cost = " + str(total_cost)) + 
               ", batch " + str(i) + " of " + str(len(train_data))
            initialized = True
            mean_cost = total_cost / len(train_data)
            logger.debug("Mean_cost = " +str(mean_cost))
        logger.debug("Epoch = " + str(epoch) + ", train_data = " +str(shape(train_data)) + ", test_data = " + str(shape(test_data)))

    return test(model, X, Y, test_data)


def test(model, X, Y, test_data):

    pred = tf.nn.softmax(model)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    tf_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_x = to_size(list(map(lambda t: t[0], test_data)), n_msmt, n_entries)
    test_y = to_size(list(map(lambda t: t[1], test_data)), 1, n_entries)
    accuracy = tf_run(tf_accuracy, feed_dict={X:test_x, Y:test_y})
    logger.debug("LSTM Accuracy = " + str(accuracy))

    return [model, accuracy]


tune_model(2)
logger.debug("done")
