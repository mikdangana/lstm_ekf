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

logger = logging.getLogger("Lstm")


initialized = False


def RNN(x, weights, biases):

    # 1-layer LSTM with n_hidden units.
    cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    #rnn_cell.zero_state(1, dtype=tf.float32)
    logger.info("RNN.cell = " + str(cell))

    # generate prediction
    outputs,states = tf.contrib.rnn.static_rnn(cell,inputs=[x],dtype=tf.float32)
    logger.info("inputs = " + str(x) + ", outputs = " + str(outputs) + ", outputs.trunced = " + str(outputs[-1]) + ", states = " + str(states) + ", weights = " + str(weights['out']) + ", biases = " + str(biases['out']))

    # there are n_entries outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def to_size(data, width, entries = n_entries):
    input = array(data)
    input.resize(entries, width)
    logger.debug("data = " + str(len(data)) + ", input = " + str(shape(input)))
    return input

def repeat(x, n):
    return list(map(lambda i: x, range(n)))


# Generates test training labels
def test_labels(model, X = None, labels = [], sample_size = 5):
    noise = 1.0 
    symbols = [lambda x: 0 if x<50 else 1, math.factorial, math.exp, math.log, math.sqrt, math.cos, math.sin, math.tan, math.erf]
    for i in range(sample_size):
        for s in range(len(symbols)):
            for k in range(10):
                def measure(msmt):
                    #res = (1 - uniform(0.0, noise)) * symbols[s](k+1)
                    val = symbols[s](k+1) #if msmt==0 else derivative(symbols[s], k+1, msmt)
                    return (1 - uniform(0.0, noise)) * val
                labels.append([0.0, repeat(list(map(measure, range(n_msmt))), n_entries), repeat(s, n_entries)])
            logger.debug("test_labels done with symbol " + str(symbols[s]))
    logger.debug("test_labels.labels = " + str(labels))
    return labels


def tf_run(*args, **kwargs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess.run(*args, **kwargs)


# Returns a trained LSTM model for R & Q Kalman Filter coefficient prediction
def tune_model(n_epochs = default_n_epochs, labelfn = test_labels):

    X = tf.placeholder("float", [n_entries, n_msmt])
    Y = tf.placeholder("float", [n_coeff, 1])

    # RNN output node weights and biases
    weights = {'out':tf.Variable(tf.ones([n_hidden,n_coeff]),name='w')}
    biases = {'out': tf.Variable(tf.zeros([1, n_coeff]), name='b')}

    model = RNN(X, weights, biases)
    logger.debug("model = " + str(model))

    cost = tf.reduce_mean(tf.square(model - Y))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(cost)
    return train_and_test(model, X, Y, train_op, cost, n_epochs, labelfn)


def lstm_initialized():
    global initialized
    return initialized


def set_lstm_initialized():
    global initialized
    initialized = True


def train_and_test(model, X, Y, train_op, cost, n_epochs, labelfn=test_labels):
    (test_data, costs, labels) = ([], [], [])

    # Training
    for epoch in range(0, n_epochs):
        labels = labelfn(model, X, list(labels))
        batch_data = list(map(lambda l: l[1:], labels))
        train_data = batch_data[0 : int(len(batch_data)*0.75)]
        test_data = test_data + batch_data[int(len(batch_data)*0.75) : ]
        for (i, (batch_x, batch_y)) in zip(range(len(train_data)), train_data):
            # Remember 'cost' contains the model
            _, total_cost = tf_run([train_op, cost], feed_dict =
                {X: to_size(batch_x, n_msmt, n_entries), 
                 Y: to_size(batch_y, 1, n_coeff)})
            logger.debug("batchx = " + str(shape(batch_x)) +  ", batchy = " + 
               str(shape(batch_y)) + ", cost = " + str(total_cost) + 
               ", batch " + str(i+1) + " of " + str(len(train_data)) + 
               ", epoch " + str(epoch+1) + " of " + str(n_epochs)) 
            set_lstm_initialized()
            costs.append(total_cost)
        logger.debug("Epoch = " + str(epoch) + ", train_data = " +str(shape(train_data)) + ", test_data = " + str(shape(test_data))+", cost=" +str(total_cost))

    pickleconc("train_costs.pickle", costs)
    return test(model, X, Y, test_data)


def test(model, X, Y, test_data):
    #pred = tf.nn.softmax(model)
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    labels = tf.reshape(Y, [1, n_coeff])
    preds = tf.reshape(model, [1, n_coeff])
    tf_mse = tf.losses.mean_squared_error(labels, preds)
    tf_cost = tf.reduce_mean(tf.square(model - Y))
    test_x = to_size(list(map(lambda t: t[0], test_data)), n_msmt, n_entries)
    test_y = to_size(list(map(lambda t: t[1], test_data)), 1, n_coeff)
    logger.debug("testx = " + str(test_x) + ", testy = " + str(test_y) + ", X = " + str(X) + ", Y = " + str(Y))
    #tf_accuracy = tf.metrics.accuracy(labels, predictions=preds)
    #tf_recall = tf.metrics.recall(labels=labels, predictions=preds)
    #tf_precision = tf.metrics.precision(labels=labels, predictions=preds)
    #tf_tn = tf.metrics.true_negatives(labels=labels, predictions=preds)
    #tf_fp = tf.metrics.false_positives(labels=labels, predictions=preds)
    [mse, cost, output] = tf_run([tf_mse, tf_cost, model], feed_dict={X:test_x, Y:test_y})
    #[accuracy, recall, precision, tn, fp] = tf_run(
    #       tf.stack([tf_accuracy, tf_recall, tf_precision, tf_tn, tf_fp]), 
    #       feed_dict={X:test_x, Y:test_y}) 
    logger.debug("LSTM MSE = " + str(mse) + ", cost = " + str(cost) + ", output = " + str(output))
    pickleadd("test_costs.pickle", mse)
    return [model, X, mse]



if __name__ == "__main__":
    tune_model(1)
    logger.debug("done")
