import tensorflow as tf
import math
from math import erf
from config import *
from utils import *
import os, re, sys, traceback
import yaml, logging, logging.handlers
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity, shape, eye
from numpy import ones, dot, divide, subtract, concatenate
from numpy.linalg import inv
from functools import reduce
from random import random, uniform
from csv import reader
from scipy.misc import derivative

logger = logging.getLogger("Lstm")


initialized = False
lqn_tbl = []


def RNN(x, weights, biases):

    if use_logistic_regression:
        cell = tf.nn.rnn_cell.LSTMCell(n_features, activation=tf.nn.relu)
        layers = [cell for layer in range(n_msmt)]
        multi_cell = tf.contrib.rnn.MultiRNNCell(layers)
        logger.info("RNN.cell,multi_cell,x = " + str((cell, multi_cell,x)))
        outputs,states = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        model = tf.matmul(outputs, weights) + biases
    else:
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden, activation=tf.nn.relu)
        logger.info("RNN.cell = " + str(cell))
        outputs,_ = tf.contrib.rnn.static_rnn(cell,inputs=[x],dtype=tf.float32)
        model = tf.matmul(outputs[-1], weights['out']) + biases['out']
    logger.info("x = " + str(x) + ", outputs = " + str(outputs) + ", weights="+
        str(weights) + ", biases = " + str(biases) + "= model = " + str(model))

    # there are n_entries outputs but
    # we only want the last output
    return model


def to_size(data, width, entries = n_entries):
    input = array(data)
    input.resize(width, entries)
    input = input.T
    logger.debug("data = " + str(len(data)) + ", input = " + str(shape(input)))
    return input


def repeat(x, n):
    return list(map(lambda i: x, range(n)))


# Generates test training labels
def test_labels(model, X = None, labels = [], sample_size = 5):
    noise = 1.0 
    #symbols = [lambda x: 0 if x<50 else 1, math.exp, math.sin] 
    symbols = [math.exp, math.sin] 
    for i in range(sample_size):
        for s in range(len(symbols)):
            for k in range(10):
                def measure(msmt):
                    val = symbols[s](k+1) 
                    return (1 - uniform(0.0, noise)) * val
                labels.append([0.0, 
                    repeat(list(map(measure, range(n_msmt))), n_entries), 
                    repeat(repeat(s, n_lstm_out), n_entries)])
            logger.debug("test_labels done with symbol " + str(symbols[s]))
    logger.debug("test_labels.labels = " + str(labels))
    return labels


def tf_run(*args, **kwargs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess.run(*args, **kwargs)


def weights_biases(base):
    base = array(base) + 1e-9
    logger.info("bias = " + str(base))
    weights = {'out':tf.Variable(tf.ones([n_hidden, n_lstm_out]),name='w')}
    if n_msmt == n_coeff / 3:
        bias = zeros([n_entries, n_lstm_out]) + \
                array(base).reshape(n_entries, n_lstm_out)
        biases = {'out': tf.convert_to_tensor(bias, name='b',dtype=tf.float32)}
    else:
        i = zeros(n_msmt * n_msmt) + base
        bias = array([concatenate((i,i,i)) for x in range(n_entries)])
        biases = {'out': tf.convert_to_tensor(bias, name='b', dtype=tf.float32)}
    return [weights, biases]


# Returns a trained LSTM model for R & Q Kalman Filter coefficient prediction
def tune_model(epochs = n_epochs, labelfn = test_labels, cost = None):
    #tf.reset_default_graph()
    X = tf.placeholder("float", [n_entries, n_msmt])
    Y = tf.placeholder("float", [n_entries, n_lstm_out])

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        if use_logistic_regression:
            [X, Y, model, cost, optimizer] = tune_logistic_model()
        else:
            weights, biases = weights_biases(best_label(None, X, labelfn))
            model = RNN(X, weights, biases) 
            cost = cost(model,X) if cost else tf.reduce_mean(tf.square(model-Y))
            optimizer = tf.train.MomentumOptimizer(learn_rate, 0.9)
        logger.debug("model = " + str(model))
        train_op = optimizer.minimize(cost)
    return train_and_test(model, X, Y, train_op, cost, epochs, labelfn)


def tune_logistic_model():
    X = tf.placeholder(tf.float32, [n_entries, n_msmt, n_features])
    Y = tf.placeholder(tf.float32, [1, n_lstm_out, n_classes])
    weights = tf.Variable(tf.truncated_normal([n_entries,n_features,n_classes]))
    biases = tf.Variable(tf.ones([n_entries, n_msmt, n_classes])*1e-2)
    model = RNN(X, weights, biases)
    labels = tf.nn.softmax(Y)
    labels = tf.reshape(Y, [n_lstm_out, n_classes])
    logits = tf.reshape(model, [n_lstm_out, n_classes])
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
     #   labels = labels, logits = model))
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    return [X, Y, model, cost, optimizer]


# Returns a LQN-based cost function
def tf_lqn_cost(model, X):
    load_lqn_table()
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        def err(output, x):
            print("err().output,x = " + str((output, x)))
            return output * tf.zeros([n_entries, n_msmt]) + \
                tf.slice(lqn_tbl, [lqn_row_id(output[-1]),0], [1,n_msmt]) -x[-1]
        cost = tf.reduce_mean(tf.square(err(model, X)))
    logger.debug("model = " + str(model) + ", cost = " + str(cost))
    return cost


def lqn_row_id(p):
    rng = get_config("lqn-range") + [1.0]
    row = 0;
    for i in range(len(rng)):
        row = row + tf.slice(p, [i], [1])[0] * rng[i]
    logger.debug("p, rowid = " + str((p, row)))
    return tf.cast(row, tf.int32)


def load_lqn_table():
    global lqn_tbl
    if len(lqn_tbl):
        return lqn_tbl
    with open(get_config("lqn-out")) as f:
        rdr = reader(f, delimiter=',')
        i = 0
        for row in rdr:
            if i > 0:
                lqn_tbl.append(tf.Variable([float32(s) for s in row]))
            i = i + 1
    return lqn_tbl


def best_label(model, X, labelfn):
    labels = labelfn(model, X, [])
    batch_data = [feature_classes(l[1:]) for l in labels]
    #batch_data.sort(key = lambda b: b[0], reverse=True)
    logger.info("best label = " + str(batch_data[0][1]))
    return array(batch_data[0][1]).T


def feature_classes(batch):
    if use_logistic_regression:
        [msmts, lbls] = batch
        logger.info("feature_classes.batch.lbls=" + str((lbls)))
        iden = eye(n_features)
        msmts = [[iden[int(erf(m)*(n_features-1))] for m in ms] for ms in msmts]
        lbls=[[iden[int(erf(c)*(n_classes-1))] for c in cs] for cs in lbls[-1:]]
        logger.info("feature_classes.msmts,lbls=" + str((msmts, lbls)))
        return [msmts, lbls]
    return batch



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
        labels = labelfn(model, X, labels)
        batch_data = [feature_classes(l[1:]) for l in labels]
        train_data = batch_data[0 : int(len(batch_data)*0.75)]
        test_data = test_data + batch_data[int(len(batch_data)*0.75) : ]
        train(train_op, cost, X, Y, train_data, costs, epoch)
        logger.debug("Epoch = " + str(epoch) + ", train_data = " +
            str((train_data)) + ", test_data = " + str((test_data)))
        
    pickleconc("train_costs.pickle", costs)
    return test(model, X, Y, test_data)



def train(train_op, cost, X, Y, train_data, costs, epoch):
    for (i, (batch_x, batch_y)) in zip(range(len(train_data)), train_data):
        # Remember 'cost' contains the model
        _, total_cost = tf_run([train_op, cost], feed_dict =
            {X: batch_x if use_logistic_regression else to_size(batch_x, n_msmt,
                n_entries), 
             Y: batch_y if use_logistic_regression else to_size(batch_y, 
                n_lstm_out, n_entries)})
        logger.debug("batchx = " + str(shape(batch_x)) +  ", batchy = " + 
           str(shape(batch_y)) + ", cost = " + str(total_cost) + 
           ", batch " + str(i+1) + " of " + str(len(train_data)) + 
           ", epoch " + str(epoch+1) + " of " + str(n_epochs)) 
        set_lstm_initialized()
        costs.append(total_cost)


def test(model, X, Y, test_data):
    [mse, cost, output] = [[], [], []]
    if use_logistic_regression:
        test_logistic(model, X, Y, test_data)
    else:
        test_x = to_size([t[0] for t in test_data], n_msmt, n_entries)
        test_y = to_size([t[1] for t in test_data], n_lstm_out, n_entries)
        labels = tf.reshape(Y, [n_lstm_out, n_entries])
        preds = tf.reshape(model, [n_lstm_out, n_entries])
        tf_mse = tf.losses.mean_squared_error(labels, preds)
        tf_cost = tf.reduce_mean(tf.square(model - Y))
        [mse, cost, output] = tf_run([tf_mse, tf_cost, model],
            feed_dict={X:test_x, Y:test_y})
        logger.debug("testx = " + str(test_x) + ", testy = " + str(test_y) + 
            ", X = " + str(X) + ", Y = " + str(Y) +
            ", (output,test_y) = " + str(list(zip(output, test_y))))
    logger.debug("LSTM MSE = " + str(mse) + ", cost = " + str(cost)) 
    pickleadd("test_costs.pickle", mse)
    return [model, X, mse]


def test_logistic(model, X, Y, test_data):
    accs = []
    soft = tf.reshape(tf.nn.softmax(model), [1, n_lstm_out, n_classes])
    preds = tf.argmax(tf.reshape(tf.one_hot(tf.nn.top_k(soft).indices, 
        n_classes), [1, n_lstm_out, n_classes]), 1)
    labels = tf.argmax(Y, 1)
    tf_accuracy = tf.metrics.accuracy(labels, predictions=preds)
    tf_recall = tf.metrics.recall(labels=labels, predictions=preds)
    tf_precision = tf.metrics.precision(labels=labels, predictions=preds)
    tf_tn = tf.metrics.true_negatives(labels=labels, predictions=preds)
    tf_fp = tf.metrics.false_positives(labels=labels, predictions=preds)
    for (test_x, test_y) in test_data:
        accs.append(tf_run(
            tf.stack([tf_accuracy, tf_recall, tf_precision, tf_tn, tf_fp]), 
            feed_dict={X:test_x, Y:test_y}))
        logger.info("pred,labels=" + str(tf_run(tf.stack([preds,labels]),
            feed_dict={X:test_x, Y:test_y})))
        logger.info("x = " + str(test_x) + ", y = " + str(test_y))
        logger.info("acc, recall,precision = "+str((accs[-1], test_y, test_x)))
        pickleadd("test_logit_accuracy.pickle", accs[-1].flatten())
    return accs[-1] 


if __name__ == "__main__":
    tune_model(3)
    logger.debug("done")
