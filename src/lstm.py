import tensorflow as tf
import math
from time import time_ns
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
tf.compat.v1.disable_v2_behavior()



def repeat(x, n):
    return array(list(map(lambda i: x, range(n))))


# Generates test training labels
def test_labels(model, X = None, labels = [], sample_size = 10):
    noise = 0.0 
    #symbols = [lambda x: 0 if x<50 else 1, math.exp, math.sin] 
    symbols = [math.sin, math.exp] 
    for i in range(sample_size):
        for s in range(len(symbols)):
            def measure(msmt):
                val = symbols[s]((msmt+1)/(sample_size*10)) 
                return (1 - uniform(0.0, noise)) * val
            if use_logistic_regression:
              labels.append([0.0, 
                repeat(array(list(map(measure, range(n_msmt)))), n_entries), 
                repeat(repeat(s, n_entries), n_coeff)])
            else:
              labels.append([0.0, 
                repeat(array(list(map(measure, range(n_msmt)))), n_entries), 
                repeat(repeat(s, n_entries), n_coeff)])
        logger.debug("test_labels done with symbol " + str(symbols[s]))
    logger.debug("test_labels.labels = " + str(labels))
    return labels


def weights_biases(base, n_lstm_out):
    base = array(base) + 1e-9
    weights = {'out':tf.convert_to_tensor(ones([n_hidden, n_lstm_out]),
                                          name='w', dtype=tf.float32)}
    if n_msmt == n_coeff / 3:
        bias = zeros([n_entries, n_lstm_out]) + \
                array(base).reshape(n_entries, n_lstm_out)
        biases = {'out': tf.convert_to_tensor(bias, name='b',dtype=tf.float32)}
    else:
        i = zeros(n_msmt * n_msmt) + base
        bias = array([concatenate((i,i,i)) for x in range(n_entries)])
        biases = {'out': tf.convert_to_tensor(bias, name='b', dtype=tf.float32)}
    logger.info("ws,bs = " + str(([v['out'].shape for v in \
                                   [weights, biases]])))
    weights['out'] = tf.Variable(weights['out'])
    biases['out'] = tf.Variable(biases['out'])
    return [weights, biases]


def err(output, x):
    lqn_tbl = load_lqn_table()
    rowid = lqn_row(output[-1])
    return tf.cond(tf.logical_or(tf.less(rowid,0), 
                   tf.greater_equal(rowid,len(lqn_tbl))),
        lambda : 1e12, #tf.float32.max,
        lambda : output * tf.zeros([n_entries, n_msmt]) + \
            tf.slice(lqn_tbl, [rowid,0], [1,n_msmt]) - x[-1])


# Returns a LQN-based cost function
def tf_lqn_cost(model, X):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        cost = tf.reduce_mean(tf.square(err(model, X)))
        with tf.name_scope("LSTM1"):
            tf.summary.scalar('cost', cost)
            #tf.summary.scalar('stddev', tf.sqrt(cost))
            #tf.summary.scalar('max', tf.reduce_max(tf.square(err(model, X))))
            #tf.summary.scalar('min', tf.reduce_min(tf.square(err(model, X))))
            tf.summary.histogram('histogram', err(model, X))
    logger.debug("model = " + str(model) + ", cost = " + str(cost))
    return cost


def lqn_row(p):
    rng = [1.0] + get_config("lqn-range") 
    row = 0;
    for i in range(len(rng)):
        row = row + tf.slice(p, [i], [1])[0] * rng[i]
    logger.debug("p, rowid, rng = " + str((p, row, rng)))
    return tf.cast(row, tf.int32)


def load_lqn_table():
    lqn_tbl = []
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
    logger.info("best_label.batch_data,labels = " + 
                str((array(batch_data).shape,batch_data, labels)))
    logger.info("best_label = " + str(batch_data[0][1]))
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



class Lstm:

    def __init__(self):
        self.initialized = False
        self.train_writer = None
        self.sess = None
        self.graph1 = tf.Graph()


    def RNN(self, x, weights, biases):
        with self.graph1.as_default():
            if use_logistic_regression:
                cell = tf.nn.rnn_cell.LSTMCell(n_features,activation=tf.nn.relu)
                layers = [cell for layer in range(n_msmt)]
                multi_cell = tf.contrib.rnn.MultiRNNCell(layers)
                logger.info("RNN.cell,multi_cell,x = "+str((cell,multi_cell,x)))
                outputs,states = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
                model = tf.matmul(outputs, weights) + biases
            else:
                cell = tf.nn.rnn_cell.LSTMCell(n_hidden, activation=tf.nn.relu)
                logger.info("RNN.cell = " + str(cell))
                outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs=[x],
                            dtype=tf.float32)
                model = tf.matmul(outputs[-1], weights['out']) + biases['out']
            logger.info("x = " +str(x)+ ", outputs = " +str(outputs)+ 
                ", weights="+str(weights) +", biases = "+ str(biases) +
                "= model = " + str(model))

            # there are n_entries outputs but we only want the last output
            return model


    def tf_run_reset(self, *args, **kwargs):
        with self.graph1.as_default():
            self.sess = tf.Session()
            sess = self.sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            return sess.run(*args, **kwargs) 


    def tf_run(self, *args, **kwargs):
        self.sess = self.sess if self.sess else tf.Session(graph=self.graph1)
        sess = self.sess
        tf.enable_eager_execution()
        with self.graph1.as_default():
            if not self.train_writer:
                self.train_writer = tf.summary.FileWriter('lstm_ekf.train', 
                                                          sess.graph)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            return sess.run(*args, **kwargs) 


    #Returns a trained LSTM model for R & Q Kalman Filter coefficient prediction
    def tune_model(self, epochs = n_epochs, labelfn = test_labels, cost = None,
                   nout = n_lstm_out):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE), \
             self.graph1.as_default():
            X = tf.placeholder("float", [n_entries, n_msmt])
            Y = tf.placeholder("float", [n_entries, nout])
            if use_logistic_regression:
                [X, Y, model, cost, optimizer] = tune_logistic_model()
            else:
                weights,biases = weights_biases(best_label(None,X,labelfn),nout)
                model = self.RNN(X, weights, biases) 
                cost = cost(model,X) if cost else tf.reduce_mean(tf.square(
                                                                 model-Y))
                optimizer = tf.train.MomentumOptimizer(learn_rate, 0.9)
            logger.debug("X,Y,model = " + str((X, Y, model)))
            train_op = optimizer.minimize(cost)
        return self.train_and_test(model, X, Y, train_op, cost, epochs, \
                                   labelfn)


    def tune_logistic_model(self)
        X = tf.placeholder(tf.float32, [n_entries, n_msmt, n_features])
        Y = tf.placeholder(tf.float32, [1, n_lstm_out, n_classes])
        wghts=tf.Variable(tf.truncated_normal([n_entries,n_features,n_classes]))
        biases = tf.Variable(tf.ones([n_entries, n_msmt, n_classes])*1e-2)
        model = self.RNN(X, wghts, biases)
        labels = tf.nn.softmax(Y)
        labels = tf.reshape(Y, [n_lstm_out, n_classes])
        logits = tf.reshape(model, [n_lstm_out, n_classes])
        cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, logits))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        return [X, Y, model, cost, optimizer]


    def train(self, train_op, cost, X, Y, train_data, costs, epoch, model):
        logger.info("train.train_data = " + str((len(train_data))))
        with self.graph1.as_default():
            tf.summary.scalar('cost', cost)
            for (i,(batch_x,batch_y)) in zip(range(len(train_data)),train_data):
                # Remember 'cost' contains the model
                ops = [train_op, cost, tf.summary.merge_all(), model] 
                logger.debug("batchx,batchy = " + str((batch_x, batch_y)))
                _, total_cost, summa, output = self.tf_run(ops, feed_dict = 
                    {X: batch_x if use_logistic_regression else to_size(batch_x,
                        n_msmt, n_entries), 
                     Y: batch_y if use_logistic_regression else to_size(batch_y,
                        n_lstm_out, n_entries)})
                self.train_writer.add_summary(summa, i)
                logger.debug("batchx = " +str(shape(batch_x)) + ", batchy = " + 
                    str(shape(batch_y)) + ", cost,out = " +
                    str((total_cost,output[-1]))+
                    ", batch " + str(i+1) + " of " + str(len(train_data)) + 
                    ", epoch " + str(epoch+1) + " of " + str(n_epochs)) 
                self.set_lstm_initialized()
                costs.append(total_cost)
                if iscostconverged(costs):
                    logger.debug("converged, window = " + str(costs[-5:]))
                    break


    def test(self, model, X, Y, test_data):
        [mse, cost, output] = [[], [], []]
        if use_logistic_regression:
            test_logistic(model, X, Y, test_data)
        else:
            for test_x,test_y in test_data:
                test_x = to_size(test_x, n_msmt, n_entries)
                test_y = to_size(test_y, -1, n_entries)
                with self.graph1.as_default():
                    labels = tf.reshape(Y, [-1, n_entries])
                    preds = tf.reshape(model, [-1, n_entries])
                    tf_mse = tf.losses.mean_squared_error(labels, preds)
                    tf_cost = tf.reduce_mean(tf.square(model - Y))
                    [mse, cost, output] = self.tf_run([tf_mse, tf_cost, model],
                        feed_dict={X:test_x, Y:test_y})
                    logger.debug("testx = " + str(test_x) + ", testy = " + 
                        str(test_y) + ", X = " + str(X) + ", Y = " + str(Y) +
                        ", (output,test_y) = " + str(list(zip(output, test_y))))
                logger.debug("LSTM MSE = " + str(mse) + ", cost = " + str(cost)) 
                print("LSTM MSE = " + str(mse) + ", cost = " + str(cost)) 
                pickleadd("test_costs.pickle", [mse])
        return [model, X, mse]


    def lstm_initialized(self):
        return self.initialized


    def set_lstm_initialized(self):
        self.initialized = True


    def train_and_test(self, model, X, Y, train_op, cost, epochs, 
                       labelfn=test_labels):
        (test_data, costs, labels) = ([], [], [])

        # Training
        for epoch in range(epochs):
            labels = labelfn(model, X, [])
            batch_data = [feature_classes(l[1:]) for l in labels]
            train_data = batch_data[0 : int(len(batch_data)*0.75)]
            test_data = test_data + batch_data[int(len(batch_data)*0.75) : ]
            start = time_ns()
            self.train(train_op, cost, X, Y, train_data, costs, epoch, model)
            logger.debug("Epoch = " + str((epoch,epochs)) + ", train_data = " +
                str(shape(train_data)) + ", test_data = " + 
                str(shape(test_data)) + ", costs = " + str(costs) + ", time = "+
                str(time_ns() - start) + " ns")
            pickleconc("train_costs.pickle", [costs[-1:]])
            pickleconc("train_times.pickle", [[time_ns()-start]])
            #if iscostconverged(costs):
            #    logger.debug("converged, window = " + str(costs[-5:]))
            #    break
         
        return self.test(model, X, Y, test_data)


    def test_logistic(self, model, X, Y, test_data):
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
            accs.append(self.tf_run(
                tf.stack([tf_accuracy, tf_recall, tf_precision, tf_tn, tf_fp]), 
                feed_dict={X:test_x, Y:test_y}))
            logger.info("pred,labels="+str(self.tf_run(tf.stack([preds,labels]),
                feed_dict={X:test_x, Y:test_y})))
            logger.info("x = " + str(test_x) + ", y = " + str(test_y))
            logger.info("acc,recall,precision = "+str((accs[-1],test_y,test_x)))
            pickleadd("test_logit_accuracy.pickle", accs[-1].flatten())
        return accs[-1] 



if __name__ == "__main__":
    Lstm().tune_model(15)
    logger.debug("done")
    print("Output in lstm_ekf.log")
