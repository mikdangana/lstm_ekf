import tensorflow as tf
import os, re, sys, traceback
import yaml, logging, logging.handlers
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random

logger = logging.getLogger("LstmKalmanTuner")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename='lstm_ekf.log', format='%(levelname)s %(asctime)s in %(funcName)s() %(filename)s-%(lineno)s: %(message)s \n', level=logging.INFO)

n_msmt = 8 # Kalman z
n_param = 8 # Kalman x
n_entries = 250
# number of units in RNN cell
n_hidden = 1 #512
learn_rate = 0.00001
n_epochs = 15
state_ids = range(0, n_msmt)
config = None
state_file = "lstm_ekf.state"
initialized = False


def measurements():
    cmd = "top -b -n 2 | grep -v Tasks | grep -v top | grep -v %Cpu | " + \
        "grep -v KiB | grep -v PID | grep [0-9] | " + \
        "awk '{print $1,$3,$4,$5,$6,$7,$9,$10}'"
    pstats = do(cmd)
    logger.info(len(pstats.split()))
    pstatsf = list(map(lambda i: float32(i) if i!="rt" else 0, pstats.split()))
    logger.info("type = " + str(type(pstatsf[0])))
    return pstatsf


def msmt_tensor():
    x = tf.Variable( resize(measurements(), (n_entries,n_msmt)), 
        name="inputs", dtype=tf.float32 )
    return x


def RNN(x, weights, biases):

    # reshape to [1, n_msmt]
    #x = tf.reshape(x, [-1, n_msmt])
    #logger.info("RNN.x = " + str(x))

    # Generate a n_msmt-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #tf.split(x, n_msmt, 1)
    logger.info("RNN.x1 = " + str(x) + ", n_hidden = " + str(n_hidden))

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.nn.rnn_cell.LSTMCell(1) #n_hidden)
    #rnn_cell.zero_state(1, dtype=tf.float32)
    logger.info("RNN.rnn_cell = " + str(rnn_cell))

    # generate prediction
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, inputs=[x], dtype=tf.float32)
    logger.info("outputs = " + str(outputs) + ", states = " + str(states))

    # there are n_entries outputs but
    # we only want the last output
    return outputs #tf.matmul(outputs[-1], weights['out']) + biases['out']


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


def get_baseline(model, sample_size):
    (baseline, ekf, msmts, history) = ([], None, [], [])
    for i in range(0, sample_size):
        new_msmts = measurements()
        logger.info("i = " + str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            (best_coeffs, best_accuracy) = ([], 0)
            if not ekf:
                coeffs = predict_coeffs(model, msmts)
                ekf = build_ekf(coeffs, [msmts]) # history)
            elif len(history):
                logger.info("get_baseline.history = " + str(shape(history)))
                update(ekf, history)
            baseline.append(ekf_accuracy(ekf, new_msmts))
        msmts = new_msmts
        history.append([msmts])
    return baseline


# Generates training labels by a bootstrapping/active-learning approach 
def bootstrap_labels(model):
    (labels, ekf, msmts, history, sample_size) = ([], None, [], [], 10)
    ekf_baseline = get_baseline(model, sample_size)
    for i in range(0, sample_size):
        new_msmts = measurements()
        logger.info("i = " + str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            (best_coeffs, best_accuracy) = ([], 0)
            for j in range(0, 10):
                coeffs = predict_coeffs(model, msmts)
                logger.info("j =" + str(i) + ", coeffs = " + str(shape(coeffs)))
                ekf = build_ekf(coeffs, [msmts]) #history) 
                accuracy = ekf_accuracy(ekf, new_msmts)
                if accuracy >= best_accuracy: # TODO max(best_accuracy, ekf_baseline[i]):
                    best_coeffs = coeffs
                #if i % sample_size/10 == 0:
                  #  do_action(ekf, msmts)
            if len(best_coeffs): # Only add labels if accuracy > ekf_baseline
                labels.append([msmts, best_coeffs])
        logger.info("i = " + str(i) + ", labels = " + str(shape(labels)))
        msmts = new_msmts
        history.append(msmts)
    return labels


def get_accuracy(pair):
    (n, d) = pair
    if d==0:
        return 1 if n==0 else 0
    elif n>d:
        return d/n
    return n/d


def ekf_accuracy(ekf, msmts):
    ekf.predict()
    (state, n_state) = ([msmts[i] for i in state_ids], len(state_ids))
    state = array(state)
    state.resize(n_state, 1)
    logger.info("state = " + str(shape(state)) + ", n_state = " + str(n_state) + ", prior = " + str(shape(ekf.x_prior)))
    #logger.info("subtract = " + str(shape(subtract(ekf.x_prior, state))) + ", ones = " + str(shape(ones((1, n_state)))))
    #logger.info("divide = " + str(shape(divide(subtract(ekf.x_prior, state), state))) + ", ones = " + str(shape(ones((1, n_state)))))
    #acc = lambda n,d: 1 if n==d else [0 if abs(n-d)>d or d==0 else abs(n-d)/d][0]
    accuracy = sum(map(get_accuracy, zip(ekf.x_prior, state)))/n_state #divide(subtract(ekf.x_prior, state), state))[0][0]
    logger.info("x_prior = " + str(shape(ekf.x_prior)) + ", accuracy = " + str(accuracy))
    return accuracy


def do_action(ekf, msmts):
    stateinfo = {"1": "nWebServers", "2": "nDb", "3": "nSearch"} # TODO fix keys
    window_size = 2 # TODO: should be based on variance of a state variable
    state = ekf.x_prior  # state prediction
    to_lqn = lambda lst: reduce(lambda a,b: str(a)+","+str(b), lst)

    for k,v in variables.items():
        util = utilization(k, state, stateinfo)
        bounds = (util*100 - windo_size, util*100 + window_wize)
        window = range(1 if bounds[0] < 1 else bounds[0], bounds[1])
        do(get_config(['model-update-cmd'], [v, to_lqn(window)]))

    out = do(get_config('model-solve-cmd'))
    util = lambda row: reduce(lambda a,b: a+b, list(map(float, row[3:])))
    low = lambda a,b: a if util(a) < util(b) else b
    best = reduce(low, list(map(lambda l: l.split(", "), out.split("\n")[1:])))
    return run_action(best)


def run_action(action):
    run_state = load_state()
    run_state = {"app": 1, "db": 1, "solr": 1} if not run_state else run_state
    action = {k:v for k,v in zip(run_state.keys(), action[0:3])}

    for res,count in action:
        if count > run_state[res]:
            for step in get_steps("provision-cmds", res):
                do(step)
            run_state[res] = run_state[res] + 1
        elif count < run_state[res]:
            for step in get_steps("deprovision-cmds", res):
                do(step)
            run_state[res] = run_state[res] - 1
    save_state(run_state)
    return True


def get_steps(*cmd_path):
    steps = []
    varMap = get_config("variables")
    cfg = get_config(cmd_path)

    for step in cfg if cfg else []:
        for cmd in step if isinstance(step, list) else [step]: 
            for k,v in varMap.items():
                cmd = re.sub(r'<' + k + '>', v, cmd)
                steps.append(cmd)
    return steps


def do(cmd):
    try:
        return os.popen(cmd).read()
    except Error as e:
        logger.debug("Error running command " + str(cmd), e)
    return None


def utilization(key, states, stateinfo): 
    maxval = get(stateinfo, key, 'max')
    return states[int(key)] / maxval if maxval else states[int(key)]


def get(o, *keys):
    if isinstance(o, dict):
        for k in keys:
            if isinstance(k, list):
                return get(o, k)
            elif not k in o:
                return None
            else:
                o = o[k]
        return o
    return None


# Build and update an EKF using the provided measurments data
def build_ekf(coeffs, z_data): 
    ekf = ExtendedKalmanFilter(dim_x=n_param, dim_z= n_msmt)
    q = array(coeffs)
    q.resize(n_param, n_param) # TODO need to determine size
    ekf.Q = q
    r = array(coeffs)
    r = r.resize(n_msmt, n_msmt) # TODO need to determine size
    #ekf.R = r
    return update(ekf, z_data)


def update(ekf, z_data):
    hjacobian = lambda x: identity(len(x))
    hx = lambda x: x
    logging.info("ekf.x = " + str(ekf.x) + ", shape = " + str(shape(ekf.x)) + ", q.shape = " + str(shape(ekf.Q)) + ", q.type = " + str(type(ekf.Q)) + ", z_data = " + str(shape(z_data)))
    for z in z_data:
        z = array(z)
        z.resize(n_msmt, 1)
        logging.info("update.z = " + str(shape(z)) + ", x_prior = " + str(shape(ekf.x_prior)) + ", hjacobian = " + str(hjacobian([1, 2])))
        ekf.update(z, hjacobian, hx)
    return ekf


def load_config():
    global config
    with open("lstm_ekf.yaml", 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.error("Unable to load config", ex)
    return config


def get_config(path, params = []):
    if not cfg:
        load_config()
    val = get(config, path)
    for (i,param) in zip(range(0,len(params)), params):
        val = re.sub(r'<param' + i + '>', param)
    return val


def load_state():
    with open(state_file, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.error("Unable to load state", ex)
    return None


def save_state(runtime_state):
    with open(state_file, 'w') as out:
        try:
            out.write(yaml.dump(runtime_state))
        except yaml.YAMLError as ex:
            logger.error("Unable to save state", ex)
            return False
    return True


def to_size(data, width):
    input = array(data)
    input.resize(n_entries, width)
    logger.debug("data = " + str(len(data)) + ", input = " + str(shape(input)))
    return input

def repeat(v, n = n_msmt):
    if isinstance(v, type(lambda i:i)):
        return map(v, range(0,n))
    else:
        return map(lambda i: v, range(0,n))


def tf_run(*args, **kwargs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess.run(*args, **kwargs)


with tf.Session() as sess:

    X = tf.placeholder("float", [n_entries, n_msmt])
    Y = tf.placeholder("float", [n_entries, 1])

    #x = msmt_tensor()
    #logger.warning(x)

    # coefficients in noise and channel matrices, flattened out
    vocab_size = n_param

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]), name='w')
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]), name='b')
    }

    model = RNN(X, weights, biases)
    logger.debug("model = " + str(model))

    # TODO: implement EKF-based cost function
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    cost = tf.reduce_mean(tf.square(model - Y))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(cost)

    test_data = []

    # Training
    for epoch in range(0, n_epochs):
        batch_data = bootstrap_labels(model)
        train_data = batch_data[0:len(batch_data)*75]
        test_data = test_data + batch_data[len(batch_data)*75:]
        for (batch_x, batch_y) in train_data:
            # Remember 'cost' contains the model
            #try:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            _, total_cost = sess.run([train_op, cost], 
                feed_dict = {X: to_size(batch_x,n_msmt), Y: to_size(batch_y,1)})
            logger.debug("batchx = " + str(shape(batch_x)) + ", batchy = " + str(shape(batch_y)) + ", cost = " + str(total_cost))
            #exit()
            initialized = True
            mean_cost = total_cost / len(train_data)
            #except:
             #   logger.error("Error training model: " + str(sys.exc_info()))
              #  exit()
            logger.debug("Epoch = " + str(epoch) + ", cost = " + str(cost))
    logger.debug("LSTM Training finished")

    # Testing
    pred = tf.nn.softmax(model)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_x = list(map(lambda t: t[0], test_data))
    test_y = list(map(lambda t: t[1], test_data))
    logger.debug("LSTM Accuracy = " + str(accuracy.eval({X:test_x, Y:test_y})))

