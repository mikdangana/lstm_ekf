import tensorflow as tf
import os, re
import yaml, logging
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, resize, zeros, float32, matmul, identity
from numpy.linalg import inv
from functools import reduce

logger = logging.getLogger("LstmKalmanTuner")
logger.setLevel(logging.DEBUG)

n_msmt = 8 # Kalman z
n_param = 3 # Kalman x
n_entries = 250
learn_rate = 0.00001
n_epochs = 15
state_ids = range(0, n_msmt)
config = None
state_file = "lstm_ekf.state"

def measurements():
    cmd = "top -b -n 2 | grep -v Tasks | grep -v top | grep -v %Cpu | " + \
        "grep -v KiB | grep -v PID | grep [0-9] | " + \
        "awk '{logger.debug $1,$3,$4,$5,$6,$7,$9,$10}'"
    pstats = do(cmd)
    logger.debug(pstats)
    logger.debug(len(pstats.split()))
    pstatsf = list(map(lambda i: float32(i) if i!="rt" else 0, pstats.split()))
    logger.debug("type = " + str(type(pstatsf[0])))
    return pstatsf


def msmt_tensor():
    x = tf.Variable( resize(measurements(), (n_entries,n_msmt)), 
        name="inputs", dtype=tf.float32 )
    return x


def RNN(x, weights, biases):

    # reshape to [1, n_msmt]
    x = tf.reshape(x, [-1, n_msmt])
    logger.debug("RNN.x = " + str(x))

    # Generate a n_msmt-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_msmt,1)
    logger.debug("RNN.x1 = " + str(x))

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    logger.debug("RNN.rnn_cell = " + str(rnn_cell))

    # generate prediction
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    logger.debug("RNN.x3 = " + str(x))

    # there are n_msmt outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def predict_coeffs(model, newdata):
    checkpoint_dir = "checkpoint/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    conf = tf.ConfigProto(allow_safe_placement=True, log_device_placement=False)
    sess = tf.Session(config = conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input = graph.get_operation_by_name("input").outputs[0]
        prediction = graph.get_operation_by_name("prediction").outputs[0]
        #newdata = msmt_tensor()
        output = sess.run(prediction, feed_dict={input:newdata})
        logger.debug("output = " + str(output) + ", input = " + str(newdata))
        return output
    return []


def get_baseline(model, sample_size):
    (baseline, ekf, msmts, history) = ([], None, None, [])
    for i in range(0, sample_size):
        new_msmts = measurements()
        if msmts:
            (best_coeffs, best_accuracy) = ([], 0)
            if not ekf:
                ekf = build_ekf(coeffs, history)
            else:
                ekf.update(history)
            baseline.append(ekf_accuracy(ekf, new_msmts))
        msmts = new_msmts
        history.append(msmts)
    return baseline


# Generates training labels by a bootstrapping/active-learning approach 
def bootstrap_labels(model):
    (labels, ekf, msmts, history, sample_size) = ([], None, None, [], 100)
    (ekf_baseline, action_step) = (get_baseline(sample_size), sample_size/10)
    for i in range(0, sample_size):
        new_msmts = measurements()
        if msmts:
            (best_coeffs, best_accuracy) = ([], 0)
            for j in range(0, 100):
                coeffs = predict_coeffs(model, msmts)
                ekf = build_ekf(coeffs, history) 
                accuracy = ekf_accuracy(ekf, new_msmts)
                if accuracy >= max(best_accuracy, ekf_baseline[i]):
                    best_coeffs = coeffs
                if i % action_step == 0:
                    do_action(ekf, msmts)
            if len(best_coeffs): # Only add labels if accuracy > ekf_baseline
                labels.append([msmts best_coeffs])
        msmts = new_msmts
        history.append(msmts)
    return labels


def ekf_accuracy(ekf, msmts):
    ekf.predict()
    (state, n_state) = (msmts[state_ids], len(state_ids))
    return 1-inner(ones(n_state), divide(substract(ekf.x_priori, state), state))


def do_action(ekf, msmts):
    stateinfo = {"1": "nWebServers", "2": "nDb", "3": "nSearch"} # TODO fix keys
    window_size = 2 # TODO: should be based on variance of a state variable
    state = ekf.x_priory  # state prediction
    to_lqn = lambda lst: reduce(lambda a,b: str(a)+","+str(b), lst)

    for k,v in variables.items():
        util = utilization(k, state, stateinfo)
        bounds = (util*100 - windo_size, util*100 + window_wize)
        window = range(bounds[0] < 1? 1: bounds[0], bounds[1])
        do(get_config(['model-update-cmd'], [v, to_lqn(window)]))

    out = do(get_config('model-solve-cmd'))
    util = lambda row: reduce(a,b: a+b, map(float, row[3:]))
    min_util = lambda a,b: a if util(a) < util(b) else b
    best = reduce(min_util, map(lambda l: l.split(", "), out.split("\n")[1:]))
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
    except e:
        logger.error("Error running command " + str(cmd), e)
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
    ekf = KalmanFilter(dim_x=n_param, dim_z= n_msmt)
    q = array(resize(coeffs, n_param, n_msmt)) # need to determine size
    ekf.Q = block_diag(q, q) 
    r = array(resize(coeffs, n_msmt, n_entries)) # need to determine size
    ekf.R = block_diag(r, r)
    hjacobian = lambda x: identity(len(x))
    hx = lambda x: x
    for z in z_data:
        ekf.update(z, hjacobian, hx)
    return ekf


def load_config():
    global config
    with open("lstm_ekf.yaml", 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.debug ex
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
            logger.debug ex
    return None


def save_state(runtime_state):
    with open(state_file, 'w') as out:
        try:
            out.write(yaml.dump(runtime_state))
        except yaml.YAMLError as ex:
            logger.debug ex
            return False
    return True


#x = msmt_tensor()
#logger.debug(x)

# coefficients in noise and channel matrices, flattened out
vocab_size = 3

# number of units in RNN cell
n_hidden = 512

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


X = tf.placeholder("float", [n_entries, n_msmts])
Y = tf.placeholder("float", [None, n_param])

model = RNN(X, weights, biases)
logger.debug("model = " + str(model))

# TODO: implement EKF-based cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate).minimize(cost)
train_op = optimizer.minimize(cost)

#_, acc, loss, onehot_pred = tf.Session.run([optimizer, accuracy, cost, pred], feed_dict={x: x, y: y})

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    test_data = []
    # Training
    for epoch in range(0, n_epochs):
        batch_data = bootstrap_labels(pred)
        train_data = batch_data[0:len(batch_data)*75]
        test_data = test_data + batch_data[len(batch_data)*75:]
        for (batch_x, batch_y) in train_data:
            # Remember 'cost' contains the model
            _, total_cost = session.run([train_op, cost], 
                    feed_dict = {X: batch_x, Y: batch_y})
            mean_cost = total_cost / len(train_data)
            logger.debug "Epoch = " + str(epoch) + ", cost = " + str(cost)
    logger.debug "LSTM Training finished"

    # Testing
    pred = tf.nn.softmax(model)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_x = map(lambda t: t[0], test_data)
    test_y = map(lambda t: t[1], test_data)
    logger.debug "LSTM Accuracy = " + str(accuracy.eval({X: test_x, Y: test_y}))

