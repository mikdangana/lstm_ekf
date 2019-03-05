import yaml, logging, logging.handlers
import ast, asyncio
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random
from utils import *


n_msmt = 8 * 6 # Kalman z
n_coeff = n_msmt * n_msmt * 3 # Kalman x
n_entries = 3
# number of units in RNN cell
n_hidden = 2
learn_rate = 0.00001
default_n_epochs = 1000
state_ids = range(n_msmt)
config = None
state_file = "lstm_ekf.state"
initialized = False
config = None
procs = []


def do_action(ekf, msmts):
    stateinfo = {"1": "nWebServers", "2": "nDb", "3": "nSearch"} # TODO fix keys
    window_size = 2 # TODO: should be based on variance of a state variable
    state = list(map(lambda s: s[0], ekf.x_prior))  # state prediction
    to_lqn = lambda lst: reduce(lambda a,b: "["+str(a)+","+str(b)+"]", lst)
    logger.debug("state = " + str(state))

    for k,v in stateinfo.items():
        util = utilization(k, state, stateinfo)
        bounds = (int(util*100 - window_size), int(util*100 + window_size))
        window = range(1 if bounds[0] < 1 else bounds[0], bounds[1])
        logger.debug("window = " + str(window))
        os_run(get_config(['model-update-cmd'], [v, to_lqn(window)]))

    out = os_run(get_config('model-solve-cmd'))
    logger.debug("out = " + str(out) + ", rows = " + str(len(out.split("\n"))))
    util = lambda row: reduce(lambda a,b: a+b, list(map(float, row[3:])), 1)
    low = lambda a,b: a if util(a) < util(b) else b
    okrows = lambda rows: filter(lambda row: len(row) > 1, rows)
    rows = list(map(lambda l: l.split(", "), okrows(out.split("\n")[1:])))
    best = reduce(low, rows)
    logger.debug("best = " + str(best) + ", rows = " + str(rows) + ", util = " + str(util(rows[0])) + ", low = " + str(low(rows[0], rows[1])))
    return run_action(best)


def run_action(action):
    run_state = load_state()
    run_state = {"app": 1, "db": 1, "solr": 1} if not run_state else run_state
    action = {k:v for k,v in zip(run_state.keys(), action[0:3])}

    for res,count in action.items():
        logger.debug("res = " + str(res) + ", count = " + str(count) + ", run_state = " + str(run_state) + ", action = " + str(action))
        if int(count) > int(run_state[res]):
            for step in get_steps("provision-cmds", res):
                os_run(step)
            run_state[res] = int(run_state[res]) + 1
        elif int(count) < int(run_state[res]):
            for step in get_steps("deprovision-cmds", res):
                os_run(step)
            run_state[res] = int(run_state[res]) - 1
    save_state(run_state)
    return true


def run_async(action):
    cmd = [get_config(action)]
    proc = yield asyncio.create_subprocess_exec(*cmd)

    procs.append(proc)
    return proc
    

def close_asyncs():
    for proc in procs:
        stdout, stderr = yield proc.terminate()


def get_steps(*cmd_path):
    steps = []
    variables = dict(ast.literal_eval(get_config("variables")))
    cfg = ast.literal_eval(get_config(cmd_path))

    logger.debug("variables =" + str(variables.items()) + ", cfg = " + str(cfg))
    for step in (cfg if cfg else []):
        for cmd in step if isinstance(step, list) else [step]: 
            for k,v in variables.items():
                cmd = re.sub(r'<' + str(k) + '>', v, cmd)
                steps.append(cmd)
    return steps


def load_config():
    global config
    with open("lstm_ekf.yaml", 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.error("Unable to load config", ex)
    return config


def get_config(path, params = []):
    global config
    if not config:
        load_config()
    val = get(config, path)
    for (i, param) in zip(range(0, len(params)), params):
        val = re.sub(r'<param' + str(i) + '>', str(param), str(val))
    if 'variables' in config:
        for k, v in config['variables'].items():
            k = "<" + str(k) + ">"
            logger.debug("k = " + str(k) + ", v = " + str(v) + ", val = " + str(val))
            if isinstance(v, type("")):
                val = re.sub(r'' + k, v, str(val))
            elif isinstance(v, list) and val and (k in val):
                return [re.sub(r'' + k, vi, str(val)) for vi in v]
    return val



if __name__ == "__main__":
    load_config()
    print("Loaded config = " + str(config))
    print("model-update-cmd = " + str(get_config(['model-update-cmd']))) 
