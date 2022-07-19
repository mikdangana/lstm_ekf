import yaml, logging, logging.handlers
import ast, asyncio
from numpy import array, resize, zeros, float32, matmul, identity, shape
from numpy import ones, dot, divide, subtract
from numpy.linalg import inv
from functools import reduce
from random import random
from subprocess import Popen
from utils import *


state_ids = []  # initialized in init_config_variables()
psize = 8
n_proc = 3
n_msmt = psize * n_proc # Kalman z
dimx = n_msmt
n_coeff = dimx * 2 + n_msmt # Kalman q, f, r
#n_entries = 3
n_entries = 10 
n_covariance = 3
n_lstm_out = n_coeff
n_lqn_out = 7
n_classes = int((5 - (-5)) / 1)
n_features = n_classes
# number of units in RNN cell
n_hidden = 3
learn_rate = 0.01
learn_threshold = 1e-4
n_epochs = 8
n_iterations = 500
use_logistic_regression = False
config = None
tasks = []
cfg_info = {}
state_file = "lstm_ekf.state"
initialized = False
config = None
norm = 1e10
norms = [norm, norm, norm, norm, norm, norm, 100, 100]
procs = []
n_user_rate_s = 1
n_users = 10
n_client_worker = 10
n_samples = -1
n_comp = -1
predictive = True
active_monitor = True
search = None
replace = None

ecount = 0
Gf = 3 # KF-based NN-weight computation frequency
Hf = 3 # EKF H matrix computation frequency

logger = logging.getLogger("Config")
    

def process_args():
    global n_coeff
    global n_epochs
    global n_iterations
    global n_msmt
    global n_lstm_out
    global n_entries
    global active_monitor
    global dimx
    global n_coeff
    global n_client_worker
    global n_users
    global n_samples
    global search
    global replace
    global predictive
    args = sys.argv[1:]
    for i,j in zip(args, args[1:] + ['']):
        if i == "--twod" or i == "-2d":
            n_coeff = n_msmt * n_msmt * 3
            logger.info("set n_coeff to " + str(n_coeff))
        elif i == "--epochs" or i == "-e":
            n_epochs = int(j)
            logger.info("set n_epochs to " + j)
        elif i == "--iterations" or i == "-i":
            n_iterations = int(j)
            logger.info("set test iterations to " + j)
        elif i == "--passive" or i == "-p":
            active_monitor = False
        elif i == "--n_msmt":
            n_msmt = int(j)
            dimx = n_msmt
            n_coeff = dimx * 2 + n_msmt # Kalman q, f, r
        elif i == "--n_lstm_out":
            n_lstm_out = int(j)
        elif i == "--n_entries":
            n_entries = int(j)
        elif i == "--n_client_worker":
            n_client_worker = int(j)
        elif i == "--n_users":
            n_users = int(j)
        elif i == "--n_samples":
            n_samples = int(j)
        elif i == "--search":
            search = j
        elif i == "--replace":
            replace = j
        elif i == "--predictive":
            predictive = "true" in j.lower()
        elif i == "--help" or i == "-h":
            usage()
            exit()

process_args()


def do_action(x_prior, host=None):
    global tasks
    tasks = get_config('lqn-tasks') if not len(tasks) else tasks
    ids = find(get_config("lqn-hosts"), host)
    pred = convert_lqn([p[0] for p in x_prior], host)
    thresh_cfg = sublist(get_config("lqn-thresholds"), ids)
    ttypes = [[k for k,v in i.items()][0] for i in thresh_cfg]
    thresholds = [float([v for k,v in i.items()][0]) for i in thresh_cfg]
    logger.info("prediction,thresholds,host = " + str((pred,thresholds,host)))
    for i,threshold in zip(range(len(thresholds)), thresholds):
        logger.info("prior.ids,pred[i],t ="+str((ids,pred[i+ids[0]],threshold)))
        if crossed(pred[i + ids[0]], threshold, ttypes[i]): 
            res_info = solve_lqn(i + ids[0]) # search for LQN input value
            for j,res in zip(range(len(tasks)), res_info[len(tasks):]):
                resname = [k for k,v in tasks[j].items()][0]
                logger.info("Resource task,pred,j = " + str((tasks[j],res,j)))
                if round(res) > tasks[j][resname]:
                    run_actions(get_config("lqn-provision-actions"), j)
                    tasks[j][resname] = tasks[j][resname] + 1
                elif round(res) < tasks[j][resname] and tasks[j][resname] > 0:
                    run_actions(get_config("lqn-deprovision-actions"), j)
                    tasks[j][resname] = tasks[j][resname] - 1


def crossed(v, t, ttype):
    return ttype == "upper" and v >= t or ttype == "lower" and v < t


def solve_lqn(metric_id):
    # Search for LQN input that produces the lowest value of associated metric
    lqnstr = lambda a,b: "["+str(a if a>0 else 1)+","+str(b if a>0 else 2)+"]"
    to_lqn = lambda lst: reduce(lqnstr, lst)
    for task in tasks:
        for k,v in task.items():
            logger.info("lqn.input = " + str({k: v}))
            os_run(get_config(['model-update-cmd'], [k, to_lqn([v-1, v+1])]))

    out = os_run(get_config('model-solve-cmd'))
    logger.info("metric_id="+str(metric_id)+", rows="+str(len(out.split("\n"))))
    metric=lambda row:float(row[3:][metric_id] if len(row[3:])>metric_id else 0)
    low = lambda a,b: a if metric(a) < metric(b) else b
    okrows = lambda rows: filter(lambda row: len(row) > 1, rows)
    rows = [l.split(", ") for l in okrows(out.split("\n")[1:])]
    best = reduce(low, rows)
    logger.info("rows = " + str(rows))
    return [float(r) for r in best]



def convert_lqn(msmts, host):
    global n_comp
    state = load_state()
    lqnout =solve_lqn(0) if n_comp<0 or not state or not host in state else None
    n_comp = len(lqnout) if n_comp < 0 else n_comp
    msmts = [array(m).T[0] for m in msmts]
    msmts.extend([msmts[-1] for i in range(n_comp - len(msmts))])
    logger.debug("msmts,shape = " + str((msmts, shape(msmts))))
    if not state or not host in state:
       m, c = solve_linear(lqnout, msmts)
       merge_state({host: {"lqn-ekf-model": {"m": m, "c": float(c)}}})
       state = load_state()
    model = state[host]['lqn-ekf-model'] if host in state else None
    (m, c) = (model['m'], model['c']) if model else (1, 0)
    return c + array(dot(getpca(n_comp, msmts).T, m)).T[0]



def solve_lqn_input(inputs, metids = None):
    to_lqn = lambda a: str(a) if isinstance(a, list) else "["+str(a)+"]"
    for inp in inputs:
        for k,v in inp.items():
            logger.info("lqn.input = " + str({k: v}))
            os_run(get_config(['model-update-cmd'], [k, to_lqn(v)]))

    try:
        out = os_run(get_config('model-solve-cmd'))
        out = out.split("\n")
        logger.info("metric_id = " + str(metids) + ", rows = " + str(len(out)))
        #metric = lambda r: [float(r[int(i)]) for i in (metids if metids and len(metids) else range(len(r)))]
        okrows = lambda rows: filter(lambda row: len(row) > 1, rows)
        rows = [[float(t) for t in l.split(", ")] for l in okrows(out[1:])]
        logger.info("rows = " + str(rows))
        return rows
    except:
        return [zeros(n_lqn_out)]



def run_actions(cmds, pos):
    logger.info("pos,cmds = " + str((pos, cmds)))
    if pos >= len(cmds):
        return
    cmd = cmds[pos]
    if isinstance(cmd, type("")):
        os_run(cmd)
    elif isinstance(cmd, list):
        for step in cmd:
            os_run(step)


def run_action(action):
    run_state = load_state()
    run_state = {"app": 1, "db": 1, "solr": 1} if not run_state else run_state
    action = {k:v for k,v in zip(run_state.keys(), action[0:3])}

    for res,count in action.items():
        logger.debug("res = " + str(res) + ", count = " + str(count) + 
            ", run_state = " + str(run_state) + ", action = " + str(action))
        if int(count) > int(run_state[res]):
            for step in get_steps("lqn-provision-cmds", res):
                os_run(step)
            run_state[res] = int(run_state[res]) + 1
        elif int(count) < int(run_state[res]):
            for step in get_steps("lqn-deprovision-cmds", res):
                os_run(step)
            run_state[res] = int(run_state[res]) - 1
    save_state(run_state)
    return true


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


def dups(cfg, *dup_key):
    steps = []
    variables = dict(ast.literal_eval(get_config(dup_key)))

    logger.debug("variables =" + str(variables.items()) + ", cfg = " + str(cfg))
    for step in (cfg if cfg else []):
        for cmd in step if isinstance(step, list) else [step]: 
            for k,v in variables.items():
                if '<' + str(k) + '>' in cmd:
                    steps.append(re.sub(r'<' + str(k) + '>', v, cmd))
    return steps


def load_config():
    global config
    yamlfile = os.path.dirname(os.path.abspath(__file__)) + "/../lstm_ekf.yaml"
    logger.info("yaml file = " + str(yamlfile))
    with open(yamlfile, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.error("Unable to load config", ex)
    return config


def get_config(path, params = []):
    if not config:
        load_config()
    val = get(config, set_variables(path))
    for (i, param) in zip(range(0, len(params)), params):
        val = re.sub(r'<param' + str(i) + '>', str(param), str(val))
    return set_variables(val)


def set_variables(val):
    if 'variables' in config:
        for k, v in config['variables'].items():
            k = "<" + str(k) + ">"
            val = set_variable(val, k, v)
    val = set_variable(val, search, replace) if search and replace else val
    return val


def set_variable(val, k, v):
    if isinstance(v, type("")) and k in str(val):
        if isinstance(val, list):
            val = [set_variable(vali, k, v) for vali in val]
        else:
            val = re.sub(r'' + k, v, str(val))
    elif isinstance(v, list) and val:
        if isinstance(val, type("")) and k in val:
            for vi in v:
                val = re.sub(r'' + k, vi, str(val))
        elif isinstance(val, list) and (k in str(val)):
            for vi in v:
                val = [set_variable(vali, k, v) for vali in val]
    return val


def lqn_state(host):
    ids = find(get_config('lqn-hosts'), host)
    logger.debug("lqn_state().ids = " + str(ids))
    return list(map(lambda t: [v for k,v in t.items()][0], tasks))[ids[0]:ids[-1]+1]


def run_async(cmd_cfg):
    cmd = get_config(cmd_cfg).split(" ")
    logger.info("cmd_cfg = " + cmd_cfg + ", cmd = " + str(cmd))
    procs.append(Popen(cmd))
    logger.info("launched process "+str(procs[-1].pid)+" for cmd = "+str(cmd))
    return procs[-1]


def close_async():
    for proc in procs:
        proc.terminate()
        logger.info("process " + str(proc.pid) + " terminated")



def normalize(v):
    (factor, i, s) = (1, v[0], v[1])
    if "g" in v[1] and v[1].replace("g", "").isdigit():
        (factor, s) = (10e9, v[1].replace("g", ""))
    elif "m" in v[1] and v[1].replace("m", "").isdigit():
        (factor, s) = (10e6, v[1].replace("m", ""))
    elif "k" in v[1] and v[1].replace("k", "").isdigit():
        (factor, s) = (10e3, v[1].replace("k", ""))
    n = float32(s)*factor/norms[i%psize] if s.replace('.','').isdigit() else 0
    return n



def usage():
    print("\nUsage: " + sys.argv[0] + " [-h | -e | -i | -2d] [n]\n" +
        "\nSets run config parameters\n" +
        "\nOptional arguments:\n\n" +
        "-h, --help              Show this help message and exit\n" +
        "-2d, --twod             Set the 2d (vs diagonal) n_coeff value\n" +
        "-e, --epochs      n     Set number of epochs\n" +
        "-i, --iterations  n     Set the number of iterations")


def init_config_variables():
    global state_ids, dimx, n_coeff, n_lstm_out, tasks
    state_ids = find(get_config('lqn-hosts'), None)
    #n_coeff = dimx * 2 + n_msmt # Kalman x
    #n_lstm_out = n_msmt #n_coeff
    tasks = get_config('lqn-tasks')
    logger.info("init_config_variables() done")


init_config_variables()


def test_config():
    print("Loaded config = " + str(config))
    print("model-update-cmd = " + str(get_config(['model-update-cmd']))) 
    print("provision-cmds.solr = " + str(get_config(['provision-cmds','solr'])))



def test_threshold_tasks():
    thresh = get_config('lqn-thresholds')
    print("thresh = " + str(isinstance(thresh, list)))
    print("lqn-thresholds = " + str(list(map(lambda t: float(t)+1, thresh)))) 
    tasks = get_config('lqn-tasks')
    print("tasks = " + str(tasks) + ", [0] = " + str([k for k,v in tasks[0].items()][0]))


def test_lqn_hosts():
    print("solve_lqn() output = " + str(solve_lqn(0)))
    print("find(lqn-hosts, db-host) = " + str(find(get_config('lqn-hosts'), get_config('lqn-hosts')[0])))


def test_linear():
    print("do_action() test")
    lqnval = solve_lqn(0)
    msmts = [[random() for i in range(n_msmt)] for j in range(10)]
    msmts = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    print("lqnval,msmts = " + str((lqnval, msmts)))
    m, c = solve_linear(lqnval, msmts)
    merge_state({"lqn-ekf-model": {None: {"m": m, "c": float(c)}}})
    state = load_state()
    model = state['lqn-ekf-model'] if 'lqn-ekf-model' in state else None
    (m, c) = (model['m'], model['c']) if model else (1, 0)
    print("m,c,msmts.shape = " + str((m,c,shape(msmts))))
    print("pcs = " + str(getpca(len(lqnval), msmts)))
    print("dot(pcs,m) = " + str(dot(getpca(len(lqnval), msmts).T, m)))
    print(get_config("login-24.148.194.126") + " -t 'cmd'")


def test_do_action():
    msmts = array([[random() for i in range(n_msmt)] for j in range(10)])
    prior = [array([m]).T for m in msmts]
    do_action(prior, "127.0.0.1")


if __name__ == "__main__":
    load_config()
    test_config()
    test_threshold_tasks()
    test_lqn_hosts()
    run_async("activity-cmd")
    test_linear()
    print(str(solve_lqn_input([{'nUsers': 1, 'nWebServers': 2, 'nDb': 1, 'nSearch': 1}])))
    #close_asyncs()
