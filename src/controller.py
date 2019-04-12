from config import *
from utils import *
from ekf import *
from lstm import *
from client import *
from time import sleep, time
from numpy import subtract, concatenate, divide
import config
import threading
import io
from contextlib import redirect_stdout


test_msmt = []
dimz = int(n_msmt/2)
msmts = []
mcount = 1
threads = []
monitor_msmts = {}
ekfs = {}
generators = get_generators()
gid = len(generators)
simulated = False


# Only the first n_msmt values are used, the rest are ignored by LSTM & EKF
def measurements(host = ""):
    if simulated:
        return sim_measurements(host)
    cmd = get_config("memory-cmd-" + host)
    cmd = "top -b -n 1 -o %MEM" if not cmd else cmd
    cmd = cmd + " | grep \"^ *[0-9]\""
    if host and len(host):
        cmd = get_config("login-"+host) + " -o LogLevel=QUIET -t '" + cmd + "'"
    pstats = os_run(cmd + " | awk '{print $1,$3,$4,$5,$6,$7,$9,$10}'")
    pstats = pstats.split() if pstats else ""
    logger.info(str(len(pstats)) + " measurements retrieved")
    pstatsf = list(map(normalize, zip(range(len(pstats)), pstats)))
    pstatsf = pstatsf[0:dimz] + list(zeros(dimz))
    logger.info("parsed measurements, size=" + str(len(pstatsf)))
    pickleadd(host + "measurements.pickle", array(pstatsf).flatten())
    return pstatsf if len(pstatsf)==n_msmt else []


def sim_measurements(pre=""):
    global mcount
    mcount = mcount + 1
    if gid < len(generators):
        mcount = mcount % 50 if generators[gid] == math.exp else mcount
        pstatsf = [generators[gid](mcount) for i in range(n_msmt)]
    else:
        delta = 1e-3
        prev = msmts[-1] if len(msmts)%100 else zeros(n_msmt) + delta
        factors = list(range(n_msmt))
        factors.sort(key = lambda v: v, reverse=True)
        pstatsf = [prev[i]*(1+i)/10 for i in factors]
    msmts.append(pstatsf)
    pickleadd(pre + "measurements.pickle", array(pstatsf).flatten())
    logger.info("msmts = " + str(pstatsf))
    return pstatsf



def predict_coeffs(model, newdata, X, randomize=False):
    logger.info("LSTM initialized = " + str(lstm_initialized()))
    if not lstm_initialized() or randomize:
        def rand_coeffs(i): 
            return list(map(lambda n: uniform(0.0,1.0), range(n_coeff)))
        return list(map(rand_coeffs, range(n_entries)))
    input = to_size(newdata, n_msmt, n_entries)
    if model == None:
        return [[]]
    elif isinstance(model, list):
        lout = []
        for lstm in model:
            lout.append(tf_run(lstm, feed_dict={X:input}))
    else:
        output = tf_run(model, feed_dict={X:input})
    logger.debug("Coeff output = " + str(shape(output)) + ", predict = " + 
        str(output[-1]) + ", input = " + str(shape(newdata)))
    return output



# Creates a baseline coeff/EKF and tracks its performance over a sample history
def baseline_accuracy(model, sample_size, X, host):
    (baseline, ekf, msmts, history) = ([], None, [], [])
    for i in range(0, sample_size):
        new_msmts = measurements(host)
        logger.info("i = " + str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            if not ekf:
                coeffs = predict_coeffs(model, msmts, X)
                ekf = build_ekf(coeffs[-1], [msmts]) 
            elif len(history):
                logger.info("baseline_accuracy.history =" + str(shape(history)))
                update_ekf(ekf, history)
            baseline.append(ekf_accuracy(ekf, new_msmts))
        msmts = new_msmts
        history.append([msmts])
    return baseline



# Generates training labels by a bootstrap active-learning approach 
def bootstrap_labels(model, X, labels=[], sample_size=10, pre="boot", host=""):
    (labels, msmts, history, coeffs) = ([], [], [], [])
    (accuracies, ekf_baseline) = ([], 
        baseline_accuracy(model, sample_size, X, host))
    for i in range(sample_size):
        new_msmts = measurements(host)
        history.append(msmts)
        add_labels(labels, accuracies, coeffs, model, X, new_msmts, history, i)
        msmts = new_msmts
    pre = host + pre
    pickleconc(pre + "_accuracies.pickle", [a[-1] for a in accuracies])
    pickleconc(pre + "_profiles.pickle", [a[0] for a in accuracies])
    pickleconc(pre + "_coeffs.pickle", [c[0] for c in coeffs])
    labels.sort(key = lambda v: v[0], reverse=True)
    logger.info("batch label accuracies = " + 
        str([l[0] for l in labels[0:int(0.1*len(labels))]]))
    logger.info("batch labels = " + 
        str([l[2] for l in labels[0:int(0.1*len(labels))]]))
    return labels[0:int(0.1 * len(labels))]



def add_labels(labels, accuracies, coeffs, model, X, new_msmts, history, step):
    (action_interval, pred_per_sample, msmts) = (40, 10, history[-1])
    if len(msmts):
        for j in range(pred_per_sample):
            coeffs.append(predict_coeffs(model, msmts, X))
            ekf = build_ekf(coeffs[-1][-1], [msmts]) 
            accs, accuracy = ekf_accuracies(ekf, new_msmts)
            #add_randoms_feature(labels, model, X, new_msmts, msmts)
            #add_raws_feature(labels, new_msmts, msmts)
            if j==0:
                ratios = add_ratio_feature([], new_msmts, history)
                add_covariance_feature(labels, new_msmts, history, ratios)
            logger.info("sample = " + str(step) + ", pred_per_sample = " + 
                str(j) + " of " + str(pred_per_sample) + ", coeffs = " +
                str(coeffs[-1][-1]) + ", accuracy = " + str(accuracy))
            accuracies.append([accs, accuracy])
            labels.append([accuracy, to_size(msmts, n_msmt, n_entries), 
                                     to_size(coeffs[-1], n_entries, n_coeff)])
        if not simulated:
            sleep(0.5)
            if step % action_interval == 0:
                do_action(ekf, msmts)



def add_randoms_feature(labels, model, X, new_msmts, msmts):
    random_coeffs = predict_coeffs(model, msmts, X, True)
    logger.info("random_coeffs = " + str(random_coeffs)) 
    rekf = build_ekf(random_coeffs, [msmts]) 
    raccs, raccuracy = ekf_accuracies(rekf, new_msmts)
    labels.append([raccuracy, to_size(msmts, n_msmt, n_entries), 
                              to_size(random_coeffs, n_entries, n_coeff)])


def add_raws_feature(labels, new_msmts, msmts):
    if n_msmt == n_coeff / 3:
        ident = ones(n_msmt)
    else:
        ident = identity(n_msmt).flatten()
    rawcoeffs = concatenate((ident, ident, ident))
    rawekf = build_ekf(rawcoeffs, [msmts]) 
    rawaccs, rawaccuracy = ekf_accuracies(rawekf, new_msmts)
    labels.append([rawaccuracy, to_size(msmts, n_msmt, n_entries), 
                                to_size(rawcoeffs, n_entries, n_coeff)])



def add_ratio_feature(labels, new_msmts, history):
    (prev, msmts) = history[-2:] if len(history)>1 else ([], [])
    noise = lambda n : list(map(lambda i: uniform(0,1), range(n)))
    prev = prev if len(prev) else msmts
    if n_coeff == n_msmt * 3:
        q = ones(n_msmt) #+ noise(n_msmt)
        f = divide(msmts,array(prev)+1e-9) + 1e-9
        q = array(f)
    else:
        q = identity(n_msmt).flatten() #+ noise(n_msmt)
        f = symmetric(divide(msmts, array(prev)+1e-9) + 1e-9).flatten()
    coeffs = list(map(lambda i: concatenate((q, f, q)), range(n_entries)))
    ekf, _ = build_ekf(coeffs[-1], history)
    ekf.predict()
    accs, acc = ekf_accuracies(ekf, new_msmts)
    logger.info("coeffs,shape = " + str((coeffs[-1], shape(coeffs[-1]))) +
        ", prev,msmts,pred = " + str((prev, msmts, ekf.x_prior.T)) +
        ", accuracy = " + str(acc) + ", new_msmts = " + str(new_msmts))
    labels.append([acc, to_size(msmts, n_msmt, n_entries), 
                        to_size(coeffs, n_entries, n_coeff)])
    return coeffs



def add_covariance_feature(labels, new_msmts, history, coeffs):
    if n_coeff==n_msmt*3:
        fs = [c[n_msmt:-n_msmt] for c in coeffs] + [ones(n_msmt)]
    else:
        index = n_msmt * n_msmt
        fs = [c[index:-index] for c in coeffs] + [eye(n_msmt).flatten()]
    (innovs, history) = get_innovations(history)
    for prev,inno in zip(innovs[-n_covariance:], innovs[-n_covariance:][1:]):
        for f in fs:
            if n_coeff == n_msmt * 3:
                q = diag(cov(array([prev, inno]).T)) + 1e-9
            else:
                q = cov(array([prev, inno]).T).flatten() + 1e-9
            cs = list(map(lambda i: concatenate((q, f, q)), range(n_entries)))
            ekf, _ = build_ekf(cs[-1], history)
            ekf.predict()
            accs, acc = ekf_accuracies(ekf, new_msmts)
            logger.info("coeffs,shape = " + str((cs[-1], shape(cs[-1]))) +
                ", prev,inno,pred = " + str((prev, inno, ekf.x_prior.T)) +
                ", accuracy = " + str(acc) + ", new_msmts = " + str(new_msmts))
            labels.append([acc, to_size(msmts, n_msmt, n_entries), 
                                to_size(cs, n_entries, n_coeff)])


def get_innovations(history):
    history = list(filter(len, history))
    _, preds = build_ekf([], history)
    preds = [array(p).T[0] for p in preds[0]]
    innovs = array(preds) - array(history)
    logger.info("preds,innovs = " + str((preds, innovs)))
    return (innovs, history)


def track_accuracies(ekf, count, filename, label=""):
    global mcount
    (accuracies, nerrs, msmts, mcount) = ([], [], [], 0)
    for i in range(count):
        if len(test_msmt) <= i:
            test_msmt.append(measurements())
        msmts.append(test_msmt[i])
        nerrs.append(ekf_accuracies(ekf, msmts[-1], range(n_msmt), label))
        accuracies.append(nerrs[-1][-1])
        update_ekf(ekf, [msmts[-1]])
        logger.info(label +" accuracy[" + str(i) + "] = " + str(accuracies[-1]))
        if not simulated:
            sleep(1)
    (priors, nerrs) = ([e[0][-1] for e in nerrs], [e[0][1] for e in nerrs])
    avgs = array(list(map(sum, transpose(array(nerrs)))))/len(nerrs)
    logger.info("nerrs = " + str(nerrs) + ", shape = " + str(shape(nerrs)))
    maxs = list(map(max, transpose(abs(array(nerrs)))))
    logger.info("nerrs.avgs = " + str(avgs) + ", maxs = " + str(maxs))
    pickledump(filename, accuracies)
    pickledump(filename.replace(".pickle", "_msmts.pickle"), msmts)
    pickledump(filename.replace(".pickle", "_priors.pickle"), priors)
    return [accuracies, avgs, maxs]



def tuned_accuracy():
    if n_iterations > 1 and not simulated:
        run_async("activity-cmd")
    lstm_model, X, lstm_accuracy = tune_model(n_epochs, bootstrap_labels)
    history = [measurements() for i in range(10)]
    ekf = [build_ekf([], history)[0]]
    for i in range(1):
        msmts = history[-(i+1)*n_entries:-(i*n_entries+1)]
        coeffs = predict_coeffs(lstm_model, msmts, X) 
        logger.info("coeffs = " + str(coeffs[-1]))
        ekf.append(build_ekf(coeffs[-1], history)[0])
    return track_accuracies(ekf, n_iterations,"tuned_accuracies.pickle","Tuned")
    


def raw_accuracy():
    history = [measurements() for i in range(10)]
    ekf, _ = build_ekf([], history)
    return track_accuracies(ekf, n_iterations, "raw_accuracies.pickle", "Raw")


def run_test():
    reset_globals()
    [tuned, tavgs, tmaxs] = tuned_accuracy()
    [raw, ravgs, rmaxs] = raw_accuracy()
    maxs = list(map(max, zip(tmaxs, rmaxs)))
    close_async()
    logger.info("mcount = " + str(mcount))
    logger.info("Tuned EKF accuracy = "+str(tuned)+", mean="+str(1+mean(tuned)))
    logger.info("Raw EKF accuracy = "+str(raw)+", mean = "+str(1+mean(raw))) 
    logger.info("(1-TunedErr/RawErr) = " + str(1-mean(tuned)/mean(raw)))


def run_test_convergence(genid):
    global simulated
    global gid
    simulated = True
    reset_globals()
    for n in [int(genid)] if genid else range(len(generators)):
        gid = n
        run_test()
        os_run("mv tuned_accuracies.pickle tuned_accuracies_"+str(n)+".pickle")
        os_run("mv tuned_accuracies_msmts.pickle tuned_msmts_"+str(n)+".pickle")
        os_run("mv tuned_accuracies_priors.pickle tuned_prio_"+str(n)+".pickle")
        os_run("mv raw_accuracies.pickle raw_accuracies_"+str(n)+".pickle")
        os_run("mv raw_accuracies_msmts.pickle raw_msmts_"+str(n)+".pickle")
        os_run("mv raw_accuracies_priors.pickle raw_priors_"+str(n)+".pickle")
        logger.info("generator done: " + str(n) + " of " + str(len(generators)))
    os_run("tar cvf test_convergence.tar *.pickle")
    os_run("rm *.pickle")
    logger.info("done")


def reset_globals():
    global test_msmt
    global msmts
    global mcount
    test_msmt = []
    msmts = []
    mcount = 1



def run_monitors():
    for host in set(get_config("lqn-hosts")):
        threads.append(threading.Thread(target=create_monitor(host)))
        threads[-1].start()
    logger.info("Monitors created")
    print("\nNow monitoring hosts " + str(set(get_config('lqn-hosts'))) + "\n")



def create_monitor(host):
    def monitor_loop():
        for sample in range(15): 
            monitor_host(host)
            logger.info("Sample " + str(sample) + " of 150 done")
            sleep(1)
        os_run("mv measurements.pickle " + host + "_measurements.pickle")
        os_run("tar rvf pickles_" + host + ".tar " + host + "*.pickle")
        os_run("rm " + host + "*.pickle")
        logger.info("Monitor done on host " + host)
    return monitor_loop



def monitor_host(host):
    if not host in monitor_msmts:
        tune_host(host)
    if active_monitor:
        do_action(update_ekf(ekfs[host], [monitor_msmts[host][-1]])[1], host)
    monitor_msmts[host].append(measurements(host))
    ekf_accuracies(ekfs[host], monitor_msmts[host][-1], None, "", False, host)



def tune_host(host):
    monitor_msmts[host] = []
    # Tune host ekf
    def label_fn(model, X, labels=[], sample_size=10):
        return bootstrap_labels(model, X, labels, sample_size, "boot", host)
    lstm_model, X, _ = tune_model(n_epochs, label_fn)
    history = [measurements(host) for x in range(10)]
    monitor_msmts[host].extend(history)
    coeffs = predict_coeffs(lstm_model, history[-n_entries:], X)
    logger.info("coeffs = " + str(coeffs[-1]))
    ekfs[host] = [build_ekf(coeffs[-1], history), build_ekf([], history)]
    lqn_vals = solve_lqn(0)
    logger.info("msmts = " + str(monitor_msmts[host]))
    m, c = solve_linear(lqn_vals, monitor_msmts[host])
    merge_state({"lqn-ekf-model": {"m": m, "c": float(c)}})
    logger.info("Tuning done for host: " + host)
 

if __name__ == "__main__":
    start = time()
    process_args()
    from config import *
    if "--test" in sys.argv or "-t" in sys.argv:
        run_test()
    elif "--test-convergence" in sys.argv or "-tc" in sys.argv:
        run_test_convergence(next(sys.argv, ["--test-convergence", "-tc"]))
    else:
        run_monitors()
    if "--generate-traffic" in sys.argv or "-g" in sys.argv:
        for endp in ["db-endpoint", "search-endpoint"]:
            sleep(10)
            f = io.StringIO()
            with redirect_stdout(f):
                test_client(endp)
            pickledump("clientout.pickle", f.getvalue())
            os_run("tar rvf pickles_" + endp + ".tar *.pickle")
            os_run("rm *.pickle")
    print("Output in lstm_ekf.log...")
    for t in threads: 
        t.join()
    print("Controller done : " + str(time() - start) + "s")
