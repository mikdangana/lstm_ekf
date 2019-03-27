from config import *
from utils import *
from ekf import *
from lstm import *
from time import sleep
from numpy import subtract, concatenate, divide
import config


test_msmt = []
dimz = int(n_msmt/2)
msmts = []
mcount = 0
monitor_msmts = {}
ekfs = {}
default_host = None


# Only the first n_msmt values are used, the rest are ignored by LSTM & EKF
def measurements(simulated = True, host = default_host):
    if simulated:
        return sim_measurements()
    cmd = "top -b -n 1 | tail -121 | sort -n | " + \
            " awk '{print $1,$3,$4,$5,$6,$7,$9,$10}'" 
    if host:
        cmd = get_config("login-"+host) + " -t '" + cmd + "'"
    pstats = os_run(cmd).split()
    logger.info(str(len(pstats)) + " measurements retrieved")
    def normalize(v):
        return float32(v[1])/norms[v[0] % 8] if v[1]!="rt" else 0
    pstatsf = list(map(normalize, zip(range(len(pstats)), pstats)))
    pstatsf = pstatsf[0:dimz]
    pstatsf.extend(zeros(dimz))
    logger.info("parsed measurements, size=" + str(len(pstatsf)))
    pickleadd("measurements.pickle", array(pstatsf).flatten())
    return pstatsf


def sim_measurements():
    global mcount
    mcount = mcount + 1
    delta = 1e-3
    prev = msmts[-1] if len(msmts)%1000 else zeros(n_msmt) + delta
    pstatsf = list(map(lambda i:prev[i]*(1+i)/10, range(n_msmt)))
    msmts.append(pstatsf)
    pickleadd("measurements.pickle", array(pstatsf).flatten())
    logger.info("msmts = " + str(pstatsf))
    return pstatsf



def predict_coeffs(model, newdata, X, randomize=False):
    logger.info("LSTM initialized = " + str(lstm_initialized()))
    if not lstm_initialized() or randomize:
        def rand_coeffs(i): 
            return list(map(lambda n: uniform(0.0,1.0), range(n_coeff)))
        return list(map(rand_coeffs, range(n_entries)))
    input = to_size(newdata, n_msmt, n_entries)
    if isinstance(model, list):
        lout = []
        for lstm in model:
            lout.append(tf_run(lstm, feed_dict={X:input}))
    else:
        output = tf_run(model, feed_dict={X:input})
    logger.debug("Coeff output = " + str(shape(output)) + ", predict = " + str(output[-1]) + ", input = " + str(shape(newdata)))
    return output



# Creates a baseline coeff/EKF and tracks its performance over a sample history
def baseline_accuracy(model, sample_size, X):
    (baseline, ekf, msmts, history) = ([], None, [], [])
    for i in range(0, sample_size):
        new_msmts = measurements()
        logger.info("i = " + str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            if not ekf:
                coeffs = predict_coeffs(model, msmts, X)
                ekf = build_ekf(coeffs[-1], [msmts]) # history)
            elif len(history):
                logger.info("baseline_accuracy.history =" + str(shape(history)))
                update_ekf(ekf, history)
            baseline.append(ekf_accuracy(ekf, new_msmts))
        msmts = new_msmts
        history.append([msmts])
    return baseline



# Generates training labels by a bootstrap active-learning approach 
def bootstrap_labels(model, X, labels=[], sample_size=10):
    (labels, msmts, history, coeffs) = ([], [], [], [])
    (accuracies, ekf_baseline) = ([], baseline_accuracy(model, sample_size, X))
    for i in range(0, sample_size):
        new_msmts = measurements()
        history.append(msmts)
        add_labels(labels, accuracies, coeffs, model, X, new_msmts, history, i)
        msmts = new_msmts
    pickleconc("boot_accuracies.pickle", list(map(lambda a:a[-1], accuracies)))
    pickleconc("boot_profiles.pickle", list(map(lambda a:a[0], accuracies)))
    pickleconc("boot_coeffs.pickle", coeffs)
    logger.info("isconverged[25:30] = " + str(list(map(lambda i: isconverged(list(map(lambda r: r[-1][i], coeffs))), range(25, 30)))))
    labels.sort(key = lambda v: v[0], reverse=True)
    logger.info("batch label accuracies = " + str(list(map(lambda l:l[0], labels[0:int(0.1*len(labels))]))))
    logger.info("batch labels = " + str(list(map(lambda l:l[1], labels[0:int(0.1*len(labels))]))))
    return labels[0:int(0.1 * len(labels))]



def add_labels(labels, accuracies, coeffs, model, X, new_msmts, history, step):
    (action_interval, pred_per_sample, msmts) = (40, 10, history[-1])
    if len(msmts):
        for j in range(0, pred_per_sample):
            coeffs.append(predict_coeffs(model, msmts, X))
            ekf = build_ekf(coeffs[-1][-1], [msmts]) 
            accs, accuracy = ekf_accuracies(ekf, new_msmts)
            #add_randoms_feature(labels, model, X, new_msmts, msmts)
            #add_raws_feature(labels, new_msmts, msmts)
            add_ratio_feature(labels, new_msmts, history)
            logger.info("sample = " + str(step) + ", pred_per_sample = " + str(j) + " of " + str(pred_per_sample) + ", coeffs = " + str(coeffs[-1]) + ", accuracy = " + str(accuracy))
            accuracies.append([accs, accuracy])
            labels.append([accuracy, to_size(msmts, n_msmt, n_entries), to_size(coeffs[-1], n_entries, n_coeff)])
        sleep(0.5)
        if step % action_interval == 0:
            do_action(ekf, msmts)



def add_randoms_feature(labels, model, X, new_msmts, msmts):
    random_coeffs = predict_coeffs(model, msmts, X, True)
    logger.info("random_coeffs = " + str(random_coeffs)) 
    rekf = build_ekf(random_coeffs, [msmts]) 
    raccs, raccuracy = ekf_accuracies(rekf, new_msmts)
    labels.append([raccuracy, to_size(msmts, n_msmt, n_entries), to_size(random_coeffs, n_entries, n_coeff)])



def add_raws_feature(labels, new_msmts, msmts):
    if n_msmt == n_coeff / 3:
        ident = ones(n_msmt)
    else:
        ident = identity(n_msmt).flatten()
    rawcoeffs = concatenate((ident, ident, ident))
    rawekf = build_ekf(rawcoeffs, [msmts]) 
    rawaccs, rawaccuracy = ekf_accuracies(rawekf, new_msmts)
    labels.append([rawaccuracy, to_size(msmts, n_msmt, n_entries), to_size(rawcoeffs, n_entries, n_coeff)])



def add_ratio_feature(labels, new_msmts, history):
    (prev, msmts) = history[-2:] if len(history)>1 else ([], [])
    if (len(prev)):
        noise = lambda n : list(map(lambda i: uniform(0,1), range(n)))
        if n_coeff == n_msmt * 3:
            q = ones(n_msmt) + noise(n_msmt)
            f = divide(msmts,array(prev)+1e-9)
        else:
            q = identity(n_msmt).flatten() + noise(n_msmt)
            f = symmetric(divide(msmts, array(prev)+1e-9)).flatten()
            logger.info("f.shape = " + str(shape(f)))
        coeffs = list(map(lambda i: concatenate((q, f, q)), range(n_entries)))
        ekf = build_ekf(coeffs[-1], history)
        accs, acc = ekf_accuracies(ekf, new_msmts)
        logger.info("coeffs = " + str(coeffs[-1]) + ", coeffs.size = " + str(shape(coeffs[-1])) + ", prev = " + str(prev) + ", msmts = " + str(msmts) + ", accuracy = " + str(acc) + ", new_msmts = " + str(new_msmts))
        labels.append([acc, to_size(msmts, n_msmt, n_entries), to_size(coeffs, n_entries, n_coeff)])



def track_accuracies(ekf, count, filename, label=""):
    accuracies = []
    nerrs = []
    for i in range(count):
        if len(test_msmt) <= i:
            test_msmt.append(measurements())
        msmts = test_msmt[i]
        nerrs.append(ekf_accuracies(ekf, msmts, range(dimz), label))
        accuracies.append(nerrs[-1][-1])
        update_ekf(ekf, [msmts])
        logger.info(label+" accuracy[" + str(i) + "] = " + str(accuracies[-1]))
        sleep(1)
    nerrs = list(map(lambda e: e[0][1], nerrs))
    avgs = array(list(map(sum, transpose(array(nerrs)))))/len(nerrs)
    maxs = list(map(max, transpose(abs(array(nerrs)))))
    logger.info("nerrs.avgs = " + str(avgs) + ", maxs = " + str(maxs))
    pickledump(filename, accuracies)
    return [accuracies, avgs, maxs]



def monitor():
    for i in range(10):
        for host in set(get_config("lqn-hosts")):
            monitor_host(host)
        sleep(1)
    logger.info("Done")



def monitor_host(host):
    if not host in monitor_msmts:
        monitor_msmts[host] = []
        # Tune host ekf
        default_host = host
        lstm_model, X, lstm_accuracy = tune_model(n_epochs, bootstrap_labels)
        os_run("tar cvf tune_pickles_" + host + ".tar *.pickle")
        os_run("rm *.pickle")
        history = list(map(lambda x : measurements(), range(10)))
        monitor_msmts[host].extend(history)
        coeffs = predict_coeffs(lstm_model, history[-n_entries:], X)
        logger.info("coeffs = " + str(coeffs[-1]))
        ekfs[host] = [build_ekf(coeffs[-1], history), build_ekf([], history)]
        logger.info("Tuning done for host: " + host)

    monitor_msmts[host].append(measurements(True, host))
    do_action(ekf_predict(ekfs[host]), monitor_msmts[host][-1], host)
    ekf_accuracies(ekfs[host], monitor_msmts[host][-1], None, "", False, host)
    #    nerrs.append(ekf_accuracies(ekf, msmts, range(dimz), label))
    #    accuracies.append(nerrs[-1][-1])
    #    update_ekf(ekf, [msmts])



def tuned_accuracy():
    if n_iterations > 1:
        run_async("activity-cmd")
    lstm_model, X, lstm_accuracy = tune_model(n_epochs, bootstrap_labels)
    history = list(map(lambda x : measurements(), range(10)))
    coeffs = predict_coeffs(lstm_model, history[-n_entries:], X)
    logger.info("coeffs = " + str(coeffs[-1]))
    ekf = [build_ekf(coeffs[-1], history), build_ekf([], history)]
    return track_accuracies(ekf, n_iterations,"tuned_accuracies.pickle","Tuned")
    


def raw_accuracy():
    history = list(map(lambda x : measurements(), range(10)))
    ekf = build_ekf([], history)
    return track_accuracies(ekf, n_iterations, "raw_accuracies.pickle", "Raw")

    

if __name__ == "__main__":
    process_args()
    from config import *
    [tuned, tavgs, tmaxs] = tuned_accuracy()
    [raw, ravgs, rmaxs] = raw_accuracy()
    maxs = list(map(max, zip(tmaxs, rmaxs)))
    tavgs = divide(tavgs, maxs)
    ravgs = divide(ravgs, maxs)
    close_async()
    logger.info("mcount = " + str(mcount))
    logger.info("Tuned EKF accuracy = "+str(tuned)+", mean="+str(1+mean(tuned)))
    logger.info("Raw EKF accuracy = "+str(raw)+", mean = "+str(1+mean(raw))) 
    logger.info("(1-TunedErr/RawErr) = " + str(1-mean(tuned)/mean(raw)))
    print("Output in lstm_ekf.log")
