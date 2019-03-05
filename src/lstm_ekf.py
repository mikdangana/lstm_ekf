from config import *
from utils import *
from ekf import *
from lstm import *
from time import sleep
from numpy import subtract

cached_msmts = []
cache_ptr = -1
dimz = int(n_msmt/2)
norms = [1000, 20, 20, 1000000000, 1000000000, 10000, 1, 1]


# Only the first n_msmt values are used, the rest are ignored by LSTM & EKF
def measurements(online = True):
    global cached_msmts, cache_ptr
    if not online:
        cache_ptr = cache_ptr + 1
        return cached_msmts[cache_ptr]
    cmd = "top -b -n 1 | grep -v Tasks | grep -v top | grep -v %Cpu | " + \
        "grep -v KiB | grep -v PID | grep [0-9] | " + \
        "awk '{print $1,$3,$4,$5,$6,$7,$9,$10}'"
        #"awk '{if ($1 != " + str(os.getpid()) + ") { print } }' | " + \
    pstats = os_run(cmd).split()
    logger.info(str(len(pstats)) + " measurements retrieved")
    def normalize(v):
        return float32(v[1])/norms[v[0] % 8] if v[1]!="rt" else 0
    pstatsf = list(map(normalize, zip(range(len(pstats)), pstats)))
    pstatsf = pstatsf[0:dimz]
    if len(cached_msmts):
        pstatsf.extend(subtract(pstatsf, cached_msmts[-1][0:dimz]))
    else:
        pstatsf.extend(zeros(dimz))
    logger.info("parsed measurements, size=" + str(len(pstatsf)))
    pickleadd("measurements.pickle", array(pstatsf).flatten())
    cached_msmts.append(pstatsf)
    return pstatsf


def build_msmt_cache():
    for i in range(1000):
        measurements(True)
        sleep(0.5)
    logger.info("cache build done")



def predict_coeffs(model, newdata, X, randomize=False):
    logger.info("LSTM initialized = " + str(lstm_initialized()))
    if not lstm_initialized() or randomize:
        def rand_coeffs(i): 
            return list(map(lambda n: uniform(0.0,1.0), range(n_coeff)))
        return list(map(rand_coeffs, range(n_entries)))
    input = to_size(newdata, n_msmt, n_entries)
    output = tf_run(model, feed_dict={X:input})
    logger.debug("Coeff output = " + str(shape(output)) + ", predict = " + str(output[-1]) + ", input = " + str(shape(newdata)))
    return output[-1]



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
def bootstrap_labels(model, X, labels=[], sample_size=10, action_interval=40):
    (labels, msmts, history, pred_per_sample, coeffs) = ([], [], [], 10, [])
    (accuracies, ekf_baseline) = ([], baseline_accuracy(model, sample_size, X))
    for i in range(0, sample_size):
        new_msmts = measurements()
        if len(msmts):
            for j in range(0, pred_per_sample):
                coeffs.append(predict_coeffs(model, msmts, X))
                #random_coeffs = predict_coeffs(model, msmts, X, True)
                logger.info("pre build_ekf.coeffs = " + str(coeffs[-1]))
                ekf = build_ekf(coeffs[-1][-1], [msmts]) 
                #rekf = build_ekf(random_coeffs, [msmts]) 
                accs, accuracy = ekf_accuracies(ekf, new_msmts)
                #raccs, raccuracy = ekf_accuracies(rekf, new_msmts)
                logger.info("sample = " + str(i) + " of " + str(sample_size) + ", pred_per_sample = " + str(j) + " of " + str(pred_per_sample) + ", coeffs = " + str(coeffs[-1]) + ", accuracy = " + str(accuracy))
                accuracies.append([accs, accuracy])
                labels.append([accuracy, to_size(msmts, n_msmt, n_entries), to_size(coeffs[-1], n_entries, n_coeff)])
                #labels.append([raccuracy, to_size(msmts, n_msmt, n_entries), to_size(random_coeffs, 1, n_coeff)])
            sleep(0.5)
            if i % action_interval == 0:
                do_action(ekf, msmts)
        msmts = new_msmts
        history.append(msmts)
    pickleconc("boot_accuracies.pickle", list(map(lambda a:a[-1], accuracies)))
    pickleconc("boot_profiles.pickle", list(map(lambda a:a[0], accuracies)))
    pickleconc("boot_coeffs.pickle", coeffs)
    labels.sort(key = lambda v: v[0], reverse=True)
    logger.info("batch label accuracies = " + str(list(map(lambda l:l[0], labels[0:int(0.1*len(labels))]))))
    return labels[0:int(0.1 * len(labels))]


def track_accuracies(ekf, count, filename, label=""):
    accuracies = []
    for i in range(count):
        msmts = measurements()
        accuracies.append(ekf_accuracy(ekf, msmts, range(dimz), label))
        update_ekf(ekf, [msmts])
        sleep(1)
    pickledump(filename, accuracies)
    return accuracies


def tuned_accuracy():
    epochs = 10
    lstm_model, X, lstm_accuracy = tune_model(epochs, bootstrap_labels)
    cont = input("Start activity script and press any key to continue.\n")
    #run_async("activity-cmd")
    history = list(map(lambda x : measurements(), range(10)))
    coeffs = predict_coeffs(lstm_model, history[-1], X)
    ekf = build_ekf(coeffs[-1], history)
    return track_accuracies(ekf, 20, "tuned_accuracies.pickle", "Tuned")
    

def raw_accuracy():
    history = list(map(lambda x : measurements(), range(10)))
    ekf = build_ekf(None, history)
    return track_accuracies(ekf, 20, "raw_accuracies.pickle", "Raw")
    

if __name__ == "__main__":
    #build_msmt_cache()
    tuned = tuned_accuracy()
    raw = raw_accuracy()
    logger.info("Tuned EKF accuracy = " + str(tuned))
    logger.info("Raw EKF accuracy = " + str(raw))
    close_asyncs()
    print("Output in lstm_ekf.log")
