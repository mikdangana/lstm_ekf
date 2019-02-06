from config import *
from utils import *
from ekf import *
from lstm import *
from time import sleep


def measurements():
    cmd = "top -b -n 2 | grep -v Tasks | grep -v top | grep -v %Cpu | " + \
        "grep -v KiB | grep -v PID | grep [0-9] | " + \
        "awk '{print $1,$3,$4,$5,$6,$7,$9,$10}'"
    pstats = os_run(cmd)
    logger.info(str(len(pstats.split())) + " measurements retrieved")
    pstatsf = list(map(lambda i: float32(i) if i!="rt" else 0, pstats.split()))
    logger.info("parsed measurements type = " + str(type(pstatsf[0])))
    pickleadd("measurements.pickle", array(pstatsf).flatten())
    return pstatsf



def predict_coeffs(model, newdata, X, randomize=False):
    logger.info("LSTM initialized = " + str(lstm_initialized()))
    if not lstm_initialized() or randomize:
        return list(map(lambda n: uniform(0,10), range(n_msmt)))
    input = to_size(newdata, n_msmt, n_entries)
    output = tf_run(model, feed_dict={X:input})
    logger.debug("Coeff output = " + str(output) + ", predict = " + str(output[-1]) + ", input = " + str(shape(newdata)))
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
                ekf = build_ekf(coeffs, [msmts]) # history)
            elif len(history):
                logger.info("baseline_accuracy.history = " + str(shape(history)))
                update_ekf(ekf, history)
            baseline.append(ekf_accuracy(ekf, new_msmts))
        msmts = new_msmts
        history.append([msmts])
    return baseline



# Generates training labels by a bootstrap active-learning approach 
def bootstrap_labels(model, X, labels=[], sample_size=10, action_interval=40):
    (labels, msmts, history, pred_per_sample) = ([], [], [], 10)
    (accuracies, ekf_baseline) = ([], baseline_accuracy(model, sample_size, X))
    for i in range(0, sample_size):
        new_msmts = measurements()
        logger.info("sample="+str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            (best_coeffs, best_accuracy) = ([], 0)
            for j in range(0, pred_per_sample):
                coeffs = predict_coeffs(model, msmts, X)
                random_coeffs = predict_coeffs(model, msmts, X, True)
                logger.info("pre build_ekf.coeffs = " + str(coeffs))
                ekf = build_ekf(coeffs, [msmts]) 
                rekf = build_ekf(random_coeffs, [msmts]) 
                accs, accuracy = ekf_accuracies(ekf, new_msmts)
                raccs, raccuracy = ekf_accuracies(rekf, new_msmts)
                logger.info("sample = " + str(i) + " of " + str(sample_size) + ", pred_per_sample = " + str(j) + " of " + str(pred_per_sample) + ", coeffs = " + str(coeffs) + ", accuracy = " + str(accuracy) + ", best_accuracy = " + str(best_accuracy))
                accuracies.append([accs, best_accuracy])
                labels.append([accuracy, to_size(msmts, n_msmt, n_entries), to_size(coeffs, 1, n_coeff)])
                labels.append([raccuracy, to_size(msmts, n_msmt, n_entries), to_size(random_coeffs, 1, n_coeff)])
            sleep(0.5)
            if i % action_interval == 0:
                do_action(ekf, msmts)
        msmts = new_msmts
        history.append(msmts)
    pickleconc("boot_accuracies.pickle", list(map(lambda a:a[-1], accuracies)))
    pickleconc("boot_profiles.pickle", list(map(lambda a:a[0], accuracies)))
    labels.sort(key = lambda v: v[0], reverse=True)
    logger.info("batch labels = " + str(list(map(lambda l:l[0], labels[0:int(0.1*len(labels))]))))
    return labels[0:int(0.1 * len(labels))]


def track_accuracies(ekf, count, filename):
    accuracies = []
    for i in range(count):
        msmts = measurements()
        accuracies.append(ekf_accuracy(ekf, msmts))
        update_ekf(ekf, [msmts])
        sleep(1)
    pickledump(filename, accuracies)
    return accuracies


def tuned_accuracy():
    lstm_model, X, lstm_accuracy = tune_model(10, bootstrap_labels)
    history = list(map(lambda x : measurements(), range(10)))
    coeffs = predict_coeffs(lstm_model, history[-1], X)
    ekf = build_ekf(coeffs, history)
    return track_accuracies(ekf, 20, "tuned_accuracies.pickle")
    

def raw_accuracy():
    history = list(map(lambda x : measurements(), range(10)))
    ekf = build_ekf(identity(int(math.sqrt(n_coeff))), history)
    return track_accuracies(ekf, 20, "raw_accuracies.pickle")
    

if __name__ == "__main__":
    tuned = tuned_accuracy()
    raw = raw_accuracy()
    logger.info("Tuned EKF accuracy = " + str(tuned))
    logger.info("Raw EKF accuracy = " + str(raw))
    print("Output in lstm_ekf.log")
