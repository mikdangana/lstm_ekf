from config import *
from utils import *
from ekf import *
from lstm import *


def measurements():
    cmd = "top -b -n 2 | grep -v Tasks | grep -v top | grep -v %Cpu | " + \
        "grep -v KiB | grep -v PID | grep [0-9] | " + \
        "awk '{print $1,$3,$4,$5,$6,$7,$9,$10}'"
    pstats = os_run(cmd)
    logger.info(len(pstats.split()))
    pstatsf = list(map(lambda i: float32(i) if i!="rt" else 0, pstats.split()))
    logger.info("type = " + str(type(pstatsf[0])))
    return pstatsf



def predict_coeffs(model, newdata, X):
    logger.info("initialized = " + str(lstm_initialized()))
    if not lstm_initialized():
        return list(map(lambda n: random(), range(n_msmt)))
    input = to_size(newdata, n_msmt, n_entries)
    output = tf_run(model, feed_dict={X:input})
    logger.debug("Coeff output = " + str(output) + ", predict = " + str(output[-1]) + ", input = " + str(shape(newdata)))
    return output[-1]



def get_baseline(model, sample_size, X):
    (baseline, ekf, msmts, history) = ([], None, [], [])
    for i in range(0, sample_size):
        new_msmts = measurements()
        logger.info("i = " + str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            (best_coeffs, best_accuracy) = ([], 0)
            if not ekf:
                coeffs = predict_coeffs(model, msmts, X)
                ekf = build_ekf(coeffs, [msmts]) # history)
            elif len(history):
                logger.info("get_baseline.history = " + str(shape(history)))
                update_ekf(ekf, history)
            baseline.append(ekf_accuracy(ekf, new_msmts))
        msmts = new_msmts
        history.append([msmts])
    return baseline



# Generates training labels by a bootstrap active-learning approach 
def bootstrap_labels(model, X, sample_size = 10, action_interval = 15):
    (labels, ekf, msmts, history, pred_per_sample) = ([], None, [], [], 1)
    ekf_baseline = get_baseline(model, sample_size, X)
    for i in range(0, sample_size):
        new_msmts = measurements()
        logger.info("sample="+str(i) + ", new_msmts = " + str(shape(new_msmts)))
        if len(msmts):
            (best_coeffs, best_accuracy) = ([], 0)
            for j in range(0, pred_per_sample):
                coeffs = predict_coeffs(model, msmts, X)
                logger.info("pre build_ekf.coeffs = " + str(coeffs))
                ekf = build_ekf(coeffs, [msmts]) #history) 
                accuracy = ekf_accuracy(ekf, new_msmts)
                logger.info("sample = " + str(i) + " of " + str(sample_size) + ", pred_per_sample = " + str(j) + " of " + str(pred_per_sample) + ", coeffs = " + str(coeffs) + ", accuracy = " + str(accuracy) + ", best_accuracy = " + str(best_accuracy))
                #if accuracy >= best_accuracy: # TODO max(best_acc, ekf_base[i]):
                best_coeffs = to_size(coeffs, 1, n_coeff)
                 #   best_accuracy = accuracy
            if len(best_coeffs): # Only add labels if accuracy > ekf_baseline
                labels.append([to_size(msmts, n_msmt, n_entries), best_coeffs])
            if i % action_interval == 0:
                do_action(ekf, msmts)
        msmts = new_msmts
        history.append(msmts)
    return labels



def model_accuracy(model, input_tensor):
    msmts = measurements()
    coeffs = predict_coeffs(lstm_model, msmts, input_tensor)
    ekf = build_ekf(coeffs, [msmts])
    return ekf_accuracy(ekf, measurements())
    


if __name__ == "__main__":
    lstm_model, X, lstm_accuracy = tune_model(2, bootstrap_labels)
    logger.info("Tuned model accuracy = " + str(model_accuracy(lstm_model, X)))
