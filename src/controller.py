from config import *
from utils import *
from ekf import *
from lstm import *
from client import *
from time import sleep, time
from numpy import subtract, concatenate, divide, nan_to_num, array_equal
import config
import threading, sys
import io, importlib
from contextlib import redirect_stdout
from statistics import mode


test_msmt = []
dimz = int(n_msmt/2)
msmts = []
mcount = 1
threads = []
monitor_msmts = {}
lstms = {}
ekfs = {}
generators = get_generators()
gid = len(generators)
has_traffic = False
simulated = False


def lstm():
    tname = threading.currentThread().getName()
    if not tname in lstms:
        lstms[tname] = Lstm()
    return lstms[tname]


# Only the first n_msmt values are used, the rest are ignored by LSTM & EKF
def measurements(host = ""):
    if simulated:
        return sim_measurements(host)
    cmd = get_config("memory-cmd-" + host)
    cmd = "top -b -n 1 " if not cmd else cmd
    cmd = cmd + " | grep \"^ *[0-9]\""
    if host and len(host):
        cmd = get_config("login-"+host) + " -o LogLevel=QUIET -t '" + cmd + "'"
    cmd = cmd + " | awk '{print $5,$5,$5,$5,$6,$7,$9,$10}' | sort -k 7 -nr"
    pstats = os_run(cmd)
    pstats = pstats.split() if pstats else ""
    logger.info(str(len(pstats)) + " measurements retrieved")
    pstatsf = list(map(normalize, zip(range(len(pstats)), pstats)))
    pstatsf = pstatsf[0:n_msmt]
    #pstatsf = pstatsf[0:dimz] + list(zeros(dimz))
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
    logger.info("LSTM initialized = " + str(lstm().lstm_initialized()))
    if not lstm().lstm_initialized() or randomize:
        def rand_coeffs(i): 
            return list(map(lambda n: uniform(0.0,1.0), range(n_coeff)))
        return list(map(rand_coeffs, range(n_entries)))
    input = to_size(newdata, n_msmt, n_entries)
    if model == None:
        return [[]]
    elif isinstance(model, list):
        lout = []
        for mdl in model:
            lout.append(lstm().tf_run_reset(mdl, feed_dict={X:input}))
    else:
        output=vote([lstm().tf_run_reset(model,feed_dict={X:input}) for i in range(1)])
    logger.debug("Coeff output = " + str(shape(output)) + ", predict = " + 
        str(output[-1]) + ", input = " + str(shape(newdata)))
    return output


def vote(outputs):
    lasts = [out[-1][0] for out in outputs]
    logger.debug("lasts = " + str(lasts))
    try:
        i = lasts.index(mode(lasts))
    except:
        i = 0;
    return outputs[i]


# Creates a baseline coeff/EKF and tracks its performance over a sample history
def baseline_accuracy(model, sample_size, X, host, sim_msmts):
    (baseline, ekf, msmts, history) = ([], None, [], [])
    for i in range(0, sample_size):
        new_msmts = sim_msmts[i] if len(sim_msmts) else measurements(host)
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


def bootstrap_fn(sim_msmts, samples=10):
    def bootstrap(model, X, labels=[], sample_size=samples,pre="boot", host=""):
        return bootstrap_labels(model,X,labels,sample_size,pre,host,sim_msmts)
    return bootstrap


# Generates training labels by a bootstrap active-learning approach 
def bootstrap_labels(model, X, labels=[], sample_size=10, pre="boot", host="",
    sim_msmts = []):
    (msmts, history, coeffs) = ([], [], [])
    (accuracies, ekf_baseline) = ([],  
        baseline_accuracy(model, sample_size, X, host, sim_msmts))
    for i in range(sample_size):
        new_msmts = sim_msmts[i] if len(sim_msmts) else measurements(host)
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
                #add_linearmodel_feature(labels, new_msmts, history, ratios)
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



def add_linearmodel_feature(labels, new_msmts, history, coeffs):
    if n_coeff==n_msmt*3:
        qs = [c[0:n_msmt] for c in coeffs] + [ones(n_msmt)]
    else:
        index = n_msmt * n_msmt
        qs = [c[0:index] for c in coeffs] + [eye(n_msmt).flatten()]
    history = list(filter(len, history))
    for prev,msmt in zip(history, history[1:] + [new_msmts]):
        for q in qs:
            logger.info("prev,noise = " + str((prev, white_noise(q))))
            f = lstsq(array(prev).T, array(msmt - white_noise(q)).T)
            f = diag(f) if n_coeff == n_msmt * 3 else array(f).flatten()
            cs = list(map(lambda i: concatenate((q, f, q)), range(n_entries)))
            ekf, _ = build_ekf(cs[-1], history)
            ekf.predict()
            accs, acc = ekf_accuracies(ekf, new_msmts)
            logger.info("coeffs,shape = " + str((cs[-1], shape(cs[-1]))) +
                ", prev,msmt = " + str((prev, msmt)) +
                ", accuracy = " + str(acc) + ", new_msmts = " + str(new_msmts))
            labels.append([acc, to_size(msmts, n_msmt, n_entries), 
                                to_size(cs, n_entries, n_coeff)])


def white_noise(q):
    n = symmetric(q) if n_coeff==n_msmt*3 else array(q).resize(n_msmt, n_msmt)
    return multivariate_normal(zeros(n_msmt), n)


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
    lstm_model, X, lstm_accuracy = lstm().tune_model(n_epochs, bootstrap_labels)
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


def generate_traffic():
    global has_traffic
    has_traffic = True
    for endp in ["db-endpoint", "search-endpoint"]:
        #sleep(10)
        f = io.StringIO()
        with redirect_stdout(f):
            #os_run(re.sub(r'<endpoint>', get_config(endp), get_config("traffic-cmd"))) #test_client(endp)
            test_client(endp)
        pickledump("clientout.pickle", f.getvalue())
        os_run("tar rvf pickles_" + endp + ".tar *.pickle")
        os_run("rm *.pickle")
    has_traffic = False


def toLqn(v):
    for i in range(4):
        v[i] = round(v[i])
    return [{'nUsers': v[0], 'nWebServers': v[1], 'nDb': v[2], 'nSearch': v[3]}]


def model_track_labels(rowid, ybase = None, ydelta = 0.5):
    rng = get_config('lqn-range')
    ybase = ybase if ybase else [int(random()*0.5*i) for i in rng]
    ybase.extend(zeros(n_lstm_out-len(ybase)))
    y = zeros([n_entries, n_lstm_out]) + to_size(ybase, n_lstm_out)
    rows = to_size(solve_lqn_input(toLqn(ybase))[rowid], n_msmt)
    logger.debug("nUsers,y,rows = " + str((ybase, y, rows)))
    def create_labels(model, X, labels=[], sample_size=10):
        for i in range(sample_size):
            labels.append([0.0, rows + random()*ydelta, y])
        return labels
    return create_labels


def model_tracking_data(noise = array(get_config('lqn-noise-factor'))):
    rows = solve_lqn_input([
        {'nUsers':[10,31],'nWebServers':[1],'nDb':[1],'nSearch':[1]}])
    raw = [rows[0] for i in range(1200)] 
    raw.extend([rows[1] for i in range(800)])
    data = [r + random()*noise for r in raw] 
    return (data, raw, data[0:int(len(data)*0.1)])


def createHj(model, X, xinit):
    dx = 0.001 * array([ones(len(xinit))]).T
    p = array([ones(len(xinit))]).T * (18 if not model else 16)
    [p,p1] = [p,p+dx] #[predict_coeffs(model,yinit+i*dx,X)[-1] for i in [0,len(xinit)-1]]
    def hjacobian(x):
        (mx, mp) = (x[0], mean(p))
        logger.info("hjacobian.predicting coeffs x,dx,mx,mp="+str((x,dx,mx,mp)))
        (x0, p0) = (x,p) if model and mx>mp or not model and mx<mp else (0,0)
        dxs = array([x0 + dx*i for i in range(len(p))]).T
        dps = array([p0 + (p1 - p)/(len(x)*(i+1)) for i in range(len(x))]).T
        Hj = nan_to_num((dps + 1e-9) / (dxs + 1e-9))[0]
        logger.info("hjacobian.dxs,dps,Hj = " + str((shape(dxs),shape(dps),Hj)))
        return Hj
    return hjacobian


    p = array([zeros(n_msmt)]).T #array([predict_coeffs(model,y,X)[-1]]).T
    logger.info("p,y = " + str((p,y)))
    def H(x):
        return x + 10 #Hlqn(x)
    return None #H


def createHx(model, X, y):
    p = array([zeros(n_msmt)]).T #array([predict_coeffs(model,y,X)[-1]]).T
    logger.info("p,y = " + str((p,y)))
    def H(x):
        return x + 10 #Hlqn(x)
    return None #H


def Hlqn(x):
    rows = solve_lqn_input(toLqn(list(x.T[0])))
    logger.debug("x,rows[0] = " + str((x, array([rows[0]]).T)))
    return array([rows[0]]).T


# Tracked (LQN) model output y.shape = [n_msmt], input p.shape = [n_lstm_out]
# y is the (LQN) model output, p is the (LQN) model input (p -> y)
# LSTM model inputs/outputs are the reverse of the (LQN) model (y -> p)
def run_model_tracking_tests(y = None):
    (msmts,raw, tmsmts) = model_tracking_data()
    avg = array([mean(c) for c in array(tmsmts).T])
    h_model,X,_ = (None, None, None) #tune_model(n_epochs,model_track_labels(0),tf_lqn_cost)
    (Hj, Hx, ys) = (createHj(h_model, X, avg), createHx(h_model, X, avg), [])
    ((ekf, _), basic_ekf) = (build_lstm_ekf(tmsmts), build_ekf([], tmsmts))
    pickleconc("modeltrack_msmts_denoise.pickle", raw[len(tmsmts):])
    for i,msmt in zip(range(int(len(msmts))), msmts[len(tmsmts):]):
        ys.append(msmt)
        ekf, priors = update_ekf(ekf, ys[-1:], ekf.R, None, Hj, Hx)
        basic_ekf, basic_priors = update_ekf(basic_ekf, ys[-1:],None,None, None)
        logger.info("updated ekf, i,ys,priors = " + str((i,ys[-1],priors)))
        error =sum(list(map(lambda x: abs(x[0][0].T-x[1]),zip(priors,ys[-1:]))))
        wavg = array([mean(c) for c in array(ys[-50:]).T])
        pickleadd("modeltrack_msmts.pickle", msmt)
        pickleadd("modeltrack_priors.pickle", priors[0][0][0])
        pickleadd("modeltrack_basic_priors.pickle", basic_priors[0][0][0])
        pickleadd("modeltrack_errors.pickle", error)
        pickleadd("modeltrack_avg.pickle", wavg)
        logger.info("i,avg,wavg,learn_threshold,diff=" +str((i,mean(avg),mean(wavg),learn_threshold,abs(mean(avg)-mean(wavg)))))
        if len(msmts)-len(tmsmts)-1000 == i: #abs(mean(wavg) - mean(avg)) > learn_threshold:
            avg = wavg
            logger.info("moving avg changed, learning new model parameters")
            #h_model,X,_ =tune_model(n_epochs,model_track_labels(1),tf_lqn_cost)
            h_model = 1
            Hj = createHj(h_model, X, avg)
    logger.info("done")


def configure_model_tracking():
    global n_msmt
    global dimx
    global n_entries
    global n_lstm_out
    n_msmt = 7 #8 * n_proc # Kalman z
    n_entries = 1
    n_lstm_out = 7 #n_coeff


def build_lstm_ekf(msmts):
    (cost, nout) = (None, n_coeff)
    #qf_model,X,lstm_accuracy = tune_model(1, bootstrap_fn(msmts,5), cost, nout)
    coeffs = [[]] #predict_coeffs(qf_model, msmts, X) 
    return build_ekf(coeffs[-1], msmts)



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
    global has_traffic
    has_traffic = True
    def monitor_loop():
        sample = 0
        logger.info("n_samples = " + str(n_samples))
        while has_traffic or sample < n_samples: 
            monitor_host(host)
            sample = sample + 1
            logger.info("Sample " + str(sample) + " done")
            sleep(1)
        os_run("mv measurements.pickle " + host + "_measurements.pickle")
        os_run("tar rvf pickles_" + host + ".tar " + host + "*.pickle")
        os_run("rm " + host + "*.pickle")
        logger.info("Monitor done on host " + host + ": pickles_"+host+".tar")
    return monitor_loop



def monitor_host(host):
    if not host in monitor_msmts:
        tune_host(host)
    if active_monitor:
        if predictive:
            priors = update_ekf(ekfs[host], monitor_msmts[host])[1]
            msmts = monitor_msmts[host][1:] + [list(priors[-1][-1].T[0])]
        else:
            msmts = monitor_msmts[host]
        if len(msmts):
            do_action([[array([m]).T] for m in msmts], host)
    monitor_msmts[host].append(measurements(host))
    if predictive:
        ekf_accuracies(ekfs[host],monitor_msmts[host][-1],None,"",False,host)


# NB: Recommended to run using the following script:
# rm *.pickle; mkdir pca; for i in {1..10}; do python src/controller.py --test-pca-kf; for x in `ls *.pickle`; do mv $x pca/${x//.pickle/_$i.pickle}; done; done

def run_test_pca_kf():
    (host, history) = ("", [])
    ekfs[host] = [build_ekf([], history)]
    def predfn(msmts, lqn_ps = None): 
        priors = update_ekf(ekfs[host], msmts)[1]
        return to_size(priors[-1], msmts.shape[1], msmts.shape[0])
    test_pca(100, predfn)



def run_test_nonpca_kf():
    global n_msmt
    global dimx
    (dimx, n_msmt, host, history) = (3, 24 * 10, "", [])
    ekfs[host] = [build_ekf([], history, None, n_msmt, dimx)]
    def predfn(msmts, lqn_ps): 
        msmts = array([to_size(m, n_msmt, 1) for m in msmts])
        lqn_ps = array([to_size(p, dimx, 1) for p in lqn_ps])
        z_ave = array([sum(c)/4 for c in array(msmts[-5:-1]).T]).T
        x_ave = to_size(lqn_ps[-2:], 1, dimx)
        H = getH(z_ave, x_ave)
        def Hj(x):
            (x1,x2) = (x-ones([dimx,1]), x+ones([dimx,1]) if len(lqn_ps)<2 else x_ave)
            z_ave = array([sum(c)/5 for c in array(msmts[-5:]).T]).T
            (H1, H2) = (H, getH(z_ave, x2)) 
            (x1, x2) = ([x1 for i in range(n_msmt)],[x2 for i in range(n_msmt)])
            hj = (H2 - H1)/(to_size(x2,dimx,n_msmt) - to_size(x1,dimx,n_msmt))
            return hj 
        Hx = lambda x: dot(H, x)
        priors = update_ekf(ekfs[host], msmts[-1:], None, None, Hj, Hx)[1]
        return to_size(priors[-1][-1], 1, dimx) if len(priors[-1]) else zeros([dimx,1])
    test_pca(100, predfn, False)


def scale(vs):
    (a,b,c) = [p[0][0] for p in vs[-3:]] if len(vs) > 2 else (1,2,3)
    print("a, b, c = "  +str((a, b, c)))
    dv = (c - b) / (1 if b==a else b - a) * (c - b) 
    return dv


def eq(v, n):
    if isinstance(v, list) or type(v) == type(array([])):
        return reduce(lambda a,b: eq(a,n) and eq(b,n), v)
    return v == n


def rnd(sy, sx, n=1e-3):
    return array([[random()*n for x in range(sx)] for y in range(sy)])
     

def cap(data, limit):
    if isinstance(data, list):
        return [cap(d, limit) for d in data]
    elif type(data) == type(array([])):
        return array([cap(d, limit) for d in data])
    elif data < 0:
        return data if data > -limit else -limit
    return data if data < limit else limit


def getH(z, x):
    x = to_size(x,1,dimx)
    z = to_size(z,1,n_msmt)
    H = lstsq(x.T, z.T)[0].T
    #print("getH.z,dot,x = " + str((H.shape, z[0:5].T, dot(H, x)[0:5].T, x)))
    return H 


def tune_host(host):
    monitor_msmts[host] = []
    if not predictive:
        return
    # Tune host ekf
    def label_fn(model, X, labels=[], sample_size=10):
        return bootstrap_labels(model, X, labels, sample_size, "boot", host)
    lstm_model, X, _ = lstm().tune_model(n_epochs, label_fn)
    history = [measurements(host) for x in range(10)]
    monitor_msmts[host].extend(history)
    coeffs = predict_coeffs(lstm_model, history[-n_entries:], X)
    logger.info("coeffs = " + str(coeffs[-1]))
    ekfs[host] = [build_ekf(coeffs[-1], history), build_ekf([], history)]
    logger.info("Tuning done for host: " + host)
 

if __name__ == "__main__":
    start = time()
    if "--test" in sys.argv or "-t" in sys.argv:
        run_test()
    elif "--test-convergence" in sys.argv or "-tc" in sys.argv:
        run_test_convergence(next(sys.argv, ["--test-convergence", "-tc"]))
    elif "--track-model" in sys.argv or "-tm" in sys.argv:
        run_model_tracking_tests()
    elif "--test-clean" in sys.argv:
        for j in range(len(get_config("lqn-deprovision-actions"))):
            run_actions(get_config("lqn-deprovision-actions"), j)
    elif "--test-pca-kf" in sys.argv:
        run_test_pca_kf()
        sys.exit()
    elif "--test-nonpca-kf" in sys.argv:
        run_test_nonpca_kf()
        sys.exit()
    else:
        run_monitors()
    if "--generate-traffic" in sys.argv or "-g" in sys.argv:
        generate_traffic()
    print("Output in lstm_ekf.log...")
    for t in threads: 
        t.join()
    logger.info("Controller done : " + str(time() - start) + "s")
    sys.exit()


