from pylab import *
from numpy import array, gradient, mean, std
from scipy import stats
from math import log
from getopt import getopt, GetoptError
from pandas import DataFrame
from surface3d import plotsurface
import pickle, re
import pylab


def load(n):
    n = n if n else "0"

    fname = 'predictions' + n + '.pickle' if n.isdigit() else n

    with open(fname, 'rb') as f:
        return (pickle.load(f), int(n) if n.isdigit() else -1)

    return ([], -1)


def loadfiles(filenames):
    data = []
    for fname in filenames:
        with open(fname, 'rb') as f:
            data.append(pickle.load(f))
    return data


def parse_string(s):
    rows = filter(lambda l: "fetch" not in l and len(l), s.split("\n"))
    rows = [re.compile(r'\W+').split(r) for r in rows]
    rows = [[r[0], r[1] + "." + r[2]] for r in rows]
    return [[float(c.replace("s","")) for c in r] for r in rows[1:]]


def plotpredictions(args):
    predictions, metric = load(args[0] if len(args) else "")
    x = list(map(lambda p: p[0][metric], predictions))
    #y = list(map(lambda p: sum(p[1].T[0])/len(p[1].T[0]), predictions))
    y = list(map(lambda p:p[1].T[0][0], predictions))

    fig, host = subplots()
    sub1 = host.twinx()

    xline, = host.plot(x)
    yline, = sub1.plot(y, "r-", label="x_prior")

    sub1.set_ylabel(yline.get_label())
    sub1.yaxis.label.set_color(yline.get_color())
    sub1.spines["right"].set_visible(True)

    lgd = ['step(x,50)', 'exp(x)', 'sin(x)', 'erf(x)']

    #legend((xline, yline), (lgd[metric], 'EKF.x_prior'))
    legend((lgd[metric if metric<len(lgd) else -1], 'EKF.x_prior'))
    title('EKF Prior Prediction vs ' + lgd[metric if metric<len(lgd) else -1])
    show()


def formatline(data, isGradient, indices, intervals, normalize):
    res = []
    for n in range(len(data)):
        d = formatlines(data, isGradient, indices,intervals,normalize,n)
        res = res + d if len(indices) else res + [d]
    if '--mean' in sys.argv:
        res.append([mean(data[0][0:i+1]) for i in range(len(data[0]))])
    return res


def formatlines(data, isGradient, indices, intervals, normalize, n):
    print("formatlines.n,shape(data) = " + str((n, len(shape(data[n])))))
    if len(indices):
        if isGradient:
            data=[[gradient(array([r[int(i)] for r in data[n]])) for i in indices]]
        elif len(shape(data[n])) > 1:
            data[n] = list(filter(lambda r: len(shape(r)), data[n]))
            data[n] = [[r[int(i)] for r in data[n]] for i in indices]
            print("formatline.data[n] = " +str(data[n][0:min(len(data[n]),10)]))
    elif (len(data[n]) and len(shape(data[n])) > 1): 
        data[n] = [r[-1] for r in data[n]]

    if len(intervals):
        (start, end) = intervals[0].split(":")
        data[n] = data[n][int(start):int(end)]

    if normalize:
        (maxd, mind) = (max([max(d) for d in data]),min([min(d) for d in data]))
        data[n] = (array(data[n]) - mind) / (maxd - mind)
    return data[n]



def parse_line_args(args):
    for k in ["--xaxis", "--yaxis", "--title"]:
        if k in args:
            i = args.index(k)
            args[i+1] = "--" + args[i+1]
    isGradient = "--gradient" in args or "-g" in args
    normalize = "--normalize" in args or "-n" in args
    intervals = list(filter(lambda f: ":" in f, args))
    indices = list(filter(lambda f : f.isdigit(), args))
    filenames = list(filter(lambda f : not f.isdigit() and not f.startswith("-") and not ":" in f, args))
    data = loadfiles(filenames)
    if not len(data):
        return (None, None, isGradient)
    filenames = filenames + ["mean"] if '--mean' in args else filenames
    if isinstance(data[0], type("")):
        data = [parse_string(data[0])]
    data = formatline(data, isGradient, indices, intervals, normalize)
    return (data, filenames, isGradient)


def parse_title(fname):
    return fname.replace("_"," ").replace(".pickle","").capitalize()


def legend(isGradient, fnames = []):
    legends = {'tuned_accuracies.pickle': 'Tuned Accuracy', 
        'raw_accuracies.pickle': 'Plain Accuracy', 'train_costs.pickle': 
        'Training Mean Square Error', 'test_costs.pickle': 
        'Testing Mean Squared Error', 'boot_coeffs.pickle': 
        'Coefficient'+(' Gradient ' if isGradient else '')+' Value' }
    for fname in fnames:
        if not fname in legends:
            legends[fname] = parse_title(fname)
    return legends


def printstats(prefix, data):
    for datum in data:
        print(prefix + ".data = " + str(shape(datum)) + ", std = " + 
            str(std(datum)) + ", mean = " + str(mean(datum))  + ", max = " + 
            str(max(datum+[0])) + ", min = " + str(min(datum+[0])) + 
            ", convergence = " + str(is_converged(datum)))


def plotlines(args):
    (data, filenames, isGradient) = parse_line_args(args)
    if not data:
         return
    printstats("plotlines()", data)
    lgd = legend(isGradient, filenames)
    xlabel(nextv(args, '--xaxis', 'Interval'))
    if len(filenames) <= 1:
        if '--vs' in args:
            plot(data[0], data[1], 'r--')
        else:
            plot(arange(0, len(data[0]), 1), data[0], 'r--')
        ylabel(nextv(args, '--yaxis', lgd[filenames[0]] + '(red)'))
        title(nextv(args, '--title', lgd[filenames[0]] + ' vs Time'))
    else:
        styles = ['r--', 'b','g','y'] #['r--', 'g^', 'bo','y']
        (x,n) =(data[0],1) if '--vs' in args else (arange(0, len(data[0]), 1),0)
        for i,datum in zip(range(len(data[n:])), data[n:]):
            df = DataFrame({'x': x, 'y' + str(i): datum})
            plot('x','y'+str(i), styles[i], data=df, label=lgd[filenames[i]])
        pylab.legend()
        ylabel(nextv(args, '--yaxis', 'Value'))
        title(nextv(args, '--title', 'Tuned vs Plain EKF Accuracy by Epoch'))
    show()


def nextv(lst, k, defaultval):
    if k in lst:
        return lst[lst.index(k)+1].replace("--", "")
    return defaultval


def delta_convergence(data):
    (deltas, step, t) = ([], 100, 0.03)
    for i in range(int(len(data)/step)):
        j,k = (i*step, (i + 1) * step)
        deltas.append(mean(gradient(data[j:k])))
    return deltas


def is_converged(data, confidence=0.95):
    return len(list(filter(lambda c: c>=confidence, convergence(data))))>0


def convergence(data):
    (stat, stds, window, step, t) = ([], [], 3, 100, 0.03)
    for i in range(int(len(data)/step)):
        n = (i+1) * step
        stds.append(std(data[i*step:n]))
        #s = std(stds)
        win = stds[-window:]
        m = mean(win) #stds[-window] if len(stds)>=window else stds)
        (l, u) = (m-t, m+t) #stats.norm.interval(t, loc=m, scale=1)
        #win = list(map(abs, data[i * step: n]))
        #grads = list(map(lambda x:abs(x[1]-x[0]), zip(stds[0:-1],stds[1:])))
        #print("win = " + str(win) + ", bounds = " + str((l,u)) + ", stds = " + str(stds) + ", grads = " + str(grads) + ", stds.mean = " + str(mean(stds)))
        #confidence = len(list(filter(lambda d: d>=l and d<=u, win)))/len(win)
        confidence = len(list(filter(lambda d: d>=l and d<=u, win)))/len(win)
        stat.append(confidence)
    print("confidences = " + str(stat) + ", stds = " + str(stds))
    return stat[1:] # Ignore 1st window, its always 100% by definition


def repeat(v, n):
    return list(map(lambda i: v, range(n)))


def mapl(fn, vals):
    return list(map(fn, vals))


def scalar(v):
    return log(abs(v) + 1e-10) * 20


def plotscatter(filenames):
    data = loadfiles(filenames)
    data = data[0] if len(data) else data
    if not len(data):
        return
    dimx = len(data[0][0])
    print("scatter.dimx = " + str(dimx))
    x = array(mapl(lambda i: repeat(i, dimx), range(len(data)))).flatten()
    y = array(mapl(lambda d: d[0][0:dimx], data)).flatten()
    w = array(mapl(lambda i: list(range(dimx)), range(len(data)))).flatten()
    z = array(mapl(lambda d: mapl(scalar, d[1][0:dimx]), data)).flatten()
    print("scatter.x.len = " + str(len(x)) + ", y = " + str(len(y)) + ", z = " + str(len(z)) + ", w = " + str(len(w)))
    data = {'a': x, 'b': y, 'c': w, 'd': z}
    scatter('a', 'b', c='c', s='d', data=data)
    xlabel('Iteration')
    ylabel('Accuracy by perf metric')
    title('Bootstrap Accuracies Per Process')
    show()



def usage():
    print("\nUsage: " + sys.argv[0] + " [-h | -s | -l | -p | -3] [file(s)]\n" +
        "\nPlots data stored in pickle files\n" +
        "\nOptional arguments:\n\n" +
        "-h, --help              Show this help message and exit\n" +
        "-s, --scatter     FILE  Scatter plot for 2d array\n" +
        "-l, --line        FILES Plot one or more 1d arrays\n" +
        "-p, --predictions FILE  Plot predictions in predicions[n].pickle\n" +
        "-3d, --surface3d  FILE  3d surface plot\n" +
        "--xaxis           TEXT  x-axis label (with --line)\n" +
        "--yaxis           TEXT  y-axis label (with --line)\n" +
        "--title           TEXT  title (with --line)\n" +
        "--vs                    dataset/FILE 1 as x-axis (with --line)\n" +
        "--mean                  add mean line to plot (with --line)")

                        

def main():
    try:
        opts, args = getopt(sys.argv[1:], "hslp:3", ["help", "scatter", "line", "predictions", "surface3d"])
    except GetoptError as err:
        print(str(err))
        usage()
        exit(2)
    for o, a in opts:
        arg = args + [a] if len(a) else args
        if o in ( "-s", "--scatter" ):
            plotscatter(arg)
        elif o in ( "-l", "--line" ):
            plotlines(arg)
        elif o in ( "-p", "--predictions" ):
            plotpredictions(arg)
        elif o in ( "-3", "--surface3d" ):
            plotsurface(arg)
        else:
            usage()
    if not len(opts):
        usage()


if __name__ == "__main__":
    main()
