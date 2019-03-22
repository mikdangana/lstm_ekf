from pylab import *
from numpy import array, gradient, mean, std
from scipy import stats
from math import log
from getopt import getopt, GetoptError
from surface3d import plotsurface
import pickle


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


def plotpredictions(args):
    predictions, metric = load(args[0] if len(args) else "")
    x = list(map(lambda x: x[0][metric], predictions))
    y = list(map(lambda x: x[1][0][0], predictions))

    xline, = plot(x)
    yline, = plot(y)

    lgd = ['step(x,50)', 'exp(x)', 'sin(x)', 'erf(x)']

    legend((xline, yline), (lgd[metric], 'EKF.x_prior'))
    title('EKF Prior Prediction Vs ' + lgd[metric])
    show()



def plotlines(filenames):
    isGradient = len(list(filter(lambda f: f=="gradient", filenames))) 
    intervals = list(filter(lambda f: ":" in f, filenames))
    indices = list(filter(lambda f : f.isdigit(), filenames))
    filenames = list(filter(lambda f : not f.isdigit() and f!="gradient" and not ":" in f, filenames))
    data = loadfiles(filenames)
    if not len(data):
        return

    if len(indices):
        if isGradient:
            data = list(map(lambda i: gradient(array(list(map(lambda r:r[int(i)],data[0])))), indices)) 
        else:
            data = list(map(lambda i: list(map(lambda r:abs(r[int(i)]),data[0])), indices)) 

    if len(intervals):
        (start, end) = intervals[0].split(":")
        data[0] = data[0][int(start):int(end)]
    print("plotlines().data[0] = " + str(len(data[0])) + ", std = " + str(std(data[0])) + ", mean = " + str(mean(data[0])) + ", convergence = " + str(is_converged(data[0])))
    x = arange(0, len(data[0]), 1) 

    lgd = {'tuned_accuracies.pickle': 'Tuned Accuracy', 'raw_accuracies.pickle': 'Plain Accuracy', 'train_costs.pickle': 'Training Mean Square Error', 'test_costs.pickle': 'Testing Mean Squared Error', 'boot_coeffs.pickle': 'Single Coefficients'+(' Gradient ' if isGradient else '')+' Values' }

    xlabel('Interval')
    if len(filenames) <= 1:
        plot(x, data[0], 'r--')
        ylabel(lgd[filenames[0]] + '(red)')
        title(lgd[filenames[0]] + ' vs Time')
    else:
        plot(x, data[0], 'r--', x, data[1], 'g^', label='L2')
        ylabel(lgd[filenames[0]] + '(red) & ' + lgd[filenames[1]] + '(green)')
        title('Comparing Tuned & Plain EKF Accuracy vs Time')
    show()


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
        m = mean(stds) #stds[-window] if len(stds)>=window else stds)
        (l, u) = (m-t, m+t) #stats.norm.interval(t, loc=m, scale=1)
        #win = list(map(abs, data[i * step: n]))
        #grads = list(map(lambda x:abs(x[1]-x[0]), zip(stds[0:-1],stds[1:])))
        #print("win = " + str(win) + ", bounds = " + str((l,u)) + ", stds = " + str(stds) + ", grads = " + str(grads) + ", stds.mean = " + str(mean(stds)))
        #confidence = len(list(filter(lambda d: d>=l and d<=u, win)))/len(win)
        win = stds[-window:]
        confidence = len(list(filter(lambda d: d>=l and d<=u, win)))/len(win)
        stat.append(confidence)
    print("confidences = " + str(stat))
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
    print("scatter.data.shape = " + str(shape(data)))
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
    print("\nUsage: " + sys.argv[0] + " [-h | -s | -l | -p | -3d] [file(s)]\n" +
        "\nPlots data stored in pickle files\n" +
        "\nOptional arguments:\n\n" +
        "-h, --help              Show this help message and exit\n" +
        "-s, --scatter     FILE  Scatter plot for 2d array\n" +
        "-l, --line        FILES Plot one or more 1d arrays\n" +
        "-p, --predictions FILE  Plot predictions in predicions[n].pickle\n" +
        "-3d, --surface3d  FILE  3d surface plot")

                        

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
