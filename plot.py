from pylab import *
from numpy import array
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


def plotpredictions(arg):
    predictions, metric = load(arg)
    x = list(map(lambda x: x[0][metric], predictions))
    y = list(map(lambda x: x[1][0][0], predictions))

    xline, = plot(x)
    yline, = plot(y)

    lgd = ['x: x<50 ? 0 : 1', 'exp(x)', 'sin(x)', 'erf(x)']

    legend((xline, yline), (lgd[metric], 'EKF.x_prior'))
    show()



def plotlines(filenames):
    data = loadfiles(filenames)
    if not len(data):
        return
    x = arange(0, len(data[0]), 1)

    lgd = {'tuned_accuracies.pickle': 'Tuned Accuracy', 'raw_accuracies.pickle': 'Plain Accuracy'}

    if len(filenames) <= 1:
        plot(x, data[0], 'r--')
    else:
        plot(x, data[0], 'r--', x, data[1], 'g^', label='L2')
        xlabel('Interval')
        ylabel(lgd[filenames[0]] + '(red) & ' + lgd[filenames[1]] + '(green)')
        title('Comparing Tuned & Plain EKF Accuracy vs Time')
    show()



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
        if o in ( "-s", "--scatter" ):
            plotscatter(args)
        elif o in ( "-l", "--line" ):
            plotlines(args)
        elif o in ( "-p", "--predictions" ):
            plotpredictions(args)
        elif o in ( "-3", "--surface3d" ):
            plotsurface(args)
        else:
            usage()
    if not len(opts):
        usage()


if __name__ == "__main__":
    main()
