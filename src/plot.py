from pylab import *
from numpy import array, gradient, mean, std, log
import numpy as np
from scipy import stats
from getopt import getopt, GetoptError
from pandas import DataFrame
from surface3d import plotsurface
from utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, re
import pylab


fig_file = None


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


def parse_client_info(filename):
    txt = pickleload(filename)
    if not isinstance(txt, str):
        return txt
    toks = list(filter(lambda s: s.endswith("s"), re.split("[\s]+", txt)))[2:]
    return [[float(i.replace("s",""))] for i in toks]


def parse_string(s):
    rows = filter(lambda l: "fetch" not in l and len(l), s.split("\n"))
    rows = [re.compile(r'\W+').split(r) for r in rows]
    rows = [[r[0], r[1] + "." + r[2]] for r in rows]
    return [[float(c.replace("s","")) for c in r] for r in rows[1:]]



def showplot():
    if fig_file:
        print("showplot().fig_file = " + str(fig_file))
        savefig(fig_file)
    else:
        show()


def plotpredictions(args):
    predictions, metric = load(args[0] if len(args) else "")
    x = list(map(lambda p: p[0][metric], predictions))
    y = list(map(lambda p:p[1].T[0][0], predictions))

    fig, host = subplots()
    sub1 = host.twinx()

    xline, = host.plot(x)
    yline, = sub1.plot(y, "r-", label="x_prior")

    sub1.set_ylabel(yline.get_label())
    sub1.yaxis.label.set_color(yline.get_color())
    sub1.spines["right"].set_visible(True)

    lgd = ['step(x,50)', 'exp(x)', 'sin(x)', 'erf(x)']

    legend((lgd[metric if metric<len(lgd) else -1], 'EKF.x_prior'))
    title('EKF Prior Prediction vs ' + lgd[metric if metric<len(lgd) else -1])
    showplot()


def formatline(data, isGradient, indices, intervals, normalize):
    res = []
    for n in range(len(data)):
        if '--log' in sys.argv:
            data[n] = log(data[n])
        d = formatlines(data, isGradient, indices,intervals,normalize,n)
        if not len(d):
            continue
        print("formatline.d.shape,n=" + str((len(shape(d[0])),n)))
        if not len(indices):
            for i in range(len(shape(d[0]))):
                d = [x[0] for x in d]
        print("formatline.d,shape = " + str((d, shape(d))))
        res = res + d if len(indices) else res + [d]
    if '--mean' in sys.argv:
        res.append([mean(data[0][0:i+1]) for i in range(len(data[0]))])
    return res


def formatlines(data, isGradient, indices, intervals, normalize, n):
    print("formatlines.n,shape(data) = " + str((n, len(shape(data[n])))))
    if len(indices):
        if isGradient:
            grad = lambda r,i : gradient(array([r[int(i)] for r in data[n]]))
            data = [[grad(r, i) for i in indices]]
        elif len(shape(data[n])) > 1:
            data[n] = list(filter(lambda r: len(shape(r)), data[n]))
            data[n] = [[r[int(i)] for r in data[n]] for i in indices]
            print("formatlines.data[n] =" +str(data[n][0:min(len(data[n]),10)]))

    print("formatlines.intervals,len = " + str((intervals, len(intervals))))
    if len(intervals):
        (start, end) = intervals[0].split(":")
        data[n] = data[n][int(start):int(end)]
        print("formatlines.data.intervals.len = " + str(len(data[n])))

    if normalize:
        (maxd, mind) = (max([max(d) for d in data]),min([min(d) for d in data]))
        data[n] = (array(data[n]) - mind) / (maxd - mind)
    return data[n]



def parse_line_args(args):
    print("parse_line_args().args = " + str(args))
    for k in ["--xaxis", "--yaxis", "--title"]:
        if k in args:
            i = args.index(k)
            args[i+1] = "--" + args[i+1]
    isGradient = "--gradient" in args or "-g" in args
    normalize = "--normalize" in args or "-n" in args
    intervals = list(filter(lambda f: ":" in f, args))
    indices = list(filter(lambda f : f.isdigit(), args))
    (yerr, args) = parse_err_args(args, indices)
    filenames = list(filter(isfile, args))
    data = loadfiles(filenames)
    if not len(data):
        return (None, None, isGradient)
    filenames = filenames + ["mean"] if '--mean' in args else filenames
    if isinstance(data[0], type("")):
        data = [parse_string(data[0])]
    data = formatline(data, isGradient, indices, intervals, normalize)
    return (data, filenames, isGradient, yerr)


def isfile(f): 
    return not f.isdigit() and not f.startswith("-") and not ":" in f and not "|" in f


def parse_err_args(args, indices):
    if "--err" in args:
        (i, data) = (args.index("--err"), [])
        while i+len(data)+1 < len(args) and isfile(args[i + len(data) + 1]):
            file = args[i + len(data) + 1]
            if 'clientout' in file and 'pickle' in file:
                data.append(parse_client_info(file))
            else:
                data.append(pickleload(file))
        args = args[0:i] + args[i+len(data)+1:]
        if len(indices):
            if len(indices)==1:
                data = [[r[int(indices[0])] for r in f] for f in data]
            else:
                data = [[[r[int(i)] for i in indices] for r in f] for f in data]
        return (data, args)
    return (None, args)


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
    (data, filenames, isGradient, yerr) = parse_line_args(args)
    if not data:
         return
    printstats("plotlines()", data)
    lgd = legend(isGradient, filenames)
    xlabel(nextv(args, '--xaxis', 'Interval'))
    if len(filenames) <= 1:
        if '--vs' in args:
            plot(data[0], data[1], 'b', linewidth=1)
        else:
            plot(arange(0, len(data[0]), 1), data[0], 'b', linewidth=1)
        ylabel(nextv(args, '--yaxis', lgd[filenames[0]] + '(red)'))
        title(nextv(args, '--title', lgd[filenames[0]] + ' vs Time'))
    else:
        styles = ['r--', 'b','g','y'] #['r--', 'g^', 'bo','y']
        (x,n) = (data[0],1) if '--vs' in args else (arange(0,len(data[0]),1),0)
        for i,datum in zip(range(len(data[n:])), data[n:]):
            df = DataFrame({'x': x, 'y' + str(i): datum})
            plot('x','y'+str(i), styles[i], data=df, label=lgd[filenames[i]])
        pylab.legend()
        ylabel(nextv(args, '--yaxis', 'Value'))
        title(nextv(args, '--title', 'Tuned vs Plain EKF Accuracy by Epoch'))
    showplot()


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
        m = mean(win) 
        (l, u) = (m-t, m+t) 
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
    print("scatter.x,y,z,w.len = " + str((len(x),len(y),len(z),len(w))))
    data = {'a': x, 'b': y, 'c': w, 'd': z}
    scatter('a', 'b', c='c', s='d', data=data)
    xlabel('Iteration')
    ylabel('Accuracy by perf metric')
    title('Bootstrap Accuracies Per Process')
    showplot()


def plot_hist(args):
    (data, files, isGradient, yerr) = parse_line_args(args)
    if not data: 
        return plotlines(args)
    data = groupby(len(files), data)
    (ind, width, ns) = (np.arange(len(data)), 0.25, len(data))
    fig, ax = plt.subplots()
    for c in range(len(data[0][0] if len(data) else [])):
        mean = [sum([r[c] for r in data[n]]) for n in range(len(data))]
        std = [(sum([r[c] for r in yerr[n]]) if yerr else 0) for n in range(ns)]
        print("plot_hist().mean,std = " + str((mean, std)))
        lbl = get(nextv(args, '--legend', 'Metric').split("|"), c)
        rect = ax.bar(ind-(c*width),mean,width, yerr=std, label=lbl)
    ax.set_ylabel(nextv(args, '--yaxis', 'Value'))
    ax.set_title(nextv(args, '--title', 'Title'))
    ax.set_xticks(ind)
    lbl = lambda f: re.sub('[0-9\.]+.pickle','',f.replace('predictive',''))
    ax.set_xticklabels([lbl(f) for f in files])
    ax.legend()
    fig.tight_layout()
    showplot()


def get(lst, i, default=None):
    last = lst[-1] if len(lst) else []
    return lst[i] if i < len(lst) else (default if default else last)


def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


def groupby(k, data):
    data = data if iscollection(data[0], 0) else [[[r] for r in d] for d in data]
    (n, rk, nk) = (range(len(data)), range(k), len(data)/k)
    grps = [[(data[i] if int(i/nk)==f else []) for i in n] for f in rk]
    conc = lambda a,b:[get(a,i)+get(b,i) for i in range(max(len(a),len(b)))]
    grps = list(map(lambda g: list(reduce(conc, g)), grps))
    print("groupby.grps,len(data),k = " + str((list(map(len, grps)),list(map(len, data)),k)))
    return grps


def plot_all(path, to_plot=True):
    global fig_file
    path = (path[0] if len(path) else "") if isinstance(path, list) else path
    print("plot_all().path = " + path)
    (sums, maxs, mins, count) = load_path(path)
    data = {k:array(v)/count[k] for k,v in sums.items()}
    for file in (data.keys() if to_plot else []):
        fname = os.path.join(path, file)
        fig_file = fname.replace(".pickle", ".png")
        yerr = array(subm(maxs[file], mins[file]) if file in maxs else [])/2
        plotmulti(to_line(file, data[file], yerr))
    print("plot_all() " + path + " done")
    return data


def load_path(path):
    (data, maxs, mins, count) = ({}, {}, {}, {})
    for r, d, f in os.walk(path):
        datum = {}
        for file in f:
            fname = os.path.join(r, file)
            if 'clientout.pickle' in file:
                datum[file] = parse_client_info(fname)
            elif 'measurements.pickle' in file:
                datum[file] = col_data(fname, 0)
        (data,maxs,mins,count) = merge_datum(datum,data,maxs,mins,count)
        print("load_path().f,d,path,count = " + str((f,d,path,count)))
    return (data, maxs, mins, count)


def merge_datum(datum, data, maxs, mins, count):
    if len(datum.items()):
        items = datum.items()
        data = {k:addm(v, data[k] if k in data else []) for k,v in items}
        maxs = {k:maxv(v, maxs[k] if k in maxs else []) for k,v in items}
        mins = {k:minv(v, mins[k] if k in mins else []) for k,v in items}
        count = {k: 1 + (count[k] if k in count else 0) for k,v in items}
    return (data, maxs, mins, count)


def col_data(path, index):
    data = pickleload(path)
    return [row[index] for row in data] if len(array(data).shape)>2 else data


def iscollection(vs, i):
    v = vs[i] if len(vs)>i else None
    return isinstance(v, list) or isinstance(v, type(array([])))


def addm(v1, v2):
    (a, b) = (v1, v2) if len(v1) >= len(v2) else (v2, v1)
    addfn = lambda v,w: addm(v, w) if iscollection(v1, 0) else v + w
    return [addfn(v,(v if i>=len(b) else b[i])) for i,v in zip(range(len(a)),a)]


def subm(v1, v2):
    (a, b) = (v1, v2) if len(v1) >= len(v2) else (v2, v1)
    subfn = lambda v,w: subm(v,w) if iscollection(v1, 0) else v - w
    return [subfn(v,(v if i>=len(b) else b[i])) for i,v in zip(range(len(a)),a)]


def maxv(v1, v2):
    (a, b) = (v1, v2) if len(v1) >= len(v2) else (v2, v1)
    maxfn = lambda v,w: maxv(v,w) if iscollection(v1, 0) else max(v, w)
    return [maxfn(v, v if i>=len(b) else b[i]) for i,v in zip(range(len(a)),a)]


def minv(v1, v2):
    (a, b) = (v1, v2) if len(v1) >= len(v2) else (v2, v1)
    minfn = lambda v,w: minv(v,w) if iscollection(v1, 0) else min(v, w)
    return [minfn(v, v if i>=len(b) else b[i]) for i,v in zip(range(len(a)),a)]


def to_line(fname, col, yerr):
    pickledump(fname, col)
    if len(yerr):
        efile = fname.replace(".pickle", "_err.pickle")
        pickledump(efile, yerr)
        args = [fname, "--err", efile, "0"]
    else:
        args = [fname]
    return args


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def plotmulti(args):
    (data, filenames, isGradient, yerr) = parse_line_args(args)
    if not data: 
        return plotlines(args)
    (lgd, styles) = (legend(isGradient, filenames), ['r--', 'b','g','y'])
    (x,n) = (data[0],1) if '--vs' in args else (arange(0, len(data[0]), 1),0)
    (figsize, i, delta, vtxt) = ((8,3), 0, 0.11, nextv(args,'--yaxis','Value'))
    (cols, rows) = (2, 1 if len(filenames)==1 else round(len(filenames)/2))
    fig1, axs = plt.subplots(rows,cols,figsize=figsize, constrained_layout=True)
    fig1.text(0.30, 0.04, nextv(args, '--xaxis', 'Interval'), ha='center')
    fig1.text(0.04, 0.5, vtxt, va='center', rotation='vertical')
    axs = trim_axs(axs, len(filenames))
    for ax, y, file in zip(axs, data[n:], filenames):
        if yerr:
            (ye, cap) = (every(yerr[filenames.index(file)], 10), 'projecting')
            ax.errorbar(x, y, ye, solid_capstyle=cap, capsize=2, ecolor='grey')
        else:
            ax.plot(x, y, '', ls='-', ms=4)
        ax.set_title(nextv(args, '--title', 'Subplot ') + str(suffix(file)))
    pylab.legend()
    showplot() 


def every(vs, n):
    print("every.vs = " + str(len(vs)))
    return [v if i%n==0 else 0 for i,v in zip(range(len(vs)), vs)]


def suffix(file):
    return re.sub(r"[_\\.].*", "", file.split("_")[1]) if "_" in file else ""


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
        opts, args = getopt(sys.argv[1:], "hslpat:3", ["help", "scatter", "line", "predictions", "surface3d", "all", "multi", "hist"])
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
        elif o in ( "-a", "--all" ):
            plot_all(args)
        elif o in ( "-m", "--multi" ):
            plotmulti(arg)
        elif o in ( "-t", "--hist" ):
            plot_hist(arg)
        else:
            usage()
    if not len(opts):
        usage()


if __name__ == "__main__":
    main()
