from pylab import *
import pickle


def load():
    n = sys.argv[1] if len(sys.argv) > 1 else "0"

    with open('predictions' + n + '.pickle', 'rb') as f:
        return (pickle.load(f), int(n))

    return ([], int(n))


def plotmetric(predictions, metric):
    x = list(map(lambda x: x[0][metric], predictions))
    y = list(map(lambda x: x[1][0][0], predictions))

    xline, = plot(x)
    yline, = plot(y)

    lgd = ['x: x<50 ? 0 : 1', 'exp(x)', 'sin(x)', 'erf(x)']

    legend((xline, yline), (lgd[metric], 'EKF.x_prior'))
    show()


if __name__ == "__main__":
    predictions, metric = load()
    plotmetric(predictions, metric)
