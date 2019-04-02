'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils import *
from wired3d import *


def sortPID(msmt):
    ncol = 8
    l = array(msmt)
    l.resize(int(len(msmt)/ncol), ncol)
    l = list(l)
    l.sort(key = lambda r: r[0])
    return array(l).flatten()


def extend(lst, n):
    return array(list(lst) + list(map(lambda i: 0, range(n))))


def plotsurface(files = []):
    print("plotsurface.files = " + str(files))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.

    msmts = pickleload(files[0] if len(files) else "measurements.pickle")

    ymax = max(list(map(lambda m: len(m), msmts)))
    xmax = len(msmts)
    xstep = 5
    print("ymax = " + str(ymax) + ", xmax = " + str(xmax) + ", msmts = " + \
        str(len(msmts)) + ", xmax/xstep = " + str(xmax/xstep))
    X = np.arange(0, xmax, xstep)
    Y = np.arange(0, ymax, max(1, int(ymax/(xmax/xstep))))[0:len(X)]
    msmts = list(map(sortPID, msmts))
    msmts = list(map(lambda m: extend(m, ymax), msmts))
    print("ymax = " + str(ymax) + ", xmax = " + str(xmax) + ", msmts.len = " + \
        str(len(msmts)) + ", X[-1]=" + str(X[-1]) + ", Y[-1]=" + str(Y[-1]) + \
        ", X.shape = " + str(X.shape) + ", Y.shape = " + str(Y.shape))
    Z = twod([[msmts[int(x)][int(y)] for y in Y] for x in X])
    X, Y = np.meshgrid(X, Y)
    means = list(map(lambda row: avg(row), Z))
    maxs = list(map(lambda row: max(row), Z))
    def normX(x):
        return [(Z[x][y]-means[x])/(maxs[x]) for y in range(len(Z[x]))]
    Z = twod([normX(i) for i in range(len(Z))]).T

    print("Z.shape = " + str(Z.shape) + ", X.shape = " + str(X.shape) + \
          ", Y.shape = " + str(Y.shape))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.set_zlabel('Normalized value')
    ax.set_xlabel('Process metric')
    ax.set_ylabel('Iteration')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.title('Normalized Process Load (% CPU, Utilization, Memory) By Time')
    plt.show()


if __name__ == "__main__":
    #plotsurface()
    v = [12,1,1,1,2,2,0,4,4,0,0,0,0,0,0,0]
    print("sorted " + str(v) + " to " + str(sortPID(v)))
