'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils import *


def plotsurface():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.

    msmts = pickleload("measurements.pickle")

    ymax = len(msmts[0])
    xmax = len(msmts)
    xstep = 5
    print("ymax = " + str(ymax) + ", xmax = " + str(xmax))
    X = np.arange(0, xmax, xstep)
    Y = np.arange(0, ymax, ymax/(xmax/xstep)) 
    Z = twod(list(map(lambda x:list(map(lambda y:msmts[int(x)][int(y)],Y)),X)))
    X, Y = np.meshgrid(X, Y)
    means = list(map(lambda row: avg(row), Z))
    maxs = list(map(lambda row: max(row), Z))
    def normX(x):
        return list(map(lambda y:(Z[x][y]-means[x])/(maxs[x]),range(len(Z[x]))))
    Z = twod(list(map(normX, range(len(Z)))))

    print("Z.shape = " + str(Z.shape) + ", Z = " + str(Z))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    plotsurface()
