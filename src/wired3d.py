'''
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
'''

from utils import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def plot3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Grab some test data.
    #X, Y, Z = axes3d.get_test_data(0.05)

    msmts = pickleload("measurements.pickle")

    X = twod(list(map(lambda x: list(repeat(x, len(msmts[0]))), range(len(msmts)))))
    Y = twod(list(map(lambda x: list(range(len(msmts[x]))), range(len(msmts)))))
    Z = twod(list(map(lambda x: list(msmts[x]), range(len(msmts)))))

    print("X = " + str(X.shape))
    print("Y = " + str(Y.shape))
    print("Z = " + str(Z.shape))


    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()
