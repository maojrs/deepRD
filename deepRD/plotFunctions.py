import numpy as np
import matplotlib.pyplot as plt
from .tools import trajectoryTools

def plotTrajectories(times, xtrajs, names = [], indexes = "all"):
    xtrajs = np.array(xtrajs)
    if indexes == "all":
        for i in range(len(xtrajs[0])):
            if names == []:
                plt.plot(times, xtrajs[:, i])
            else:
                plt.plot(times, xtrajs[:, i], label=names[i])
                plt.legend()
    else:
        for i in indexes:
            if names == []:
                plt.plot(times, xtrajs[:, i])
            else:
                plt.plot(times, xtrajs[:, i], label=names[i])
                plt.legend()
    plt.xlabel("time")
    plt.ylabel("Copy number")


def plotFPTs(FPTs, bins = 50, name = "FPT distribution"):
    plt.hist(FPTs, bins = bins, label = name, density = True, alpha = 0.6)
    plt.xlim([0,max(FPTs)])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("probability")

def plotDistribution(trajs, variableIndex, numbins= 50, label = None):
    variableArray = trajectoryTools.extractVariableFromTrajectory(trajs, variableIndex)
    plt.hist(variableArray, bins=numbins, density=True, alpha=0.5, label=label)
    plt.legend()
    plt.show()