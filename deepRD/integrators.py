import numpy as np


def integrateGillespie(reactionModel, tfinal = None):
    '''Do Gillespie integration loop'''
    Xtraj = []
    Xtraj.append(reactionModel.X)
    t = 0.0
    i = 0
    if tfinal == None:
        tfinal = reactionModel.timesteps * reactionModel.dt
    while t <= tfinal:
        r1 = np.random.rand()
        r2 = np.random.rand()
        lambda0 = np.sum(reactionModel.lambdas)
        ratescumsum = np.cumsum(reactionModel.lambdas)
        # Gillespie, time and transition
        lagtime = np.log(1.0 / r1) / lambda0
        reactionIndex = int(sum(r2 * lambda0 > ratescumsum))
        nextX = Xtraj[i] + reactionModel.reactionVectors[reactionIndex]
        Xtraj.append(nextX)
        reactionModel.X = nextX
        reactionModel.updatePropensities()
        t = t + lagtime
        i = i + 1
        print(len(Xtraj))
    return Xtraj