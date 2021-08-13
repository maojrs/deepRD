import numpy as np

def integrateGillespie(reactionModel, tfinal = None):
    '''
    Integrate reaction model using Gillespie and outputs full trajetory Xtraj
    '''
    if tfinal == None:
        tfinal = reactionModel.timesteps * reactionModel.dt
    percentage_resolution = tfinal/1000.0
    time_for_percentage = - 1 * percentage_resolution
    # Begins Gillespie algorithm
    t = 0.0
    i = 0
    Xtraj = [reactionModel.X]
    times = [t]
    while t <= tfinal:
        r1 = np.random.rand()
        r2 = np.random.rand()
        lambda0 = np.sum(reactionModel.propensities)
        ratescumsum = np.cumsum(reactionModel.propensities)
        # Gillespie, time and transition (reaction index)
        lagtime = np.log(1.0 / r1) / lambda0
        reactionIndex = int(sum(r2 * lambda0 > ratescumsum))
        nextX = Xtraj[i] + reactionModel.reactionVectors[reactionIndex]
        # Update variables
        Xtraj.append(nextX)
        reactionModel.X = nextX
        reactionModel.updatePropensities()
        t += lagtime
        i += 1
        times.append(t)
        # Print integration percentage
        if (t - time_for_percentage >= percentage_resolution):
            time_for_percentage = 1 * t
            print("Percentage complete ", round(100*t/tfinal,1), "%           ", end="\r")
    return times, Xtraj


def integrateTauLeap(reactionModel, tfinal = None):
    '''
    Integrate reaction model using tau-leap algorithm and outputs full trajetory Xtraj
    '''
    if tfinal == None:
        tfinal = reactionModel.tfinal
    percentage_resolution = tfinal / 100.0
    time_for_percentage = - 1 * percentage_resolution
    # Begins tau-leap algorithm
    Xtraj = [reactionModel.X]
    tsteps = int(tfinal / reactionModel.dt)
    times = np.zeros(tsteps + 1)
    for i in range(tsteps):
        nextX = Xtraj[i]
        # Count number of reactions with several Poisson rate with the corresponding propensity
        numReactions = np.random.poisson(reactionModel.propensities * reactionModel.dt, reactionModel.nreactions)
        for j in range(reactionModel.nreactions):
            nextX += numReactions[j] * reactionModel.reactionVectors[j]
        ## Avoid negative copy numbers
        #for k in range(len(reactionModel.X)):
        #   if nextX[k] < 0:
        #       nextX[k] = 0
        # Update variables
        Xtraj.append(nextX)
        reactionModel.X = nextX
        reactionModel.updatePropensities()
        times[i + 1] = times[i] + reactionModel.dt
        # Print integration percentage
        if (t - time_for_percentage >= percentage_resolution):
            time_for_percentage = 1 * t
            print("Percentage complete ", round(100*t/tfinal,1), "%           ", end="\r")
    return times, Xtraj

# def propagate(self, x, tfinal=None, algorithm="tau-leap"):
#     ''' Propagate using tau-leap or SSA, from t=0 to t=tfinal. For
#     different time intervals, you can run integrateTauLeap or integrateGillespie directly.
#     Function definition so the code can be run in parallel'''
#     if tfinal == None:
#         tfinal = self.tfinal
#     if algorithm == "tau-leap":
#         xt = np.float32(self.integrateTauLeap(x, tfinal))
#         xstrided = np.zeros(int(self.timesteps / self.stride))
#     elif algorithm == "gillespie" or algorithm == "SSA":
#         xt = np.float32(self.integrateGillespie(x, tfinal))
#         xstrided = np.zeros(int(len(xt) / self.stride))
#     for j in range(len(xstrided)):
#         xstrided[j] = 1.0 * xt[j * self.stride]
#     return xstrided