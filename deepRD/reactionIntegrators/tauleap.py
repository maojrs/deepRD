import numpy as np
from .reactionIntegrator import reactionIntegrator

class tauleap(reactionIntegrator):
    '''
    Integrator class to integrate a well-mixed reaction model using the tau-leap algorithm
    '''

    def __init__(self, dt, stride=1, tfinal=1000):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal)

    def integrateOne(self, reactionModel):
        nextX = reactionModel.X
        # Count number of reactions with several Poisson rate with the corresponding propensity
        numReactions = np.random.poisson(reactionModel.propensities * self.dt,
                                         reactionModel.nreactions)
        for j in range(reactionModel.nreactions):
            nextX = nextX + numReactions[j] * np.array(reactionModel.reactionVectors[j])
        ## Avoid negative copy numbers
        # for k in range(len(reactionModel.X)):
        #   if nextX[k] < 0:
        #       nextX[k] = 0
        return nextX

    def integrateMany(self, reactionModel, tau, substeps=10):
        '''
        Integrate up to an interval tau with a given number of substeps
        '''
        subdt = tau/substeps
        for i in range(substeps):
            nextX = 1.0 * reactionModel.X
            numReactions = np.random.poisson(reactionModel.propensities * subdt,
                                             reactionModel.nreactions)
            for j in range(reactionModel.nreactions):
                nextX = nextX + numReactions[j] * np.array(reactionModel.reactionVectors[j])
            reactionModel.X = 1.0 * nextX
            reactionModel.updatePropensities()
        return reactionModel.X


    def propagate(self, reactionModel):
        '''
        Integrate reaction model using tau-leap algorithm and outputs full trajetory Xtraj
        '''
        percentage_resolution = self.tfinal / 100.0
        time_for_percentage = - 1 * percentage_resolution
        # Begins tau-leap algorithm
        Xtraj = [reactionModel.X]
        times = np.zeros(self.timesteps + 1)
        for i in range(self.timesteps):
            nextX = self.integrateOne(reactionModel)
            # Update variables
            Xtraj.append(nextX)
            reactionModel.X = nextX
            reactionModel.updatePropensities()
            times[i + 1] = times[i] + self.dt
            # Print integration percentage
            if (times[i] - time_for_percentage >= percentage_resolution):
                time_for_percentage = 1 * times[i]
                print("Percentage complete ", round(100 * times[i] / self.tfinal, 1), "%           ", end="\r")
        print("Percentage complete 100%       ", end="\r")
        return times, Xtraj
