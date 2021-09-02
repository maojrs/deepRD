import numpy as np
from .reactionIntegrator import reactionIntegrator

class tauleap(reactionIntegrator):

    def __init__(self, dt, stride, tfinal):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal)

    def integrateOne(self, reactionModel):
        nextX = reactionModel.X
        # Count number of reactions with several Poisson rate with the corresponding propensity
        numReactions = np.random.poisson(reactionModel.propensities * self.dt,
                                         reactionModel.nreactions)
        for j in range(reactionModel.nreactions):
            nextX += numReactions[j] * reactionModel.reactionVectors[j]
        ## Avoid negative copy numbers
        # for k in range(len(reactionModel.X)):
        #   if nextX[k] < 0:
        #       nextX[k] = 0
        return nextX

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
