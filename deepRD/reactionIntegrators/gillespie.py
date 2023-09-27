import numpy as np
from .reactionIntegrator import reactionIntegrator

class gillespie(reactionIntegrator):
    '''
    Integrator class to integrate a well-mixed reaction model using the gillespie or SSA algorithm
    '''

    def __init__(self, stride=1, tfinal=1000):
        # inherit all methods from parent class
        super().__init__(0, stride, tfinal)

    def integrateOne(self, reactionModel, returnReactionIndex=False):
        '''
        One iteration of the Gillespie algorithm. Outputs lagtime and
        final value of copy numbers after iteration
        '''
        lambda0 = np.sum(reactionModel.propensities)
        ratescumsum = np.cumsum(reactionModel.propensities)
        # Gillespie, time and transition (reaction index)
        r1 = np.random.rand()
        lagtime = np.log(1.0 / r1) / lambda0
        if reactionModel.nreactions > 1:
            r2 = np.random.rand()
            reactionIndex = int(sum(r2 * lambda0 > ratescumsum))
        else:
            reactionIndex = 0
        deltaX = reactionModel.reactionVectors[reactionIndex]
        nextX = reactionModel.X + deltaX
        if returnReactionIndex:
            return lagtime, nextX, reactionIndex
        else:
            return lagtime, nextX


    def integrateMany(self, reactionModel, tau, substeps=None):
        '''
        Integrates Gillespies up to a time interval tau, substeps is unused in this
        routine, but is left for similarity with tau-leap implementation
        '''
        time = 0.0
        while (time <= tau):
            lagtime, nextX = self.integrateOne(reactionModel)
            time += lagtime
            if (time <= tau):
                reactionModel.X = nextX
                reactionModel.updatePropensities()
        return nextX


    def propagate(self, reactionModel):
        '''
        Integrate reaction model until tfinal using the Gillespie and
        outputs full trajetory Xtraj
        '''
        percentage_resolution = self.tfinal / 1000.0
        time_for_percentage = - 1 * percentage_resolution
        # Begins Gillespie algorithm
        t = 0.0
        Xtraj = [reactionModel.X]
        times = [t]
        while t <= self.tfinal:
            lagtime, nextX = self.integrateOne(reactionModel)
            # Update variables
            Xtraj.append(nextX)
            reactionModel.X = nextX
            reactionModel.updatePropensities()
            t += lagtime
            times.append(times[-1] + lagtime)
            # Print integration percentage
            if (t - time_for_percentage >= percentage_resolution):
                time_for_percentage = 1 * t
                print("Percentage complete ", round(100 * t / self.tfinal, 1), "%           ", end="\r")
        print("Percentage complete 100%       ", end="\r")
        return times, Xtraj


