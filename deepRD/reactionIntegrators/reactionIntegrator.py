'''
Parent class of reaction dynamics integrators. The integrators would generally take a reactionModel
as input (see models/reactionModel abstract class).
'''

import numpy as np

class reactionIntegrator:
    '''
    Parent (abstract) class for all reaction integrators
    '''

    def __init__(self, dt=0.0001, stride=1, tfinal=1000):
        # Define default simulation parameters
        self.setSimulationParameters(dt, stride, tfinal)

    def setSimulationParameters(self, dt, stride=1, tfinal=1000):
        '''
        Function to set simulation parameters. This will be inherited
        and used by child classes
        '''
        self.dt = dt
        self.stride = stride
        self.tfinal = tfinal
        if self.dt == 0:
            # In case of no fixed timestepping, like Gillespie, these variables are useless
            self.timesteps  = 0
        else:
            self.timesteps = int(self.tfinal/self.dt)

    def integrateOne(self, reactionModel):
        '''
        'Abstract' method used to integrate one time step or iteration of the
        current algorithm
        '''
        raise NotImplementedError("Please Implement integrateOne method")

    def propagate(self, reactionModel):
        '''
        'Abstract' method used to integrate propagate the algorithm up to
        tfinal.
        '''
        raise NotImplementedError("Please Implement propagate method")

