'''
Integrators of reaction dynamics. The integrators would generally take a reactionModel
as input (see models/reactionModel abstract class).
'''

import numpy as np

class integrator:
    '''
    Parent (abstract) class for all integrators
    '''

    def __init__(self, dt, stride, tfinal):
        # Define default simulation parameters
        self.setSimulationParameters(dt, stride, tfinal)

    def setSimulationParameters(self, dt, stride, tfinal):
        '''
        Function to set simulation parameters. This will be inherited
        and used by child classes
        '''
        self.dt = dt
        self.stride = stride
        self.tfinal = tfinal
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

