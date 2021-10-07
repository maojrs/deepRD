'''
Parent class of diffusion dynamics integrators. The integrators would generally take a particle/particle list
as input (see particle class)
'''

import numpy as np
import sys
from ..particle import particle

class diffusionIntegrator:
    '''
    Parent (abstract) class for all diffusion integrators
    '''

    def __init__(self, dt=0.0001, stride=1, tfinal=1000, kBT = 1, boxsize = None, boundary = 'periodic'):
        # Define default simulation parameters
        self.setSimulationParameters(dt, stride, tfinal, kBT, boxsize, boundary)
        self.externalPotential = None
        self.pairPotential = None

    def setSimulationParameters(self, dt, stride, tfinal, kBT, boxsize, boundary):
        '''
        Function to set simulation parameters. This will be inherited
        and used by child classes
        '''
        self.dt = dt
        self.stride = stride
        self.tfinal = tfinal
        self.kBT = kBT
        self.timesteps = int(self.tfinal/self.dt)
        self.boxsize = boxsize
        if np.isscalar(boxsize):
            self.boxsize = [boxsize, boxsize, boxsize]
        self.boundary = boundary

    def integrateOne(self, particleList):
        '''
        'Abstract' method used to integrate one time step or iteration of the
        current algorithm
        '''
        raise NotImplementedError("Please Implement integrateOne method")

    def propagate(self, particleList):
        '''
        'Abstract' method used to integrate propagate the algorithm up to
        tfinal.
        '''
        raise NotImplementedError("Please Implement propagate method")

    def enforceBoundary(self, particleList):
        if self.boundary == 'periodic' and self.boxsize != None:
            for particle in particleList:
                for j in range(particleList.dimension):
                    if (particle.position[j] >= self.boxsize[j]/2):
                        particle.position[j] -= self.boxsize[j]
                    if (particle.position[j] <= - self.boxsize[j] / 2):
                        particle.position[j] += self.boxsize[j]


    def calculateForce(self, particleList, particleIndex):
        ''' Default force term is zero. General force calculations can be implemented here. It should
        output the force exterted into particle indexed by particleIndex'''
        force = 0.0 * particleList[particleIndex].velocity
        if self.externalPotential != None:
            force += self.externalPotential.calculateForce(particleList[particleIndex])
        if self.pairPotential != None:
            ''' Could be implemented more efficiently, at the moment calculating twice every interaction'''
            for i, part in enumerate(particleList):
                if i != particleIndex:
                    force += self.pairPotential.calculateForce(particleList[particleIndex])
        return force

    def setExternalPotential(self, externalPot):
        self.externalPotential = externalPot

    def setPairPotential(self, pairPot):
        self.pairPotential = pairPot