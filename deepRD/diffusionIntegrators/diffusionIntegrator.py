'''
Parent class of diffusion dynamics integrators. The integrators would generally take a particle/particle list
as input (see particle class)
'''

import numpy as np
import sys
import itertools
from ..particle import particle

class diffusionIntegrator:
    '''
    Parent (abstract) class for all diffusion integrators
    '''

    def __init__(self, dt=0.0001, stride=1, tfinal=1000, kBT = 1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0):
        # Define default simulation parameters
        self.setSimulationParameters(dt, stride, tfinal, kBT, boxsize, boundary)
        self.externalPotential = None
        self.pairPotential = None
        self.equilibrationSteps = equilibrationSteps
        self.forceField = None
        self.firstRun = True
        self.currentOrNext = 'next' # Determines if integration done on 'current' or 'next' position particle variable


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

    def prepareSimulation(self, particleList):
        '''
        Routine used to setup integration for a particular list of particles.
        In general, it will calculate forceTorque for the first time step,
        but it can be overriden for more complex behavior. This routine is only run
        once before the first integration.
        '''
        self.currentOrNext = 'next'
        self.calculateForceField(particleList)
        self.firstRun = False

    def integrateOne(self, particleList):
        '''
        'Abstract' method used to integrate one time step or iteration of the
        current algorithm
        '''
        raise NotImplementedError("Please Implement integrateOne method")

    def propagate(self, particleList, showProgress = False):
        '''
        'Abstract' method used to integrate propagate the algorithm up to
        tfinal. If showProgress ==  True, prints percentage of integration
        completed.
        '''
        raise NotImplementedError("Please Implement propagate method")

    def enforceBoundary(self, particleList, currentOrNextOverride = None):
        '''
        The whichPosition variable can take values of 'current' or 'next'.
        '''
        if (currentOrNextOverride != None):
            currentOrNext = currentOrNextOverride
        else:
            currentOrNext = self.currentOrNext
        if self.boundary == 'periodic' and self.boxsize != None:
            if currentOrNext  == 'current':
                for particle in particleList:
                    for j in range(particleList.dimension):
                        if (particle.position[j] >= self.boxsize[j]/2):
                            particle.position[j] -= self.boxsize[j]
                        if (particle.position[j] <= - self.boxsize[j] / 2):
                            particle.position[j] += self.boxsize[j]
            elif currentOrNext  == 'next':
                for particle in particleList:
                    for j in range(particleList.dimension):
                        if (particle.nextPosition[j] >= self.boxsize[j]/2):
                            particle.nextPosition[j] -= self.boxsize[j]
                        if (particle.nextPosition[j] <= - self.boxsize[j] / 2):
                            particle.nextPosition[j] += self.boxsize[j]


    def calculateForceField(self, particleList, currentOrNextOverride = None):
        ''' Default force term is zero. General force calculations can be implemented here. It should
        output the force exterted into particle indexed by particleIndex. The currentOrNext variable can
        take values of 'current' or 'next'.'''
        if (currentOrNextOverride != None):
            currentOrNext = currentOrNextOverride
        else:
            currentOrNext = self.currentOrNext
        dim = len(particleList[0].velocity)
        fField = [np.zeros(dim)]*len(particleList)
        if self.externalPotential != None:
            for i, particle in enumerate(particleList):
                fField[i] += self.externalPotential.calculateForce(particle, currentOrNext)
        if self.pairPotential != None:
            for ij in list(itertools.combinations(range(len(particleList)), 2)):
                i = ij[0]
                j = ij[1]
                force = self.pairPotential.calculateForce(particleList[i], particleList[j], currentOrNext)
                fField[i] += 1.0 * force
                fField[j] -= 1.0 * force
        self.forceField = fField

    def setExternalPotential(self, externalPot):
        self.externalPotential = externalPot

    def setPairPotential(self, pairPot):
        self.pairPotential = pairPot
        self.pairPotential.boxsize = self.boxsize
        self.pairPotential.boundary = self.boundary