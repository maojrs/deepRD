import numpy as np
from .potentials import externalPotential

class harmonic(externalPotential):
    '''
    Potential class for harmonic potential
    '''
    def __init__(self, kconstant):
        self.kconstant = kconstant
        if np.isscalar(kconstant):
            self.boxsize = [kconstant, kconstant, kconstant]

    def evaluate(self, particle):
        return np.dot(self.konstant * particle.position, particle.position)

    def calculateForce(self, particle):
        '''
        Calculates potential
        '''
        return -2 * self.kconstant * particle.position