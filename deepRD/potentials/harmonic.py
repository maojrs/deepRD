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

    def calculateForce(self, particle, whichPosition = 'current'):
        '''
        Calculates force due to potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        if whichPosition == 'current':
            return -2 * self.kconstant * particle.position
        else:
            return -2 * self.kconstant * particle.nextPosition