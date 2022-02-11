import numpy as np
from .potentials import externalPotential

class bistable(externalPotential):
    '''
    Bistable external potential in 3D corresponding to
    scaleFactor ( k1 (1-(x/minimaDist)^2)^2 + k2 y^2 + k3 z^2)
    The minimas are x = +/- minimaDist
    '''
    def __init__(self, minimaDist, kconstants, scale = 1):
        self.minimaDist = minimaDist
        self.kconstants = kconstants
        if np.isscalar(kconstants):
            self.kconstant = np.array([kconstants, kconstants, kconstants])
        self.scale = scale

    def evaluate(self, particle):
        x = particle.position
        bistablePot = self.kconstants[0] * (1 - (x[0] / self.minimaDist)**2)**2 + \
                      self.kconstants[1] * x[1]**2 + self.kconstants[2] * x[2]**2
        return self.scale * bistablePot

    def calculateForce(self, particle, currentOrNext = 'current'):
        '''
        Calculates force due to potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        force = np.zeros(3)
        if currentOrNext == 'current':
            x = particle.position
        elif currentOrNext == 'next':
            x = particle.nextPosition
        else:
            raise NotImplementedError("CurrentOrNext variable must take values of current or next.")
        pifactor = np.power(2*np.pi, 3.0 / 2.0)
        force[0] = - self.kconstants[0] * 4 * x[0] * (x[0]**2 - self.minimaDist**2)/self.minimaDist**4
        force[1] = - self.kconstants[1] * 2 * x[1]
        force[2] = - self.kconstants[2] * 2 * x[2]
        return self.scale * force
