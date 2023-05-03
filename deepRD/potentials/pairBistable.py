import numpy as np
from .potentials import pairPotential

class pairBistable(pairPotential):
    '''
    Potential class for bistable potential between two particles (dimer with
    two possible configurations). The potential is of the form:
    scale ( 1 - ((x-(x0+rad))/rad)^2 )^2
    x0 location of first minima
    rad half the distance between minimas (distance between minimas = 2 * rad)
    scale factor that scales the whole potential
    '''
    def __init__(self, x0, rad, scale = 1):
        super().__init__()
        self.x0 = x0
        self.rad = rad
        self.scale = scale

    def evaluate(self, particle1, particle2):
        relPos = self.relativePosition(particle1.position, particle2.position)
        x = np.linalg.norm(relPos)
        bistablePot = (1 - ((x - (self.x0 + self.rad))/self.rad)**2 )**2
        return self.scale * bistablePot

    def calculateForce(self, particle1, particle2, currentOrNext = 'current'):
        '''
        Calculates force due to potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        force = np.zeros(3)
        if currentOrNext == 'current':
            relPos = self.relativePosition(particle2.position, particle1.position)
        elif currentOrNext == 'next':
            relPos = self.relativePosition(particle2.nextPosition, particle1.nextPosition)
        else:
            raise NotImplementedError("CurrentOrNext variable must take values of current or next.")
        x = np.linalg.norm(relPos)
        arg = (x - (self.x0 + self.rad)) / self.rad
        dVdr = 4.0 * (1 - arg**2) * arg / self.rad
        force[0] = dVdr * relPos[0] / x
        force[1] = dVdr * relPos[1] / x
        force[2] = dVdr * relPos[2] / x
        return self.scale * force

class pairBistableBias(pairPotential):
    '''
    Potential class for bias bistable potential between two particles (dimer with
    two possible configurations). The potential is of the form:
    scale ( 1 - ((x-(x0+rad))/rad)^2 )^2 + a * log(x)
    x0 approx location of first minima
    rad approx half the distance between minimas (distance between minimas = 2 * rad)
    scale factor that scales the whole potential
    '''
    def __init__(self, x0, rad, scale = 1):
        super().__init__()
        self.x0 = x0
        self.rad = rad
        self.scale = scale
        self.a = 1.0

    def evaluate(self, particle1, particle2):
        relPos = self.relativePosition(particle1.position, particle2.position)
        x = np.linalg.norm(relPos)
        bistablePot = (1 - ((x - (self.x0 + self.rad))/self.rad)**2 )**2 + self.a * np.log(x)
        return self.scale * bistablePot

    def calculateForce(self, particle1, particle2, currentOrNext = 'current'):
        '''
        Calculates force due to potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        force = np.zeros(3)
        if currentOrNext == 'current':
            relPos = self.relativePosition(particle2.position, particle1.position) #CHECK ORDER OF ARGUMENTS
        elif currentOrNext == 'next':
            relPos = self.relativePosition(particle2.nextPosition, particle1.nextPosition)
        else:
            raise NotImplementedError("CurrentOrNext variable must take values of current or next.")
        x = np.linalg.norm(relPos)
        arg = (x - (self.x0 + self.rad)) / self.rad
        dVdr = 4.0 * (1 - arg**2) * arg / self.rad - self.a/x
        force[0] = dVdr * relPos[0] / x
        force[1] = dVdr * relPos[1] / x
        force[2] = dVdr * relPos[2] / x
        return self.scale * force