import numpy as np
from .potentials import externalPotential

class bistable(externalPotential):
    '''
    Potential class for bistable potential made with two Gaussians of the form:
    -scale * (exp( -((x-mu1)*(x-mu1)/sigma1^2))/(2*sigma1**3) +
    +         exp( -((x-mu2)*(x-mu2)/sigma2^2))/(2*sigma2**3)).
    Assumes mu1 and mu2 are vectors and sigma1 and sigma 2 scalars (equal variance in all directions).
    '''
    def __init__(self, mu1, mu2, sigma1, sigma2, scale = 1):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.scale = scale

    def evaluate(self, particle):
        x1 = particle.position - self.mu1
        x2 = particle.position - self.mu2
        pifactor = np.power(2*np.pi, 3.0 / 2.0)
        gaussian1 = np.exp(-np.dot(x1,x1) / (2*self.sigma1**2)) / (pifactor * self.sigma1**3)
        gaussian2 = np.exp(-np.dot(x2,x2) / (2*self.sigma2**2)) / (pifactor * self.sigma2**3)
        return -1 * self.scale * (gaussian1 + gaussian2)

    def calculateForce(self, particle, whichPosition = 'current'):
        '''
        Calculates force due to potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        force = np.zeros(3)
        if whichPosition == 'current':
            x1 = particle.position - self.mu1
            x2 = particle.position - self.mu2
        elif whichPosition == 'next':
            x1 = particle.nextPosition - self.mu1
            x2 = particle.nextPosition - self.mu2
        pifactor = np.power(2*np.pi, 3.0 / 2.0)
        gaussian1 = np.exp(-np.dot(x1,x1) / (2*self.sigma1**2)) / (pifactor * self.sigma1**3)
        gaussian2 = np.exp(-np.dot(x2,x2) / (2*self.sigma2**2)) / (pifactor * self.sigma2**3)
        force = -(x1/self.sigma1**2) * gaussian1 -(x2/self.sigma2**2) * gaussian2
        return self.scale * force
