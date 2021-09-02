import numpy as np
from .langevin import langevin

class langevinAux(langevin):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle following the Langevin equation,
    where the noise term and interaction terms in the velocity equation are sampled from an alternate data-based
    model. Takes same input as Langevin integrator with the addition of a noise sampler. The noise sampler
    takes certain input and outputs a corresponding noise term.
    '''

    def __init__(self, dt, stride, tfinal, noiseSampler, kBT=1, boxsize = None, boundary = 'periodic'):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary)
        self.noiseSampler = noiseSampler
        self.prevNoiseTerm = np.zeros(3)

    def integrateOne(self, particleList):
        nextPositions = [None] * len(particleList)
        nextVelocities = [None] * len(particleList)
        for i, particle in enumerate(particleList):
            position = particle.position
            velocity = particle.velocity
            # Integrate BAOAB
            position, velocity = self.integrateB(position, velocity, particle.mass)
            position, velocity = self.integrateA(position, velocity)
            position, velocity = self.integrateO(position, velocity, particle.D, particle.mass, particle.dimension)
            position, velocity = self.integrateA(position, velocity)
            position, velocity = self.integrateB(position, velocity, particle.mass)
            nextPositions[i] = position
            nextVelocities[i] = velocity
        return nextPositions, nextVelocities

    def integrateO(self, position, velocity, D, mass, dimension):
        '''Integrates velocity full time step given friction and noise term'''
        eta = self.kBT / D # friction coefficient
        xi = np.sqrt(self.kBT * (1 - np.exp(-2 * eta * self.dt)))
        conditionedVariables = [position]
        noiseTerm = self.noiseSampler.sample(conditionedVariables)
        self.prevNoiseTerm = noiseTerm
        velocity = (np.exp(-self.dt * eta) / mass) * velocity + noiseTerm
        return position, velocity