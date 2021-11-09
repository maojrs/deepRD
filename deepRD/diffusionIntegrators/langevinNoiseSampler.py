import numpy as np
import sys
from .langevin import langevin

class langevinNoiseSampler(langevin):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle following the Langevin equation,
    where the noise term and interaction terms in the velocity equation are sampled from an alternate data-based
    model. Takes same input as Langevin integrator with the addition of a noise sampler. The noise sampler
    takes certain input and outputs a corresponding noise term.
    '''

    def __init__(self, dt, stride, tfinal, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'qi'):
        # inherit all methods from parent class
        integratorType = "ABOBA"
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary,integratorType, equilibrationSteps)
        self.noiseSampler = noiseSampler
        self.prevNoiseTerm = np.zeros(3)
        self.prevprevNoiseTerm = np.zeros(3)
        self.conditionedOn = conditionedOn

    def getConditionedVars(self, particle):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return (self.prevNoiseTerm)
        elif self.conditionedOn == 'ririm':
            return np.concatenate((self.prevNoiseTerm, self.prevprevNoiseTerm))
        elif self.conditionedOn == 'qi':
            return (particle.nextPosition)
        elif self.conditionedOn == 'qiri':
            return np.concatenate((particle.nextPosition, self.prevNoiseTerm))
        elif self.conditionedOn == 'qiririm':
            return np.concatenate((particle.nextPosition, self.prevNoiseTerm, self.prevprevNoiseTerm))
        elif self.conditionedOn == 'pi':
            return (particle.nextVelocity)
        elif self.conditionedOn == 'piri':
            return np.concatenate((particle.nextVelocity, self.prevNoiseTerm))
        elif self.conditionedOn == 'piririm':
            return np.concatenate((particle.nextVelocity, self.prevNoiseTerm, self.prevprevNoiseTerm))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def integrateOne(self, particleList):
        ''' Integrates one time step of ABOBA '''
        self.integrateA(particleList)
        self.enforceBoundary(particleList, 'next')
        self.integrateBOB(particleList)
        self.integrateA(particleList)
        self.enforceBoundary(particleList, 'next')
        particleList.updatePositionsVelocities()

    def integrateBOB(self, particleList):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        externalForceField = self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            # Calculate friction term and external potential term
            eta = self.kBT / particle.D  # friction coefficient
            expterm = np.exp(-self.dt * eta / particle.mass)
            frictionForceTerm = particle.nextVelocity * expterm
            frictionForceTerm += (1 + expterm) * externalForceField[i] * self.dt/(2*particle.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars(particle)
            interactionNoiseTerm = self.noiseSampler.sample(conditionedVars)
            self.prevprevNoiseTerm = 1.0 * self.prevNoiseTerm
            self.prevNoiseTerm = 1.0 * interactionNoiseTerm
            particle.nextVelocity = frictionForceTerm + interactionNoiseTerm

    # def integrateO(self, particleList):
    #     '''Integrates velocity full time step given friction and noise term'''
    #     for particle in particleList:
    #         conditionedVars = self.getConditionedVars(particle)
    #         eta = self.kBT / particle.D # friction coefficient
    #         noiseTerm = self.noiseSampler.sample(conditionedVars)
    #         self.prevNoiseTerm = 1.0 * noiseTerm
    #         frictionTerm = np.exp(-self.dt * eta/ particle.mass) * particle.nextVelocity
    #         particle.nextVelocity = frictionTerm + noiseTerm/particle.mass

    # def integrateOneSymplecticEuler(self, particleList):
    #     for i, particle in enumerate(particleList):
    #         conditionedVars = self.getConditionedVars(particle)
    #         force = self.calculateForce(particleList, i)
    #         eta = self.kBT / particle.D  # friction coefficient
    #         frictionTerm = -(self.dt * eta / particle.mass) * particle.nextVelocity
    #         noiseTerm = self.noiseSampler.sample(conditionedVars)
    #         self.prevNoiseTerm = 1.0 * noiseTerm
    #         particle.nextVelocity = particle.nextVelocity + self.dt * (force / particle.mass) + \
    #                                 frictionTerm + noiseTerm/particle.mass
    #     for particle in particleList:
    #         particle.nextPosition = particle.nextPosition + self.dt * particle.nextVelocity
