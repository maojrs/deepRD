import numpy as np
import sys
from .langevinNoiseSampler import langevinNoiseSampler

class langevinInteractionSampler(langevinNoiseSampler):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle following the Langevin equation,
    where only pair interaction terms in the velocity equation are sampled from an alternate data-based
    model. Takes same input as Langevin integrator with the addition of a noise sampler for the interaction term.
    The variable particle.aux1 and particle.aux2 store the values of the interaction terms r for
    previous time steps, t-dt and t-2dt, respectively. The variable particle.aux3 stores the noise terms.
    It is the same as langevinNoiseSampler, but only uses the data sampler to sample pair interactions; the noise
    is integrated explicitly.
    '''

    def integrateBOB(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the interaction Sampler from the
        external potential and noise term. '''
        for i, particle in enumerate(particleList):
            # Calculate friction term and external potential term
            expterm = np.exp(-dt * self.Gamma / particle.mass)
            frictionForceTerm = particle.nextVelocity * expterm
            frictionForceTerm += (1 + expterm) * self.forceField[i] * dt/(2*particle.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars(particle)
            interactionTerm = self.noiseSampler.sample(conditionedVars)

            xi = np.sqrt((self.kBT / particle.mass) * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            noiseTerm = xi  * np.random.normal(0., 1, particle.dimension)

            particle.aux2 = 1.0 * particle.aux1
            particle.aux1 = 1.0 * interactionTerm
            particle.aux3 = 1.0 * noiseTerm
            particle.nextVelocity = frictionForceTerm + interactionTerm + noiseTerm