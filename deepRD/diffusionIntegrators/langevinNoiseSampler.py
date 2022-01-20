import numpy as np
import sys
from .langevin import langevin

class langevinNoiseSampler(langevin):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle following the Langevin equation,
    where the noise term and interaction terms in the velocity equation are sampled from an alternate data-based
    model. Takes same input as Langevin integrator with the addition of a noise sampler. The noise sampler
    takes certain input and outputs a corresponding noise term. The variable particle.aux1 and particle.aux2 store
    the values of the noise/interactions small term r for previous time steps, t-dt and t-2dt,
    respectively.
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'qi'):
        self.noiseSampler = noiseSampler
        self.conditionedOn = conditionedOn
        self.integratorType = "dataDrivenABOBA"
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, kBT, boxsize, boundary, equilibrationSteps)


    def prepareSimulation(self, particleList):
        '''
        Overrides parent method to set up additional variables at the start of the routine.
        See parent method for more details. The variable particle.aux1 and particle.aux2 store
        the values of the noise/interactions small term r for previous time steps, t-dt and t-2dt,
        respectively.
        '''
        for part in particleList:
            part.aux1 = np.zeros(particleList.dimension)
            part.aux2 = np.zeros(particleList.dimension)
        self.currentOrNext = 'next'
        self.calculateForceField(particleList)
        self.firstRun = False

    def getConditionedVars(self, particle):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return (particle.aux1)
        elif self.conditionedOn == 'ririm':
            return np.concatenate((particle.aux1, particle.aux2))
        elif self.conditionedOn == 'qi':
            return (particle.nextPosition)
        elif self.conditionedOn == 'qiri':
            return np.concatenate((particle.nextPosition, particle.aux1))
        elif self.conditionedOn == 'qiririm':
            return np.concatenate((particle.nextPosition, particle.aux1, particle.aux2))
        elif self.conditionedOn == 'pi':
            return (particle.nextVelocity)
        elif self.conditionedOn == 'piri':
            return np.concatenate((particle.nextVelocity, particle.aux1))
        elif self.conditionedOn == 'piririm':
            return np.concatenate((particle.nextVelocity, particle.aux1, particle.aux2))
        elif self.conditionedOn == 'qipi':
            return ((particle.nextPosition, particle.nextVelocity))
        elif self.conditionedOn == 'qipiri':
            return np.concatenate((particle.nextPosition, particle.nextVelocity, particle.aux1))
        elif self.conditionedOn == 'qipiririm':
            return np.concatenate((particle.nextPosition, particle.nextVelocity, particle.aux1, particle.aux2))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.integrateBOB(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()


    def integrateBOB(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        for i, particle in enumerate(particleList):
            # Calculate friction term and external potential term
            expterm = np.exp(-dt * self.Gamma / particle.mass)
            frictionForceTerm = particle.nextVelocity * expterm
            frictionForceTerm += (1 + expterm) * self.forceField[i] * dt/(2*particle.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars(particle)
            interactionNoiseTerm = self.noiseSampler.sample(conditionedVars)

            ## For testing and consistency.
            #xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            #interactionNoiseTerm = xi / particle.mass * np.random.normal(0., 1, particle.dimension)

            particle.aux2 = 1.0 * particle.aux1
            particle.aux1 = 1.0 * interactionNoiseTerm
            particle.nextVelocity = frictionForceTerm + interactionNoiseTerm

    def propagate(self, particleList, showProgress = False, outputAux = False):
        '''
        Same as Langevin propagator, but this one can output particle.aux1, corresponding to the 'r' variables.
        '''
        if self.firstRun:
            self.prepareSimulation(particleList)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
        # Begins integration
        time = 0.0
        tTraj = [time]
        xTraj = [particleList.positions]
        vTraj = [particleList.velocities]
        if outputAux:
            rTraj = [particleList.aux1List]
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                tTraj.append(time)
                xTraj.append(particleList.positions)
                vTraj.append(particleList.velocities)
                if outputAux:
                    rTraj.append(particleList.aux1List)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time/ self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        self.firstRun = True
        if outputAux:
            return np.array(tTraj), np.array(xTraj), np.array(vTraj), np.array(rTraj)
        else:
            return np.array(tTraj), np.array(xTraj), np.array(vTraj)
