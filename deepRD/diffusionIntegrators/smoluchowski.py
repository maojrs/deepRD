import numpy as np
import sys
import deepRD
import deepRD.tools.particleTools as particleTools
from .diffusionIntegrator import diffusionIntegrator

class smoluchowski(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a standard Brownian particle (overdamped Lanegvin regime)
    '''

    def __init__(self, dt, stride, tfinal, boxsize = None, boundary = 'reflective', equilibrationSteps = 0,
                 sigma = 0.0, R = 1.0, kappa = 1.0, deltar = 0.1, cR = 1.0):
        '''
        inherit all methods from parent class
        Boundary delimited by [sigma,R], reactive boundary at sigma, boundary in contact with reservoir at R,
        deltax is the width of boundary layer to interact with reservoir and cR the concentration of the reservoir.
        '''
        kBT = 1
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary, equilibrationSteps)
        self.sigma = sigma
        self.R = R
        self.kappa = kappa
        self.deltar = deltar
        self.cR = cR
        self.injectionRate = 0.0
        self.D = 0.0

    def setIntrinsicReactionRate(self, kappa):
        self.kappa = kappa

    def setInjectionRate(self, reservoirConcentation):
        perParticleJumpRate = (self.D / (self.deltar ** 2)) * (1 - self.deltar / self.R)
        self.inectionRate = reservoirConcentation * perParticleJumpRate

    def injectParticles(self, particleList, deltat):
        # Count number of reactions with several Poisson rate with the corresponding propensity
        numInjectedParticles = np.random.poisson(self.inectionRate * deltat)
        for i in range(numInjectedParticles):
            position = particleTools.uniformShell(R-self.deltar, R)
            particle = deepRD.particle(position, D = self.D)
            particleList.addParticle(particle)

    def diffuseParticles(self, particleList, deltat):
        self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            if particle.active:
                sigma = np.sqrt(2 * deltat * particle.D)
                force = self.forceField[i]
                particle.nextPosition = particle.nextPosition + force * deltat * particle.D / self.kBT + \
                                   sigma * np.random.normal(0., 1, particle.dimension)

    def enforceBoundary(self, particleList, currentOrNextOverride = None):
        for i, particle in enumerate(particleList):
            rr = np.linalg.norm(particle.nextPosition)
            # If particle reached reactive boundary, either react or reflect
            if rr <= self.sigma:
                # Gillespie time of reaction check
                r1 = np.random.rand()
                lagtime = np.log(1.0 / r1) / self.kappa
                if lagtime <= self.deltat:
                    particleList.deactivateParticle(i)
                else:
                    # Reflection BC
                    dr = self.sigma - rr
                    particle.nextPosition = particleTools.uniformShell(self.sigma, self.sigma + dr)
            # Deactivate particles leaving into Reservoir (r>R)
            elif rr > self.R:
                particleList.deactivateParticle(i)

    def integrateOne(self, particleList):

        # Begin splitting algorithm
        self.injectParticles(particleList, self.dt/2.0)

        self.diffuseParticles(particleList, self.dt)

        # Enforce reactive boundaries and reservoir boundary
        self.enforceBoundary(particleList)

        self.injectParticles(particleList, self.dt/2.0)

        particleList.updatePositions()

        # Could be run every several timesteps
        particleList.removeInactiveParticles()

    def propagate(self, particleList, showProgress = False):
        if self.firstRun:
            # Set injectionRate, assuming all particles have sam diffusion coefficient
            self.D = particleList[0].D
            self.setInjectionRate(self.cR)
            self.prepareSimulation(particleList)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
        # Begins integration
        time = 0.0
        xTraj = [particleList.positions]
        tTraj = [time]
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                xTraj.append(particleList.positions)
                tTraj.append(time)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time/ self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        return np.array(tTraj), np.array(xTraj)