import numpy as np
import sys
import deepRD
import deepRD.tools.particleTools as particleTools
from .diffusionIntegrator import diffusionIntegrator
from ..reactionModels import reservoir
from ..reactionIntegrators import tauleap

class smoluchowski(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a standard Brownian particle (overdamped Lanegvin regime)
    '''

    def __init__(self, dt, stride, tfinal, D = 1.0, kappa = 1.0, sigma = 0.0, R = 1.0, deltar = 0.1, cR = 1.0,
                 equilibrationSteps = 0):
        '''
        inherit all methods from parent class
        Boundary delimited by [sigma,R], reactive boundary at sigma, boundary in contact with reservoir at R,
        deltar is the width of boundary layer to interact with reservoir and cR the concentration of the reservoir.
        refreshTimeStep is the number of timesteps that the particleList is refreshed (due to deactivated particles).
        '''
        kBT = 1
        super().__init__(dt, stride, tfinal, kBT, None, None, equilibrationSteps)
        self.D = D
        self.kappa = kappa
        self.sigma = sigma
        self.R = R
        self.deltar = deltar
        self.cR = cR
        self.nR = None
        self.injectionRate = 0.0
        self.reservoirModel = None

        self.tauleapSubsteps = 10
        self.refreshTimeSteps = 50
        self.tauleapIntegrator = tauleap(self.dt)

    def setIntrinsicReactionRate(self, kappa):
        self.kappa = kappa

    def setReservoirModel(self, reservoirConcentation):
        Rn = self.R - self.deltar/2.0
        perParticleJumpRate = (self.D / (self.deltar ** 2)) * (1 - self.deltar / Rn)
        reservoirVolume = 4 * np.pi * ((self.R + self.deltar)**3 - self.R**3)/3.0
        self.reservoirModel =  reservoir(perParticleJumpRate, reservoirVolume, reservoirConcentation)

    # def setInjectionRate(self, reservoirConcentation):
    #     Rn = self.R - self.deltar/2.0
    #     perParticleJumpRate = (self.D / (self.deltar ** 2)) * (1 - self.deltar / Rn)
    #     reservoirVolume = 4 * np.pi * ((self.R + self.deltar)**3 - self.R**3)/3.0
    #     self.nR = reservoirVolume * reservoirConcentation
    #     self.injectionRate = self.nR * perParticleJumpRate

    def injectParticles(self, particleList, deltat):
        # Count number of reactions by running a tau-leap approximation
        #numInjectedParticles = np.random.poisson(self.injectionRate * deltat)
        self.reservoirModel.X = np.array([0])
        self.reservoirModel.updatePropensities()
        X = self.tauleapIntegrator.integrateMany(self.reservoirModel, self.tauleapSubsteps)
        numInjectedParticles = np.int(X[0])
        for i in range(numInjectedParticles):
            position = particleTools.uniformShell(self.R - self.deltar, self.R)
            particle = deepRD.particle(position, D = self.D)
            particleList.addParticle(particle)

    def diffuseParticles(self, particleList, deltat):
        self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            if particle.active:
                sigma = np.sqrt(2 * deltat * particle.D)
                force = self.forceField[i]
                particle.nextPosition = particle.position + force * deltat * particle.D / self.kBT + \
                                   sigma * np.random.normal(0., 1, particle.dimension)

    def enforceBoundary(self, particleList, currentOrNextOverride = None):
        for i, particle in enumerate(particleList):
            if particle.active:
                rr = np.linalg.norm(particle.nextPosition)
                # If particle reached reactive boundary, either react or reflect
                if rr <= self.sigma:
                    # Gillespie time of reaction check
                    r1 = np.random.rand()
                    lagtime = np.log(1.0 / r1) / self.kappa
                    if lagtime <= self.dt:
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

    def propagate(self, particleList, showProgress = False):
        if self.firstRun:
            # Set injectionRate, assumes all particles have same diffusion coefficient
            if len(particleList) > 1 and self.D != particleList[0].D:
                raise NotImplementedError("Please check diffusion coefficient of integrator matches particles")
            self.setReservoirModel(self.cR)
            self.prepareSimulation(particleList)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
            if i%self.refreshTimeSteps == 0:
                particleList.removeInactiveParticles()
        time = 0.0
        xTraj = [particleList.activePositions]
        tTraj = [time]
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            if i%self.refreshTimeSteps == 0 or i == self.timesteps - 1:
                particleList.removeInactiveParticles()
                #sys.stdout.write(str(int(100*(i+1)/self.timesteps)) + '% Number of particles: ' + str(particleList.countParticles()) + "\r")
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                xTraj.append(particleList.activePositions)
                tTraj.append(time)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time/ self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        return np.array(tTraj), xTraj