import numpy as np
import sys
from .diffusionIntegrator import diffusionIntegrator

class overdampedLangevin(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a standard Brownian particle (overdamped Lanegvin regime)
    '''

    def __init__(self, dt, stride, tfinal, boxsize = None, boundary = 'periodic', equilibrationSteps = 0):
        # inherit all methods from parent class
        kBT = 1
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary, equilibrationSteps)


    def integrateOne(self, particleList):
        self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            sigma = np.sqrt(2 * self.dt * particle.D)
            force = self.forceField[i]
            particle.nextPosition = particle.nextPosition + force * self.dt * particle.D / self.kBT + \
                               sigma * np.random.normal(0., 1, particle.dimension)
        self.enforceBoundary(particleList)
        particleList.updatePositions()

    def propagate(self, particleList, showProgress = False):
        if self.firstRun:
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