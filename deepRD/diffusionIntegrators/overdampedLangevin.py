import numpy as np
import sys
from .diffusionIntegrator import diffusionIntegrator

class overdampedLangevin(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a standard Brownian particle (overdamped Lanegvin regime)
    '''

    def __init__(self, dt, stride, tfinal, boxsize = None, boundary = 'periodic'):
        # inherit all methods from parent class
        kBT = 1
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary)


    def integrateOne(self, particleList):
        nextPositions = [None] * len(particleList)
        for i, particle in enumerate(particleList):
            sigma = np.sqrt(2 * self.dt * particle.D)
            nextPositions[i] = particle.position + sigma * np.random.normal(0., 1, particle.dimension)
        return nextPositions

    def propagate(self, particleList):
        percentage_resolution = self.tfinal / 100.0
        time_for_percentage = - 1 * percentage_resolution
        # Begins Euler-Maruyama algorithm
        Xtraj = [particleList.positions]
        times = np.zeros(self.timesteps + 1)
        for i in range(self.timesteps):
            nextPositions = self.integrateOne(particleList)
            # Update variables
            Xtraj.append(nextPositions)
            particleList.positions = nextPositions
            # Enforce boundary conditions
            self.enforceBoundary(particleList)
            times[i + 1] = times[i] + self.dt
            # Print integration percentage
            if (times[i] - time_for_percentage >= percentage_resolution):
                time_for_percentage = 1 * times[i]
                sys.stdout.write("Percentage complete " + str(round(100 * times[i] / self.tfinal, 1)) + "% " + "\r")
        sys.stdout.write("Percentage complete 100% \r")
        return times, np.array(Xtraj)