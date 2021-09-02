import numpy as np
from .diffusionIntegrator import diffusionIntegrator

class overdampedLangevin(diffusionIntegrator):

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
                print("Percentage complete ", round(100 * times[i] / self.tfinal, 1), "%           ", end="\r")
        print("Percentage complete 100%       ", end="\r")
        return times, np.array(Xtraj)