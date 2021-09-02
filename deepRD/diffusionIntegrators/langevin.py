import numpy as np
from .diffusionIntegrator import diffusionIntegrator

class langevin(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle (full Langevin dynamics)
    '''

    def __init__(self, dt, stride, tfinal, kBT=1, boxsize = None, boundary = 'periodic'):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary)

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

    def propagate(self, particleList):
        percentage_resolution = self.tfinal / 100.0
        time_for_percentage = - 1 * percentage_resolution
        # Begins Euler-Maruyama algorithm
        Xtraj = [particleList.positions]
        Vtraj = [particleList.velocities]
        times = np.zeros(self.timesteps + 1)
        for i in range(self.timesteps):
            nextPositions, nextVelocities = self.integrateOne(particleList)
            # Update variables
            Xtraj.append(nextPositions)
            Vtraj.append(nextVelocities)
            particleList.positions = nextPositions
            particleList.positions = nextVelocities
            # Enforce boundary conditions
            self.enforceBoundary(particleList)
            times[i + 1] = times[i] + self.dt
            # Print integration percentage
            if (times[i] - time_for_percentage >= percentage_resolution):
                time_for_percentage = 1 * times[i]
                print("Percentage complete ", round(100 * times[i] / self.tfinal, 1), "%           ", end="\r")
        print("Percentage complete 100%       ", end="\r")
        return times, np.array(Xtraj), np.array(Vtraj)

    def integrateA(self, position, velocity):
        '''Integrates position half a time step given velocity term'''
        position = position + self.dt/2 * velocity
        return position, velocity

    def integrateB(self, position, velocity, mass):
        '''Integrates velocity half a time step given potential or force term. Note this does
        nothing in its current implementation. Left force term here for future more general
        implementations. '''
        force = 0.0 * velocity
        velocity = velocity + self.dt / 2 * force / mass
        return position, velocity

    def integrateO(self, position, velocity, D, mass, dimension):
        '''Integrates velocity full time step given friction and noise term'''
        eta = self.kBT / D # friction coefficient
        xi = np.sqrt(self.kBT * (1 - np.exp(-2 * eta * self.dt)))
        velocity = (np.exp(-self.dt * eta) / mass) * velocity + xi / mass * np.random.normal(0., 1, dimension)
        return position, velocity

