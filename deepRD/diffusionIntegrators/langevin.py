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
        ''' Integrates one time step '''
        # Integrate BAOAB
        self.integrateB(particleList)
        self.integrateA(particleList)
        self.integrateO(particleList)
        self.integrateA(particleList)
        self.integrateB(particleList)
        particleList.updatePositionsVelocities()

    def propagate(self, particleList):
        percentage_resolution = self.tfinal / 100.0
        time_for_percentage = - 1 * percentage_resolution
        # Begins Euler-Maruyama algorithm
        Xtraj = [particleList.positions]
        Vtraj = [particleList.velocities]
        times = np.zeros(self.timesteps + 1)
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            # Update variables
            Xtraj.append(particleList.positions)
            Vtraj.append(particleList.velocities)
            # Enforce boundary conditions
            self.enforceBoundary(particleList)
            times[i + 1] = times[i] + self.dt
            # Print integration percentage
            if (times[i] - time_for_percentage >= percentage_resolution):
                time_for_percentage = 1 * times[i]
                print("Percentage complete ", round(100 * times[i] / self.tfinal, 1), "%           ", end="\r")
        print("Percentage complete 100%       ", end="\r")
        return times, np.array(Xtraj), np.array(Vtraj)

    def integrateA(self, particleList):
        '''Integrates position half a time step given velocity term'''
        for particle in particleList:
            particle.nextPosition = particle.nextPosition + self.dt/2 * particle.nextVelocity

    def integrateB(self, particleList):
        '''Integrates velocity half a time step given potential or force term. Note this does
        nothing in its current implementation.  '''
        for i, particle in enumerate(particleList):
            force = self.calculateForce(particleList, i)
            particle.nextVelocity = particle.nextVelocity + (self.dt / 2) * (force / particle.mass)

    def integrateO(self, particleList):
        '''Integrates velocity full time step given friction and noise term'''
        for particle in particleList:
            eta = self.kBT / particle.D # friction coefficient
            xi = np.sqrt(self.kBT * (1 - np.exp(-2 * eta * self.dt)))
            frictionTerm = (np.exp(-self.dt * eta) / particle.mass) * particle.nextVelocity
            particle.nextVelocity = frictionTerm + xi / particle.mass * np.random.normal(0., 1, particle.dimension)

    def calculateForce(self, particleList, particleIndex):
        ''' Default force term is zero. General force calculations can be implemented here. It should
        output the force exterted into particle indexed by particleIndex'''
        return 0.0 * particleList[0].velocity

