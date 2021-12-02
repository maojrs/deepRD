import numpy as np
import sys
from .diffusionIntegrator import diffusionIntegrator

class langevin(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle (full Langevin dynamics). This
    is the parent class, which implement the BAOAB method by default. Other methods are available through
    the child classes
    '''

    def __init__(self, dt, stride, tfinal, Gamma, kBT=1, boxsize = None, boundary = 'periodic', equilibrationSteps = 0):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary)
        self.Gamma = Gamma # friction coefficient in units of mass/time
        self.equilibrationSteps = equilibrationSteps

    def integrateOne(self, particleList):
        ''' Integrates one time step using the BAOAB algorithm '''
        self.integrateB(particleList)
        self.integrateA(particleList)
        self.integrateO(particleList)
        self.integrateA(particleList)
        self.enforceBoundary(particleList)
        particleList.updatePositions()
        self.integrateB(particleList)
        particleList.updateVelocities()

    def propagate(self, particleList, showProgress = False):
        #percentage_resolution = self.tfinal / 100.0
        #time_for_percentage = - 1 * percentage_resolution
        # Begin equilbration
        particleList.resetNextPositionsVelocities()
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
        # Begins integration
        time = 0.0
        xTraj = [particleList.positions]
        vTraj = [particleList.velocities]
        tTraj = [time]
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                xTraj.append(particleList.positions)
                vTraj.append(particleList.velocities)
                tTraj.append(time)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time/ self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        return np.array(tTraj), np.array(xTraj), np.array(vTraj)

    def integrateA(self, particleList):
        '''Integrates position half a time step given velocity term'''
        for particle in particleList:
            particle.nextPosition = particle.nextPosition + self.dt/2 * particle.nextVelocity

    def integrateB(self, particleList):
        '''Integrates velocity half a time step given potential or force term. Note this does
        nothing in its current implementation.  '''
        forceField = self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            force = forceField[i]
            particle.nextVelocity = particle.nextVelocity + (self.dt / 2) * (force / particle.mass)

    def integrateO(self, particleList):
        '''Integrates velocity full time step given friction and noise term'''
        for particle in particleList:
            xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * self.dt/particle.mass)))
            frictionTerm = np.exp(-self.dt * self.Gamma/particle.mass) * particle.nextVelocity
            particle.nextVelocity = frictionTerm + xi / particle.mass * np.random.normal(0., 1, particle.dimension)


class langevinBAOAB(langevin):
    '''
    Same as langevin class, but method stated explicitly. Uses BAOAB method for integration.
    '''
    pass


class langevinABOBA(langevin):
    '''
    Same as langevin class, but method stated explicitly. Uses ABOBA method for integration.
    '''
    def integrateOne(self, particleList):
        ''' Integrates one time step using the ABOBA algorithm '''
        self.integrateA(particleList)
        particleList.updatePositions()
        self.integrateB(particleList)
        self.integrateO(particleList)
        self.integrateB(particleList)
        self.integrateA(particleList)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()

class langevinSemiImplicitEuler(langevin):
    '''
    Same as langevin class, but method stated explicitly. Uses the semi-implicit Euler method for integration.
    '''
    def integrateOne(self, particleList):
        ''' Integrates one time step using the semi-eimplicit Euler algorithm '''
        forceField = self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            force = forceField[i]
            xi = np.sqrt(2 * self.kBT * self.Gamma * self.dt)
            frictionTerm = -(self.dt * self.Gamma / particle.mass) * particle.nextVelocity
            particle.nextVelocity = particle.nextVelocity + self.dt * (force / particle.mass) + \
                                    frictionTerm + (xi / particle.mass) * np.random.normal(0., 1, particle.dimension)
        for particle in particleList:
            particle.nextPosition = particle.nextPosition + self.dt * particle.nextVelocity
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()


