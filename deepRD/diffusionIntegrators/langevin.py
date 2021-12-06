import numpy as np
import sys
from .diffusionIntegrator import diffusionIntegrator

class langevin(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle (full Langevin dynamics). This
    is the parent class, which implement the BAOAB method by default. Other methods are available through
    the child classes
    '''

    def __init__(self, dt, stride, tfinal, Gamma, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary,equilibrationSteps)
        self.Gamma = Gamma # friction coefficient in units of mass/time

    def integrateOne(self, particleList):
        ''' Integrates one time step using the BAOAB algorithm '''
        self.integrateB(particleList, self.dt/2.0)
        self.integrateA(particleList, self.dt/2.0)
        self.integrateO(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList, 'next')
        self.calculateForceField(particleList, 'next')
        self.integrateB(particleList, self.dt/2.0)
        particleList.updatePositionsVelocities()

    def propagate(self, particleList, showProgress = False):
        if self.firstRun:
            self.calculateForceField(particleList, 'next')
            self.firstRun = False
        # Equilbration runs
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

    def integrateA(self, particleList, dt):
        '''Integrates position half a time step given velocity term'''
        for particle in particleList:
            particle.nextPosition = particle.nextPosition + dt * particle.nextVelocity

    def integrateB(self, particleList, dt):
        '''Integrates velocity half a time step given potential or force term. Note this does
        nothing in its current implementation.  '''
        for i, particle in enumerate(particleList):
            force = self.forceField[i]
            particle.nextVelocity = particle.nextVelocity + (dt / 2) * (force / particle.mass)

    def integrateO(self, particleList, dt):
        '''Integrates velocity full time step given friction and noise term'''
        for particle in particleList:
            xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt/particle.mass)))
            frictionTerm = np.exp(-dt * self.Gamma/particle.mass) * particle.nextVelocity
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
        self.integrateA(particleList, self.dt/2.0)
        self.calculateForceField(particleList, 'next')
        self.integrateB(particleList, self.dt/2.0)
        self.integrateO(particleList, self.dt)
        self.integrateB(particleList, self.dt/2.0)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList, 'next')
        particleList.updatePositionsVelocities()

class langevinSemiImplicitEuler(langevin):
    '''
    Same as langevin class, but method stated explicitly. Uses the semi-implicit Euler method for integration.
    '''
    def integrateOne(self, particleList):
        ''' Integrates one time step using the semi-eimplicit Euler algorithm '''
        self.calculateForceField(particleList, 'next')
        self.integrateO(particleList, self.dt)
        self.integrateB(particleList, self.dt)
        self.integrateA(particleList, self.dt)
        self.enforceBoundary(particleList, 'next')
        particleList.updatePositionsVelocities()


