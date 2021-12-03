import numpy as np

class particle:
    '''
    Main class for particles, takes initial position r0 and diffusion coefficient D as input.
    Can be used as a parent class for more complex particles (e.g. particles/bodies with orientation)
    '''
    def __init__(self, position, D = 0, velocity = [0,0,0], mass = None, state = None):
        self.position = np.array(position)
        self.D = D
        self.velocity = np.array(velocity)
        self.mass = mass
        self.state = state
        self.dimension = len(position)
        self.nextPosition = np.array(position)
        self.nextVelocity = np.array(velocity)


class particleList:
    '''
    Class to store a particle list along some useful routines to operate on particle lists. The input
    is a list of particles given by the particle class.
    '''
    def __init__(self, particleList):
        self.particleList = particleList
        self.numParticles = len(particleList)
        self.dimension = particleList[0].dimension

    def __getitem__(self, i):
        return self.particleList[i]

    def __setitem__(self, i, value):
        self.particleList[i] = value

    def __len__(self):
        return len(self.particleList)

    @property
    def positions(self):
        positions = [particle.position for particle in self.particleList]
        return np.array(positions)

    @positions.setter
    def positions(self, newPositions):
        for i, particle in enumerate(self.particleList):
            particle.position = newPositions[i]

    @property
    def velocities(self):
        velocities = [particle.velocity for particle in self.particleList]
        return np.array(velocities)

    @velocities.setter
    def velocities(self, newVelocities):
        for i, particle in enumerate(self.particleList):
            particle.velocity = newVelocities[i]

    def updatePositions(self):
        for particle in self.particleList:
            particle.position = 1.0 * particle.nextPosition

    def updateVelocities(self):
        for particle in self.particleList:
            particle.velocity = 1.0 * particle.nextVelocity

    def updatePositionsVelocities(self):
        for particle in self.particleList:
            particle.position = 1.0 * particle.nextPosition
            particle.velocity = 1.0 * particle.nextVelocity

    def resetNextPositionsVelocities(self):
        for particle in self.particleList:
            particle.nextPosition = 1.0 * particle.position
            particle.nextVelocity = 1.0 * particle.velocity