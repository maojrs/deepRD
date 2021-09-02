import numpy as np

class particle:
    '''
    Main class for particles, takes initial position r0 and diffusion coefficient D as input.
    Can be used as a parent class for more complex particles (e.g. particles/bodies with orientation)
    '''
    def __init__(self, position, D, state = 0):
        self.position = np.array(position)
        self.D = D
        self.state = state
        self.dimension = len(position)


class particleList:
    '''
    Class to store a particle list along some useful routines to operate on particle lists. The input
    is a list of particles given by the particle class.
    '''
    def __init__(self, particleList):
        self.particleList = particleList
        self.numParticles = len(particleList)

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