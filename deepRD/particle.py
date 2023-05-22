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
        self.active = True
        self.dimension = len(self.position)
        self.nextPosition = np.array(position)
        self.nextVelocity = np.array(velocity)
        self.aux1 = None
        self.aux2 = None
        self.aux3 = None


class particleList:
    '''
    Class to store a particle list along some useful routines to operate on particle lists. The input
    is a list of particles given by the particle class.
    '''
    def __init__(self, particleList):
        self.particleList = particleList
        self.numParticles = len(particleList)
        if len(particleList) > 0:
            self.dimension = particleList[0].dimension
        else:
            self.dimension = None
        self.neighbor_list = [[] for i in range(self.numParticles)]
        self.inactiveIndexList = []

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

    @property
    def aux1List(self):
        aux1 = [particle.aux1 for particle in self.particleList]
        return np.array(aux1)

    @property
    def aux2List(self):
        aux2 = [particle.aux2 for particle in self.particleList]
        return np.array(aux2)

    @property
    def aux3List(self):
        aux3 = [particle.aux3 for particle in self.particleList]
        return np.array(aux3)

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

    def updatePositionsVelocitiesIndex(self,index):
        for particle in self.particleList:
            particle.position[index] = 1.0 * particle.nextPosition[index]
            particle.velocity[index] = 1.0 * particle.nextVelocity[index]

    def resetNextPositionsVelocities(self):
        for particle in self.particleList:
            particle.nextPosition = 1.0 * particle.position
            particle.nextVelocity = 1.0 * particle.velocity

    def addParticle(self, particle):
        if self.dimension == None:
            self.dimension = particle.dimension
        self.particleList.append(particle)
        self.numParticles += 1

    def deleteParticle(self, index):
        self.particleList.pop(index)
        self.numParticles -= 1

    def deactivateParticle(self,indexlist):
        if type(indexlist) is list:
            self.inactiveIndexList = self.inactiveIndexList + indexlist
            for i in indexlist:
                self.particleList[i].active = False
        else:
            self.inactiveIndexList = self.inactiveIndexList + [indexlist]
            self.particleList[indexlist].active = False

    def removeInactiveParticles(self):
        self.inactiveIndexList = sorted(set(self.inactiveIndexList))
        for index in sorted(self.inactiveIndexList, reverse=True):
            self.particleList.pop(index)
        self.numParticles = len(self.particleList)
        self.inactiveIndexList = []

    def countParticles(self):
        numParticles = 0
        for particle in self.particleList:
            if particle.active:
                numParticles += 1
        self.numParticles = numParticles
        return numParticles
