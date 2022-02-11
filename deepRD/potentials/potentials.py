'''
Parent class of external potentials. The potentials' functions usually take as input one or two
particles, depending if it is an external potential or a pair potential.
'''

class externalPotential:
    '''
    Parent(abstract) class for external potentials
    '''
    def evaluate(self, particle):
        '''
        'Abstract' method used to calculate forces of a given potential.
        '''
        raise NotImplementedError("Please Implement evaluate method for this potential")

    def calculateForce(self, particle, currentOrNext = 'current'):
        '''
        'Abstract' method used to calculate forces of a given potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        raise NotImplementedError("Please Implement calculateForce method for this potential")


class pairPotential:
    '''
    Parent(abstract) class for external potentials
    '''
    def __init__(self):
        self.boxsize = None
        self.boundary = 'periodic'

    def evaluate(self, particle1, particle2):
        '''
        'Abstract' method used to calculate forces of a given potential.
        '''
        raise NotImplementedError("Please Implement evaluate method for this potential")

    def calculateForce(self, particle1, particle2, currentOrNext = 'current'):
        '''
        'Abstract' method used to calculate forces of a given potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        raise NotImplementedError("Please Implement calculateForce method for this potential")

    def relativePosition(self, pos1, pos2):
        p1periodic = 1.0 * pos1
        if (self.boundary == "periodic" and self.boxsize != None):
            for i in range(3):
                if (pos2[i] - pos1[i]) > 0.5 * self.boxsize[i]:
                    p1Periodic[i] += self.boxsize[i]
                if (pos2[i] - pos1[i]) < -0.5 * self.boxsize[i]:
                    p1Periodic[i] -= self.boxsize[i]
        return pos2 - p1periodic
