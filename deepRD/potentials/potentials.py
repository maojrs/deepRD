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

    def calculateForce(self, particle, whichPosition = 'current'):
        '''
        'Abstract' method used to calculate forces of a given potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        raise NotImplementedError("Please Implement calculateForce method for this potential")


class pairPotential:
    '''
    Parent(abstract) class for external potentials
    '''

    def evaluate(self, particle1, particle2):
        '''
        'Abstract' method used to calculate forces of a given potential.
        '''
        raise NotImplementedError("Please Implement evaluate method for this potential")

    def calculateForce(self, particle1, particle2, whichPosition = 'current'):
        '''
        'Abstract' method used to calculate forces of a given potential. If whichPosition == "current", calculate
        using current position, if "next, calculate it using the next position."
        '''
        raise NotImplementedError("Please Implement calculateForce method for this potential")