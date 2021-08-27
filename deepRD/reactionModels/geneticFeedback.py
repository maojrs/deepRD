import numpy as np
from .reactionModel import reactionModel
from ..integrators import gillespie

class geneticFeedback(reactionModel):
    
    def __init__(self, G, Gstar, M, P):
        # inherit all methods from parent class
        super().__init__()

        # Define default initial conditions and names
        self.X = np.array([G, Gstar, M, P])
        self.names = ['gene', 'gene*', 'mRNA', 'protein']

        # Define default model parameters
        self.rhou = 2.5
        self.rhob = 10**(-1)
        self.sigmau = 10**5
        self.sigmab = 10**3
        self.dm = 10.0
        self.dp = 1.0
        self.k = 1.0
        self.volume = 10.0
        self.nreactions = 7
        self.reactionVectors = np.zeros([self.nreactions, len(self.X)])
        self.propensities = np.zeros(self.nreactions)
        self.populateReactionVectors()
        self.updatePropensities()

    def setModelParameters(self, rhou, rhob, sigmau, sigmab, dm, dp, k, volume):
        self.rhou = rhou
        self.rhob = rhob
        self.sigmau = sigmau
        self.sigmab = sigmab
        self.dm = dm
        self.dp = dp
        self.k = k
        self.volume = volume

    def populateReactionVectors(self):
        self.reactionVectors[0] = [0, 0, 1, 0]   # G -rhou-> G+M
        self.reactionVectors[1] = [0, 0, 1, 0]   # Gstar -rhob-> Gstar+M
        self.reactionVectors[2] = [0, 0, 0, 1]   # M -k-> M+P
        self.reactionVectors[3] = [-1, 1, 0, -1] # G + P -sigmab-> Gstar
        self.reactionVectors[4] = [1, -1, 0, 1]  # Gstar -sigmau-> G + P
        self.reactionVectors[5] = [0, 0, -1, 0]  # M -dm-> 0
        self.reactionVectors[6] = [0, 0, 0, -1]  # P -dp-> 0

    def updatePropensities(self):
        G = self.X[0]
        Gstar = self.X[1]
        M = self.X[2]
        P = self.X[3]
        self.propensities[0] = self.rhou * G
        self.propensities[1] = self.rhob * Gstar
        self.propensities[2] = self.k * M
        self.propensities[3] = self.sigmab * G * P / self.volume
        self.propensities[4] = self.sigmau * Gstar
        self.propensities[5] = self.dm * M
        self.propensities[6] = self.dp * P

    '''
    Additional functions specific to the genetic feedback model
    '''

    def oneCycleFPTs(self, numSamples, G, Gstar, M, P):
        '''
        Calculates first passage times (FPTs) from the first creation of an
        mRNA (M) to the second creation of an mRNA.
        '''
        integrator = gillespie(0,1)
        FPTs = np.zeros(numSamples)
        for i in range(numSamples):
            self.X = np.array([G, Gstar, M, P])
            self.updatePropensities()
            firstMRNAproduction = False
            secondMRNAproduction = False
            t = 0
            while(not secondMRNAproduction):
                # On iteration of Gillespie algorithm
                lagtime, nextX, reactionIndex = integrator.integrateOne(self, returnReactionIndex = True)

                if firstMRNAproduction == True:
                    t += lagtime
                    if reactionIndex == 0:
                        secondMRNAproduction = True

                if reactionIndex == 0:
                    firstMRNAproduction = True

                # Update variables
                self.X = nextX
                self.updatePropensities()
            FPTs[i] = t
        return FPTs