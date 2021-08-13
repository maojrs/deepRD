import numpy as np
from ..reactionModel import reactionModel

class schlogl(reactionModel):
    
    def __init__(self, X):
        # inherit all methods from parent class
        super().__init__()

        # Define default initial conditions and names
        self.X = np.array([X])
        self.names = ['gene', 'gene*', 'mRNA', 'protein']

        # Define base paramters, based on ODE model and data generation parameters
        self.concA = 10.0
        self.concB = 20.0
        self.k1 = 6.0
        self.k2 = 1.0
        self.k3 = 230.0
        self.k4 = 1000.0
        self.vol = 8.0
        self.nreactions = 4
        self.reactionVectors = np.zeros([self.nreactions, len(self.X)])
        self.propensities = np.zeros(self.nreactions)
        self.populateReactionVectors()
        self.updatePropensities()

        # Define default simulation parameters
        self.setSimulationParameters(dt = 0.0001, stride = 1, tfinal = 1000, datasize = 2560)
        
    def setModelParameters(self, concA, concB, k1, k2, k3, k4, vol):
        self.concA = concA
        self.concB = concB
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.vol = vol

    def populateReactionVectors(self):
        self.reactionVectors[0] = [1]   # A + 2X -k1-> 3X
        self.reactionVectors[1] = [-1]  # 3X -k2-> A + 2X
        self.reactionVectors[2] = [1]   # B -k3-> X
        self.reactionVectors[3] = [-1]  # X -k4-> B

    def updatePropensities(self):
        x = self.X[0]
        self.propensities[0] = self.k1 * self.concA * x * (x-1)/self.vol
        self.propensities[1] = self.k2 * x * (x-1) * (x-2)/ (self.vol**2)
        self.propensities[2] = self.k3 * self.concB * self.vol
        self.propensities[3] = self.k4 * x


    '''
    Additional functions specific to the Schlogl model
    '''

    def lambdan(self):
        '''Define CME birth rate '''
        x = self.X[0]
        return self.concA*self.k1*x*(x-1)/self.vol + self.concB*self.k3*self.vol

    def mun(self):
        '''Define CME death rate '''
        x = self.X[0]
        return self.k2*x*(x-1)*(x-2)/self.vol**2 + x*self.k4

    def ODE_func(self, x,k1,k2,k3,k4,a,b):
        '''Define ODE (LMA) function to explore parameters '''
        return k1*a*x**2 - k2*x**3- k4*x + k3*b

    def steadystate_solution(self, n):
        '''Calculate nonequilibrium steady state solution'''
        result = 1.0
        for i in range(n):
            result = result*(self.lambdan(i)/self.mun(i+1))
        return result

