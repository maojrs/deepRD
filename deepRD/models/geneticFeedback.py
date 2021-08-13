import numpy as np

class geneticFeedback:
    
    def __init__(self, G, Gstar, M, P):
        # Define default initial conditions
        self.X = np.array([G, Gstar, M, P])

        # Define default model paramters
        self.rhou = 2.5
        self.rhob = 10**(-1)
        self.sigmau = 10**5
        self.sigmab = 10**3
        self.dm = 10.0
        self.dp = 1.0
        self.k = 1.0
        self.vol = 10.0
        self.nreactions = 7
        self.lambdas = np.zeros(self.nreactions)
        self.reactionVectors = np.zeros([self.nreactions, len(self.X)])
        self.updatePropensities()
        self.populateReactionVectors()

        # Define default simulation parameters
        self.setDataParameters(0.0001, 10, 1000, 2560)
        self.dt = 0.0001
        self.stride = 1
        self.tfinal = 1000
        self.datasize = 2560
        self.filename =  "data/genetic_feedback_data_vol" + str(self.vol) + "_ndata" + str(self.datasize) + ".dat"
        self.timesteps = int(self.tfinal/self.dt)

    def setModelParameters(self, rhou, rhob, sigmau, sigmab, dm, dp, k, vol):
        self.rhou = rhou
        self.rhob = rhob
        self.sigmau = sigmau
        self.sigmab = sigmab
        self.dm = dm
        self.dp = dp
        self.k = k
        self.vol = vol


    def setDataParameters(self, dt, stride, tfinal, datasize):
        self.dt = dt
        self.stride = stride
        self.tfinal = tfinal
        self.datasize = datasize
        self.timesteps = int(tfinal/dt)

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
        self.lambdas[0] = self.rhou * G
        self.lambdas[1] = self.rhob * Gstar
        self.lambdas[2] = self.k * M
        self.lambdas[3] = self.sigmab * G * P / self.vol
        self.lambdas[4] = self.sigmau * Gstar
        self.lambdas[5] = self.dm * M
        self.lambdas[6] = self.dp * P
