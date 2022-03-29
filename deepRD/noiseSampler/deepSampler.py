import numpy as np
import random
import sys
import os
from ..tools import trajectoryTools
from .binning import binnedData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

'''
Class to sample auxiliary variables from deep neural network based model. It bins trajectory data r_i+1
obtained from a particle/molecular simulation, trains the Neural network to output most distributions
over bins given c_i conditioned variables and chooses most likely bin to sample from data. 
The classes here assume the trajs object consist of an array/list of trajectories, where trajectory 
consists of an array of points of the form: 
(time, positionx, positiony, positionz, velocityx, velocityy, velocityz, type, rx, ry, rz).  
'''

class deepSampler(binnedData):
    '''
    Class to sample auxiliary variables from deep neural network based model, (ri+1|qi,ri).
    '''

    def __init__(self, numbins = 100, lagTimesteps = 1, conditionOnPosition = False,
                 conditionOnVelocity = False, numAuxVars = 1):
        self.lagTimesteps = lagTimesteps
        self.conditionOnPosition = conditionOnPosition
        self.conditionOnVelocity = conditionOnVelocity
        self.numAuxVars = numAuxVars
        self.inputData = None
        self.targetData = None

        # Other important variables
        self.dimension = 3
        self.posIndex = 1  # Index of x position coordinate in trajectory files
        self.velIndex = 4  # Index of x velocity coordinate in trajectory files
        self.auxIndex = 8  # Position of x coordinate of r in trajectory files
        self.dataTree = None # dataTree structure to find nearest neighbors
        self.occupiedTuplesArray = None # array of tuples corresponding to occupied bins
        self.conditiondVarsDimension = self.conditionOnPosition * self.dimension + \
                                       self.conditionOnVelocity * self.dimension + self.numAuxVars

        if not isinstance(lagTimesteps, (int)):
            raise Exception('lagTimesteps should be an integer.')
        self.lagTimesteps = lagTimesteps # number of data timesteps to look back into data (integer)

        if isinstance(numbins, (list, tuple, np.ndarray)):
            if len(numbins) != self.dimension:
                raise Exception('Numbins should be a scalar or an array matching the chosen dimension')
            self.numbins = numbins
        else:
            self.numbins = [numbins]*self.dimension


    def adjustBoxAux(self, trajs, nsigma=-1):
        '''
        Calculate boxlimits of ri+1 auxiliary variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.auxBoxIndex correspond to the index of the x-coordinate of the first
        aux variable in the boxsize array; numAuxVars corresponds to the number of auxiliary
        variables, e.g. in ri+1|ri,ri-1, it would be two.
        '''
        if nsigma < 0:
            minvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 3])
            maxvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 3])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][self.auxIndex: self.auxIndex + 3]
                    for j in range(3):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [self.auxIndex,self.auxIndex + 3])
            stddev = trajectoryTools.calculateStdDev(trajs, [self.auxIndex,self.auxIndex + 3], mean)
            minvec = mean - nsigma*stddev
            maxvec = mean + nsigma*stddev
        # Adjust boxsize and bins accordingly
        for k in range(3):
            self.boxsize[k] = (maxvec[k] - minvec[k])
            voxeledge = self.boxsize[k] / self.numbins[k]
            self.bins[k] = np.arange(minvec[k], maxvec[k], voxeledge)

    def getBinIndex(rauxplus):
        '''
        If rauxplus are out of domain of bins, it
        will return the closest possible bin. Note bins in the bins
        array are labeled by their value at their left edge.
        '''
        indexes = [None] * self.dimension
        for i in range(self.dimension):
            try:
                indexes[i] = np.max(np.where(selfbins[i] <= rauxplus[i]))
            except:
                indexes[i] = 0
        return tuple(indexes)

    def getSequentialIndex(binIndex):
        seqIndex = binIndex[0] + self.numbins[0] * binIndex[1] + self.numbins[0] * self.numbins[1] * binIndex[2]
        return seqIndex

    def sampleFromBin(self, binIndex):
        '''
        Given conditioned variables, calculates the corresponding bin ,
        finds the closest non-empty bin (including itself) and samples one
        value randomly from all those available in that bin.
        '''
        occupiedBinIndex = self.nearestOccupiedNeighbour(binIndex)
        availableData = self.data[occupiedBinIndex]
        return random.choice(availableData)

    def loadData(self, trajs, nsigma=-1):
        '''
        Loads data into binning class and into training data for neural network.
        If nsigma < 0, it creates a box around all data. If it is a numerical value, it
        includes up to nsigma standard deviations around the mean.
        '''
        # Adjust boxes size for binning
        self.adjustBoxAux(trajs, nsigma) # Adjust box limits for r variables
        # Loop over all data and load into dictionary
        print('Binning ri+1 data...')
        for k, traj in enumerate(trajs):
            for j in range(len(traj) - self.numBinnedAuxVars * self.lagTimesteps):
                i = j + (self.numBinnedAuxVars - 1) * self.lagTimesteps
                conditionedVars = []
                if self.conditionOnPosition:
                    qi = traj[i][self.posIndex:self.posIndex + 3]
                    conditionedVars.append(qi)
                if self.conditionOnVelocity:
                    pi = traj[i][self.velIndex:self.velIndex + 3]
                    conditionedVars.append(pi)
                for m in range(self.numBinnedAuxVars):
                    ri = traj[i - m * self.lagTimesteps][self.auxIndex:self.auxIndex + 3]
                    conditionedVars.append(ri)
                conditionedVars = np.concatenate(conditionedVars)
                self.inputData.append(conditionedVars)
                riplus = traj[i + self.lagTimesteps][self.auxIndex:self.auxIndex + 3]
                ijk = self.getBinIndex(riplus)
                seqIndex = self.getSequentialIndex(ijk)
                outputVal = np.zeros(np.product(self.numbins))
                outputVal[seqIndex] = 1
                self.outputData.append(outputVal)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.inputData = np.array(self.inputData)
        self.outputData = np.array(self.outputData)
        self.updateDataStructures()
        self.percentageOccupiedBins = 100 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + str(int(self.percentageOccupiedBins)) + "% of bins occupied. \n" )


        '''
        Neural networks methods: note neural network class is nested inside deepSampler
        '''

        class Net(nn.Module):
            def __init__(self, layersSize = [30,100]):
                super(Net, self).__init__()
                # Activations fucntions: nn.ReLU, nn.Elu, nn.Sigmoid, nn.Tanh
                self.activationFunction = nn.ReLU()
                self.net = nn.Sequential(
                    nn.Linear(in_features=self.conditiondVarsDimension, out_features=layersSize[0]), self.activationFunction,
                    nn.Linear(layersSize[0], layersSize[1]), self.activationFunction,
                    nn.Linear(layersSize[1], layersSize[0]), self.activationFunction,
                    nn.Linear(layersSize[0], np.product(self.bins))
                )

            def forward(self, input: torch.FloatTensor):
                return self.net(input)

        # Create dataset for pyTorch, with batch training
        def createDataSet(self, batchsize):
            dataset = TensorDataset(torch.tensor(self.inputData, dtype=torch.float),
                                    torch.tensor(self.outputData, dtype=torch.float))
            self.dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

        def setNet(self, layersSize = [30,100]):
            self.net = Net(layersSize)

            # Define optimizer and loss function
            self.optim = torch.optim.Adam(Net.parameters(net), lr=0.001)
            self.Loss = nn.MSELoss()

        def trainNet(self, epochs = 1000):
            # Start training below:
            for epoch in range(epochs):
                loss = None
                for batch_x, batch_y in self.dataloader:
                    y_predict = self.net(batch_x)
                    loss = self.Loss(y_predict, batch_y)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                # Print the log every 100 times
                if (epoch + 1) % 10 == 0:
                    print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))