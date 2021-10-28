from typing import List, Any, Union

import numpy as np
import random
import sys
from scipy import spatial
from itertools import product

'''
Classes to bin trajectory data obtained from a particle/molecular simulation.
The classes here assume the trajs object consist of an array/list 
of trajectories, where trajectory consists of an array of points of 
the form: (time, positionx, positiony, positionz, type, rx, ry, rz).  
'''

class binnedData:
    '''
    Parent class to bin data, just to be used as parent class. The dimension consists of the
    combined dimension of all the variable upon which one is conditioning, e.g (ri+1|qi,ri)
    would have dimension 6, 3 for qi and 3 for ri.
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, dimension = 3):
        self.dimension = dimension
        self.posIndex = 1  # Position of x coordinate in trajectory files
        self.rIndex = 5  # Position of x coordinate of r in trajectory files
        self.neighborsDictionary = {}
        self.dataTree = None # dataTree structure to find nearest neighbors
        self.occupiedTuplesArray = None # array of tuples corresponding to occupied bins
        self.parameterDictionary = {}

        if not isinstance(lagTimesteps, (int)):
            raise Exception('lagTimesteps should be an integer.')
        self.lagTimesteps = lagTimesteps # number of data timesteps to look back into data (integer)

        if isinstance(boxsize, (list, tuple, np.ndarray)):
            if len(boxsize) != dimension:
                raise Exception('Boxsize should be a scalar or an array matching the chosen dimension')
            self.boxsize = boxsize
        else:
            self.boxsize = [boxsize]*dimension

        if isinstance(numbins, (list, tuple, np.ndarray)):
            if len(numbins) != dimension:
                raise Exception('Numbins should be a scalar or an array matching the chosen dimension')
            self.numbins = numbins
        else:
            self.numbins = [numbins]*dimension

        # Create bins
        bins = [None]*self.dimension
        for i in range(self.dimension):
            bins[i] = np.arange(-self.boxsize[i] / 2., self.boxsize[i] / 2., self.boxsize[i] / self.numbins[i])
        self.bins = bins
        self.data = {}

    def adjustBoxLimits(self,trajs, indexes = None):
        '''
        Calculate boxlimits from trajectories for binning and adjust boxsize accordingly. Assumes
        'r' has dimension: self.dimension - 3; the other three correspond to position. For more complicated
        implementations, this function needs to be overriden.
        '''
        minvec = [0]*self.dimension
        maxvec = [0]*self.dimension
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.posIndex: self.posIndex + 3]
                condVar2 = traj[i][self.rIndex: self.rIndex + self.dimension - 3]
                for j in range(3):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
                for j in range(self.dimension - 3):
                    minvec[j+3] = min(minvec[j+3], condVar[j+3])
                    maxvec[j+3] = max(maxvec[j+3], condVar[j+3])
        condVarMin = np.array(minvec)
        condVarMax = np.array(maxvec)
        # Adjust boxsize and bins accordingly
        if indexes == None:
            self.boxsize = (condVarMax - condVarMin)
            voxeledge = self.boxsize / self.numbins
            for i in range(self.dimension):
                self.bins[i] = np.arange(condVarMin[i], condVarMax[i], voxeledge[i])
        else:
            for index in indexes:
                self.boxsize[index] = (condVarMax[index] - condVarMin[index])
                voxeledge = self.boxsize[index] / self.numbins[index]
                self.bins[index] = np.arange(condVarMin[index], condVarMax[index], voxeledge)

    def getBinIndex(self, conditionedVars):
        '''
        If conditioned variables are out of domain of bins, it
        will return the closest possible bin
        '''
        indexes = [None]*self.dimension
        for i in range(self.dimension):
            try:
                indexes[i] = np.max(np.where(self.bins[i] <= conditionedVars[i]))
            except:
                indexes[i] = 0
        return tuple(indexes)

    def createEmptyDictionary(self):
        '''
        Creates dictionary with all possible keys but empty values. Useful for
        possible applications. Note it also creates the trash index
        tuple [-1]*self.dimension
        '''
        emptyDict = {}
        iterable = (range(n) for n in self.numbins)
        for ijk in product(*iterable):  # ijk is a tuple
            emptyDict[ijk] = []
        trash = tuple([-1]*self.dimension)
        emptyDict[trash] = []
        return emptyDict

    def updateDataStructures(self):
        '''
        Needs to be called after loading data so nearest neighbor
        searches and other functions work.
        '''
        self.occupiedTuplesArray = []
        for key in self.data:
            self.occupiedTuplesArray.append(np.array(key))
        self.dataTree = spatial.cKDTree(self.occupiedTuplesArray)

    def nearestOccupiedNeighbour(self, referenceTupleIndex):
        '''
        Given the tuple index of a bin, finds nearest non-empty neighbor in bin space
        and return it as a tuple. It returns itself if occupied.
        '''
        referencePoint = np.array(referenceTupleIndex)
        nearestNeighborIndex = self.dataTree.query(referencePoint)[1]
        nearestPoint = self.occupiedTuplesArray[nearestNeighborIndex]
        return tuple(nearestPoint)

    def sample(self, conditionedVars):
        '''
        Given conditioned variables, calculates the corresponding bin ,
        finds the closest non-empty bin (including itself) and samples one
        value randomly from all those available in that bin.
        '''
        binIndex = self.getBinIndex(conditionedVars)
        occupiedBinIndex = self.nearestOccupiedNeighbour(binIndex)
        availableData = self.data[occupiedBinIndex]
        return random.choice(availableData)


class binnedData_qi(binnedData):
    '''
    Class to bin data of positions (three-dimensional). Useful to
    simulate the stochastic closure model of Mori-Zwanzig dynamics
    using ri+1|qi
    '''

    def __init__(self, boxsize, numbins=100, lagTimesteps = 1,):
        dimension = 3
        super().__init__(boxsize, numbins, lagTimesteps, dimension)

    def loadData(self, trajs):
        '''
        Loads data into binning class
        '''
        #self.createEmptyDictionary()
        # Loop over all data and load into dictionary
        print("Binning data for ri+1|qi ...")
        for k, traj in enumerate(trajs):
            for i in range(len(traj) - self.lagTimesteps):
                qi = traj[i][self.posIndex:self.posIndex + 3]
                riplus = traj[i + self.lagTimesteps][self.rIndex:]
                ijk = self.getBinIndex(qi)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()




class binnedData_ri(binnedData):
    '''
    Class to bin data of auxiliary variable r (three-dimensional). Useful to
    simulate the stochastic closure model of Mori-Zwanzig dynamics using ri+1|ri.
    Uses as base the binnedData_qi class
    '''
    def __init__(self, numbins=100, lagTimesteps = 1,):
        dimension = 3
        super().__init__(1, numbins, lagTimesteps, dimension)

    def loadData(self, trajs):
        '''
        Loads data into binning class
        '''
        self.adjustBoxLimits(trajs)
        print("Binning data for ri+1|ri ...")
        # Loop over all data and load into dictionary
        for k, traj in enumerate(trajs):
            for i in range(len(traj) - self.lagTimesteps):
                ri = traj[i][self.rIndex:self.rIndex + 3]  #
                riplus = traj[i + self.lagTimesteps][self.rIndex:self.rIndex + 3]
                ijk = self.getBinIndex(ri)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")

        self.updateDataStructures()


class binnedData_qiri(binnedData):
    '''
    Class to bin data of positions auxiliary variable r (six-dimensional).
    Useful to simulate the stochastic closure model of Mori-Zwanzig dynamics
    using (ri+1|qi,ri)
    '''

    def __init__(self, boxsize, numbins=100, lagTimesteps = 1,):
        dimension = 6
        super().__init__(boxsize, numbins, lagTimesteps, dimension)

    def loadData(self, trajs):
        '''
        Loads data into binning class
        '''
        # Loop over all data and load into dictionary
        self.adjustBoxLimits(trajs, indexes = [3,4,5]) # Just adjust box limits for r variables
        print("Binning data for ri+1|qi,ri ...")
        for k, traj in enumerate(trajs):
            for i in range(len(traj) - self.lagTimesteps):
                qi = traj[i][self.posIndex:self.posIndex + 3]
                ri = traj[i][self.rIndex:self.rIndex + 3]
                qiri = np.concatenate([qi,ri])
                riplus = traj[i + self.lagTimesteps][self.rIndex:self.rIndex + 3]
                ijk = self.getBinIndex(qiri)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()


class binnedData_qiririm(binnedData):
    '''
    Class to bin data of positions, auxiliary variable r, and auixiliary
    variable at a previous time (nine-dimensional).
    Useful to simulate the stochastic closure model of Mori-Zwanzig dynamics
    using (ri+1|qi,ri,ri-1)
    '''

    def __init__(self, boxsize, numbins=100, lagTimesteps = 1,):
        dimension = 9
        super().__init__(boxsize, numbins, lagTimesteps, dimension)

    def adjustBoxLimits(self,trajs):
        '''
        Calculate boxlimits from trajectories for binning and adjust boxsize accordingly. It only
        adjusts the r variables. It uses the same limits for ri and ri-1.
        '''
        minvec = [0., 0., 0.]
        maxvec = [0., 0., 0.]
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.rIndex: self.rIndex + 3]
                for j in range(3):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        condVarMin = np.array(minvec)
        condVarMax = np.array(maxvec)
        # Adjust boxsize and bins accordingly of ri and ri-1
        for index in range(3):
            self.boxsize[index + 3] = (condVarMax[index] - condVarMin[index])
            self.boxsize[index + 6] = (condVarMax[index] - condVarMin[index])
            voxeledge1 = self.boxsize[index + 3] / self.numbins[index + 3]
            voxeledge2 = self.boxsize[index + 6] / self.numbins[index + 6]
            self.bins[index + 3] = np.arange(condVarMin[index], condVarMax[index], voxeledge1)
            self.bins[index + 6] = np.arange(condVarMin[index], condVarMax[index], voxeledge2)

    def loadData(self, trajs):
        '''
        Loads data into binning class
        '''
        # Loop over all data and load into dictionary
        self.adjustBoxLimits(trajs) # Just adjust box limits for r variables
        print("Binning data for ri+1|qi,ri,ri-1 ...")
        for k, traj in enumerate(trajs):
            for j in range(len(traj) - 2 * self.lagTimesteps):
                i = j + self.lagTimesteps
                qi = traj[i][self.posIndex:self.posIndex + 3]
                ri = traj[i][self.rIndex:self.rIndex + 3]
                ri_minus = traj[i - self.lagTimesteps][self.rIndex:self.rIndex + 3]
                qiriri = np.concatenate([qi,ri,ri_minus])
                riplus = traj[i + self.lagTimesteps][self.rIndex:self.rIndex + 3]
                ijk = self.getBinIndex(qiriri)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()