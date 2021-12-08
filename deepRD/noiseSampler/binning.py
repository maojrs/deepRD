from typing import List, Any, Union

import numpy as np
import random
import sys
from scipy import spatial
from itertools import product
from ..tools import trajectoryTools

'''
Classes to bin trajectory data obtained from a particle/molecular simulation.
The classes here assume the trajs object consist of an array/list 
of trajectories, where trajectory consists of an array of points of 
the form: (time, positionx, positiony, positionz, velocityx, velocityy, velocityz, type, rx, ry, rz).  
'''

class binnedData:
    '''
    Parent class to bin data. The dimension consists of the
    combined dimension of all the variable upon which one is conditioning, e.g (ri+1|qi,ri)
    would have dimension 6, 3 for qi and 3 for ri.
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, binPosition = False,
                 binVelocity = False, numBinnedAuxVars = 1, adjustPosVelBox = True):
        self.binPosition = binPosition
        self.binVelocity = binVelocity
        self.numBinnedAuxVars = numBinnedAuxVars
        self.adjustPosVelBox = adjustPosVelBox # If true adjust box limit for position and velocities variables
        self.dimension = None
        self.numConditionedVariables = None
        self.binningLabel = ''
        self.binningLabel2 = ''
        self.percentageOccupiedBins = None

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel()

        # Other important variables
        self.posIndex = 1  # Index of x position coordinate in trajectory files
        self.velIndex = 4  # Index of x velocity coordinate in trajectory files
        self.auxIndex = 8  # Position of x coordinate of r in trajectory files
        self.dataTree = None # dataTree structure to find nearest neighbors
        self.occupiedTuplesArray = None # array of tuples corresponding to occupied bins
        self.parameterDictionary = {}

        # Obtain indexes in box array
        self.posBoxIndex, self.velBoxIndex, self.auxBoxIndex = self.calculateBoxIndexes()

        if not isinstance(lagTimesteps, (int)):
            raise Exception('lagTimesteps should be an integer.')
        self.lagTimesteps = lagTimesteps # number of data timesteps to look back into data (integer)

        if isinstance(boxsize, (list, tuple, np.ndarray)):
            if len(boxsize) != self.dimension:
                raise Exception('Boxsize should be a scalar or an array matching the chosen dimension')
            self.boxsize = boxsize
        else:
            self.boxsize = [boxsize]*self.dimension

        if isinstance(numbins, (list, tuple, np.ndarray)):
            if len(numbins) != self.dimension:
                raise Exception('Numbins should be a scalar or an array matching the chosen dimension')
            self.numbins = numbins
        else:
            self.numbins = [numbins]*self.dimension

        # Create bins
        bins = [None]*self.dimension
        for i in range(self.dimension):
            bins[i] = np.arange(-self.boxsize[i] / 2., self.boxsize[i] / 2., self.boxsize[i] / self.numbins[i])
        self.bins = bins
        self.data = {}

    def calculateDimensionAndBinningLabel(self):
        self.binningLabel = 'ri+1|'
        self.binningLabel2 = ''
        self.dimension = 0
        self.numConditionedVariables = 0
        if self.binPosition:
            self.binningLabel += 'qi,'
            self.binningLabel2 += 'qi'
            self.dimension +=3
            self.numConditionedVariables += 1
        if self.binVelocity:
            self.binningLabel += 'pi,'
            self.binningLabel2 += 'pi'
            self.dimension +=3
            self.numConditionedVariables += 1
        for i in range(self.numBinnedAuxVars):
            self.dimension +=3
            self.numConditionedVariables += 1
            if i == 0:
                self.binningLabel += 'ri,'
                self.binningLabel2 += 'ri'
            else:
                self.binningLabel += 'ri-' +str(i) +','
                self.binningLabel2 += 'ri' + 'm'*i

    def calculateBoxIndexes(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        if self.binPosition and self.binVelocity:
            posBoxIndex = 0
            velBoxIndex = 3
            auxBoxIndex = 6
        elif not self.binPosition and self.binVelocity:
            posBoxIndex = None
            velBoxIndex = 0
            auxBoxIndex = 3
        elif self.binPosition and not self.binVelocity:
            posBoxIndex = 0
            velBoxIndex = None
            auxBoxIndex = 3
        elif not self.binPosition and not self.binVelocity:
            posBoxIndex = None
            velBoxIndex = None
            auxBoxIndex = 0
        else:
            posBoxIndex = None
            velBoxIndex = None
            auxBoxIndex = None
        return posBoxIndex, velBoxIndex, auxBoxIndex

    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
        else:
            print('Variable for adjustBox functions must be position or velocity')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + 3])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + 3])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + 3]
                    for j in range(3):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + 3])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + 3], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
        # Adjust boxsize and bins accordingly
        for k in range(3):
            self.boxsize[boxIndex + k] = (maxvec[k] - minvec[k])
            voxeledge = self.boxsize[boxIndex + k] / self.numbins[boxIndex + k]
            self.bins[boxIndex + k] = np.arange(minvec[k], maxvec[k], voxeledge)

    def adjustBoxAux(self, trajs, nsigma=-1):
        '''
        Calculate boxlimits of auxiliary variables from trajectories for binning and
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
        for m in range(self.numBinnedAuxVars):
            for k in range(3):
                boxIndex = self.auxBoxIndex + k + 3*m
                self.boxsize[boxIndex] = (maxvec[k] - minvec[k])
                voxeledge = self.boxsize[boxIndex] / self.numbins[boxIndex]
                self.bins[boxIndex] = np.arange(minvec[k], maxvec[k], voxeledge)

    def getBinIndex(self, conditionedVars):
        '''
        If conditioned variables are out of domain of bins, it
        will return the closest possible bin. Note bins in the bins
        array are labeled by their value at their left edge.
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

    def loadData(self, trajs, nsigma=-1):
        '''
        Loads data into binning class. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        '''
        # Adjust boxes size for binning
        if self.adjustPosVelBox and self.binPosition:
            self.adjustBox(trajs, 'position', nsigma)
        if self.adjustPosVelBox and self.binVelocity:
            self.adjustBox(trajs, 'velocity', nsigma)
        if self.numBinnedAuxVars > 0:
            self.adjustBoxAux(trajs, nsigma) # Adjust box limits for r variables
        # Loop over all data and load into dictionary
        print('Binning data for ' + self.binningLabel + ' ...')
        for k, traj in enumerate(trajs):
            for j in range(len(traj) - self.numBinnedAuxVars * self.lagTimesteps):
                i = j + (self.numBinnedAuxVars - 1) * self.lagTimesteps
                conditionedVars = []
                if self.binPosition:
                    qi = traj[i][self.posIndex:self.posIndex + 3]
                    conditionedVars.append(qi)
                if self.binVelocity:http://localhost:8889/notebooks/stochasticClosure/02_runLangevinNoiseSampler.ipynb#
                    pi = traj[i][self.velIndex:self.velIndex + 3]
                    conditionedVars.append(pi)
                for m in range(self.numBinnedAuxVars):
                    ri = traj[i - m * self.lagTimesteps][self.auxIndex:self.auxIndex + 3]
                    conditionedVars.append(ri)
                conditionedVars = np.concatenate(conditionedVars)
                riplus = traj[i + self.lagTimesteps][self.auxIndex:self.auxIndex + 3]
                ijk = self.getBinIndex(conditionedVars)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()
        self.percentageOccupiedBins = 100 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + str(int(self.percentageOccupiedBins)) + "% of bins occupied. \n" )
