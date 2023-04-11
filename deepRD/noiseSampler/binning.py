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
        self.posBoxIndex = None
        self.velBoxIndex = None
        self.auxBoxIndex = None
        self.calculateBoxIndexes()


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
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 3
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for relative distance, relative velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 3
        self.auxBoxIndex = maxIndexSoFar


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
            numvars = 3
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 3
        else:
            print('Variable for adjustBox functions must be position or velocity')
        # Adjust min and max of box
        minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
        maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][trajIndex: trajIndex + numvars]
                for j in range(numvars):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        # Don't take into account data beyond nsigma standard deviations
        if nsigma > 0:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvecAlt = mean - nsigma * stddev
            maxvecAlt = mean + nsigma * stddev
            for j in range(numvars):
                minvec[j] = max(minvec[j], minvecAlt[j])
                maxvec[j] = min(maxvec[j], maxvecAlt[j])
        # Adjust boxsize and bins accordingly
        for k in range(numvars):
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
        # Adjust min and max of box
        minvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 3])
        maxvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 3])
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.auxIndex: self.auxIndex + 3]
                for j in range(3):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        # Don't take into account data beyond nsigma standard deviations
        if nsigma > 0:
            mean = trajectoryTools.calculateMean(trajs, [self.auxIndex,self.auxIndex + 3])
            stddev = trajectoryTools.calculateStdDev(trajs, [self.auxIndex,self.auxIndex + 3], mean)
            minvecAlt = mean - nsigma*stddev
            maxvecAlt = mean + nsigma*stddev
            for j in range(3):
                minvec[j] = max(minvec[j], minvecAlt[j])
                maxvec[j] = min(maxvec[j], maxvecAlt[j])
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
                if self.binVelocity:
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
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimer(binnedData):
    '''
    Specialized version of binnedData for dimer example
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, binRelDistance = False,
                 binRelSpeed = False, binVelCenterMass = False, numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, False, False, numBinnedAuxVars,
                         adjustPosVelBox)
        self.binRelDistance = binRelDistance
        self.binRelSpeed = binRelSpeed # Same as axisRelVelocity
        self.binVelCenterMass = binVelCenterMass

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()

        # Other important variables
        self.relDistIndex = 11 # Index of relative distance(scalar) in trajectory files (between dimer)
        self.relSpeIndex = 12 # Index of relative speed (along axis of a dimer)
        self.velCenterMassIndex = 13 # Index center of mass velocity (two components (axis, tangential))

        # Obtain indexes in box array
        self.relDistBoxIndex = None
        self.relSpeBoxIndex = None
        self.velCenterMassBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
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
        if self.binRelDistance:
            self.binningLabel += 'dqi,'
            self.binningLabel2 += 'dqi'
            self.dimension +=1
            self.numConditionedVariables += 1
        if self.binRelSpeed:
            self.binningLabel += 'dpi,'
            self.binningLabel2 += 'dpi'
            self.dimension +=1
            self.numConditionedVariables += 1
        if self.binVelCenterMass:
            self.binningLabel += 'vi,'
            self.binningLabel2 += 'vi'
            self.dimension +=1
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

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 3
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for relative distance, relative velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 3

        if self.binRelDistance and self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            self.relSpeBoxIndex = maxIndexSoFar + 1
            maxIndexSoFar += 2
        elif not self.binRelDistance and self.binRelSpeed:
            self.relSpeBoxIndex = maxIndexSoFar
            maxIndexSoFar += 1
        elif self.binRelDistance and not self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            maxIndexSoFar =+ 1

        # Finally index for VelCenterMass
        if self.binVelCenterMass:
            self.velCenterMassBoxIndex = maxIndexSoFar
            self.auxBoxIndex = maxIndexSoFar + 2
        else:
            self.auxBoxIndex = maxIndexSoFar



    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 3
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 3
        elif variable == 'relDistance':
            trajIndex = self.relDistIndex
            boxIndex = self.relDistBoxIndex
            onlyPositive =  True
            numvars = 1
        elif variable == 'relSpeed':
            trajIndex = self.relSpeIndex
            boxIndex = self.relSpeBoxIndex
            numvars = 1
        elif variable == 'velCenterMass':
            trajIndex = self.velCenterMassIndex
            boxIndex = self.velCenterMassBoxIndex
            onlyPositive = True
            numvars = 2
        else:
            print('Variable for adjustBox functions must be position, velocity, relDistance, relSpeed or velCenterMass')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            if onlyPositive:
                for j in range(numvars):
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        for k in range(numvars):
            self.boxsize[boxIndex + k] = (maxvec[k] - minvec[k])
            voxeledge = self.boxsize[boxIndex + k] / self.numbins[boxIndex + k]
            self.bins[boxIndex + k] = np.arange(minvec[k], maxvec[k], voxeledge)


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
        if self.adjustPosVelBox and self.binRelDistance:
            self.adjustBox(trajs, 'relDistance', nsigma)
        if self.adjustPosVelBox and self.binRelSpeed:
            self.adjustBox(trajs, 'relSpeed', nsigma)
        if self.adjustPosVelBox and self.binVelCenterMass:
            self.adjustBox(trajs, 'velCenterMass', nsigma)
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
                if self.binVelocity:
                    pi = traj[i][self.velIndex:self.velIndex + 3]
                    conditionedVars.append(pi)
                if self.binRelDistance:
                    dqi = traj[i][self.relDistIndex:self.relDistIndex + 1]
                    conditionedVars.append(dqi)
                if self.binRelSpeed:
                    dpi = traj[i][self.relSpeIndex:self.relSpeIndex + 1]
                    conditionedVars.append(dpi)
                if self.binVelCenterMass:
                    dpcm = traj[i][self.velCenterMassIndex:self.velCenterMassIndex + 2]
                    conditionedVars.append(dpcm)
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
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimerGlobal(binnedData):
    '''
    Same as binnedDataDimer class, but applied to global state (ri+1 is 6 dimensions instead of 3)
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, binPosition = False,
                 numBinnedVelVars = 1, binRelDistance = False, binRelSpeed = False, binCMvelocity = False,
                 numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, binPosition, False, numBinnedAuxVars,
                         adjustPosVelBox)
        self.numBinnedVelVars = numBinnedVelVars
        if self.numBinnedVelVars > 0:
            self.binVelocity = True

        self.binRelDistance = binRelDistance
        self.binRelSpeed = binRelSpeed
        self.binCMvelocity = binCMvelocity

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()
        self.numparticles = 2

        # Other important variables
        self.relDistIndex = 11
        self.relSpeIndex = 12
        self.CMvelocityIndex = 13

        # Obtain indexes in box array
        self.relDistBoxIndex = None
        self.relSpeBoxIndex = None
        self.CMvelocityBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
        self.binningLabel = 'ri+1|'
        self.binningLabel2 = ''
        self.dimension = 0
        self.numConditionedVariables = 0
        if self.binPosition:
            self.binningLabel += 'qi,'
            self.binningLabel2 += 'qi'
            self.dimension +=6
            self.numConditionedVariables += 1
        for i in range(self.numBinnedVelVars):
            if i == 0:
                self.binningLabel += 'pi,'
                self.binningLabel2 += 'pi'
            else:
                self.binningLabel += 'pi-' + str(i) + ','
                self.binningLabel2 += 'pi' + 'm' * i
            self.dimension +=6
            self.numConditionedVariables += 1
        if self.binRelDistance:
            self.binningLabel += 'dqi,'
            self.binningLabel2 += 'dqi'
            self.dimension +=1
        if self.binRelSpeed:
            self.binningLabel += 'dpi,'
            self.binningLabel2 += 'dpi'
            self.dimension +=1
            self.numConditionedVariables += 1
        if self.binCMvelocity:
            self.binningLabel += 'vi,'
            self.binningLabel2 += 'vi'
            self.dimension +=2
            self.numConditionedVariables += 1
        for i in range(self.numBinnedAuxVars):
            self.dimension +=6
            self.numConditionedVariables += 1
            if i == 0:
                self.binningLabel += 'ri,'
                self.binningLabel2 += 'ri'
            else:
                self.binningLabel += 'ri-' +str(i) +','
                self.binningLabel2 += 'ri' + 'm'*i

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 6
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for component velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        elif not self.binVelocity:
            maxIndexSoFar = max(indexes) + 6
        else:
            maxIndexSoFar = max(indexes) + 6 * self.numBinnedVelVars

        # Index for relDistance and relSpeed
        if self.binRelDistance and self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            self.relSpeBoxIndex = maxIndexSoFar + 1
            maxIndexSoFar += 2
        elif not self.binRelDistance and self.binRelSpeed:
            self.relSpeBoxIndex = maxIndexSoFar
            maxIndexSoFar += 1
        elif self.binRelDistance and not self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            maxIndexSoFar = + 1

        if self.binCMvelocity:
            self.CMvelocityBoxIndex = maxIndexSoFar
            maxIndexSoFar = + 2

        self.auxBoxIndex = maxIndexSoFar


    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'relDistance':
            trajIndex = self.relDistIndex
            boxIndex = self.relDistBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'relSpeed':
            trajIndex = self.relSpeIndex
            boxIndex = self.relSpeBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'CMvelocity':
            trajIndex = self.CMvelocityIndex
            boxIndex = self.CMvelocityBoxIndex
            numvars = 2
            onlyPositive = [False] * numvars
        else:
            print('Variable for adjustBox functions must be position, velocity')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            for j in range(numvars):
                if onlyPositive[j]:
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        if variable == 'position':
            for k in range(self.numparticles * numvars):
                kk = k%numvars
                currentBoxindex = boxIndex + k
                self.boxsize[currentBoxindex] = (maxvec[kk] - minvec[kk])
                voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                self.bins[currentBoxindex] = np.arange(minvec[kk], maxvec[kk], voxeledge)
        elif variable == 'velocity':
            for k in range(self.numparticles * numvars * self.numBinnedVelVars):
                kk = k%(numvars)
                currentBoxindex = boxIndex + k
                self.boxsize[currentBoxindex] = (maxvec[kk] - minvec[kk])
                voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                self.bins[currentBoxindex] = np.arange(minvec[kk], maxvec[kk], voxeledge)
        else:
            for k in range(numvars):
                currentBoxindex = boxIndex + k
                self.boxsize[currentBoxindex] = (maxvec[k] - minvec[k])
                voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                self.bins[currentBoxindex] = np.arange(minvec[k], maxvec[k], voxeledge)


    def adjustBoxAux(self, trajs, nsigma=-1):
        '''
        Calculate boxlimits of auxiliary variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.auxBoxIndex correspond to the index of the x-coordinate of the first
        aux variable in the boxsize array; numAuxVars corresponds to the number of auxiliary
        variables, e.g. in ri+1|ri,ri-1, it would be two.
        '''
        # Adjust min and max of box
        minvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 3])
        maxvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 3])
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.auxIndex: self.auxIndex + 3]
                for j in range(3):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        # Don't take into account data beyond nsigma standard deviations
        if nsigma > 0:
            mean = trajectoryTools.calculateMean(trajs, [self.auxIndex,self.auxIndex + 3])
            stddev = trajectoryTools.calculateStdDev(trajs, [self.auxIndex,self.auxIndex + 3], mean)
            minvecAlt = mean - nsigma*stddev
            maxvecAlt = mean + nsigma*stddev
            for j in range(3):
                minvec[j] = max(minvec[j], minvecAlt[j])
                maxvec[j] = min(maxvec[j], maxvecAlt[j])
        # Adjust boxsize and bins accordingly
        for m in range(self.numBinnedAuxVars):
            for k in range(6):
                kk = k % 3
                boxIndex = self.auxBoxIndex + k + 6 * m
                self.boxsize[boxIndex] = (maxvec[kk] - minvec[kk])
                voxeledge = self.boxsize[boxIndex] / self.numbins[boxIndex]
                self.bins[boxIndex] = np.arange(minvec[kk], maxvec[kk], voxeledge)


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
        if self.adjustPosVelBox and self.binRelDistance:
            self.adjustBox(trajs, 'relDistance', nsigma)
        if self.adjustPosVelBox and self.binRelSpeed:
            self.adjustBox(trajs, 'relSpeed', nsigma)
        if self.adjustPosVelBox and self.binCMvelocity:
            self.adjustBox(trajs, 'CMvelocity', nsigma)
        if self.numBinnedAuxVars > 0:
            self.adjustBoxAux(trajs, nsigma) # Adjust box limits for r variables
        # Loop over all data and load into dictionary
        print('Binning data for ' + self.binningLabel + ' ...')
        for k, traj in enumerate(trajs):
            for j in range(int( (len(traj)/2 - (self.numBinnedAuxVars + 1) * self.lagTimesteps) )):
                i = j + self.numBinnedAuxVars * self.lagTimesteps
                conditionedVars = []
                if self.binPosition:
                    qi1 = traj[2*i][self.posIndex:self.posIndex + 3]
                    qi2 = traj[2*i+1][self.posIndex:self.posIndex + 3]
                    conditionedVars.append(qi1)
                    conditionedVars.append(qi2)
                if self.binVelocity:
                    for m in range(self.numBinnedVelVars):
                        ii = i - m * self.lagTimesteps
                        pi1 = traj[2 * ii][self.velIndex:self.velIndex + 3]
                        pi2 = traj[2 * ii + 1][self.velIndex:self.velIndex + 3]
                        conditionedVars.append(pi1)
                        conditionedVars.append(pi2)
                if self.binRelDistance:
                    dqi = traj[2*i][self.relDistIndex:self.relDistIndex + 1]
                    conditionedVars.append(dqi)
                if self.binRelSpeed:
                    dpi = traj[2*i][self.relSpeIndex:self.relSpeIndex + 1]
                    conditionedVars.append(dpi)
                if self.binCMvelocity:
                    si = traj[2*i][self.CMvelocityIndex:self.CMvelocityIndex + 2]
                    conditionedVars.append(si)
                for m in range(self.numBinnedAuxVars):
                    ii = i - m * self.lagTimesteps
                    ri1 = traj[2*ii][self.auxIndex:self.auxIndex + 3]
                    ri2 = traj[2*ii+1][self.auxIndex:self.auxIndex + 3]
                    conditionedVars.append(ri1)
                    conditionedVars.append(ri2)
                conditionedVars = np.concatenate(conditionedVars)
                ii = i + self.lagTimesteps
                riplus1 = traj[2*ii][self.auxIndex:self.auxIndex + 3]
                riplus2 = traj[2*ii+1][self.auxIndex:self.auxIndex + 3]
                riplus = np.concatenate((riplus1, riplus2))
                ijk = self.getBinIndex(conditionedVars)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimerRotatedVel(binnedData):
    '''
    Another specialized version of binnedData for dimer example, that works with rotated velocities
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1,
                 binRotatedVelocity = False, numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, False, False, numBinnedAuxVars,
                         adjustPosVelBox)
        self.binRotatedVelocity = binRotatedVelocity

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()

        # Other important variables
        self.relPositionIndex = 11
        self.rotatedVelocityIndex = 14 # Index of component velocity in trajectory files (between dimer)

        # Obtain indexes in box array
        self.rotatedVelocityBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
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
        if self.binRotatedVelocity:
            self.binningLabel += 'vi,'
            self.binningLabel2 += 'vi'
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

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 3
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for component velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 3

        if self.binRotatedVelocity:
            self.rotatedVelocityBoxIndex = maxIndexSoFar
            self.auxBoxIndex = maxIndexSoFar + 3
        else:
            self.auxBoxIndex = maxIndexSoFar


    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'rotatedVelocity':
            trajIndex = self.rotatedVelocityIndex
            boxIndex = self.rotatedVelocityBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        else:
            print('Variable for adjustBox functions must be position, velocity, relDistance, relSpeed or velCenterMass')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            for j in range(numvars):
                if onlyPositive[j]:
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        for k in range(numvars):
            self.boxsize[boxIndex + k] = (maxvec[k] - minvec[k])
            voxeledge = self.boxsize[boxIndex + k] / self.numbins[boxIndex + k]
            self.bins[boxIndex + k] = np.arange(minvec[k], maxvec[k], voxeledge)


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
        if self.adjustPosVelBox and self.binRotatedVelocity:
            self.adjustBox(trajs, 'rotatedVelocity', nsigma)
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
                if self.binVelocity:
                    pi = traj[i][self.velIndex:self.velIndex + 3]
                    conditionedVars.append(pi)
                if self.binRotatedVelocity:
                    vi = traj[i][self.rotatedVelocityIndex:self.rotatedVelocityIndex + 3]
                    conditionedVars.append(vi)
                for m in range(self.numBinnedAuxVars):
                    ri = traj[i - m * self.lagTimesteps][self.auxIndex:self.auxIndex + 3]
                    conditionedVars.append(ri)
                conditionedVars = np.concatenate(conditionedVars)
                # NEED TO ROTATE THIS BAD BOY
                deltaX = traj[i][self.relPositionIndex:self.relPositionIndex + 3]
                riplus = traj[i + self.lagTimesteps][self.auxIndex:self.auxIndex + 3]
                riplus = trajectoryTools.rotateVec(deltaX,riplus)
                ijk = self.getBinIndex(conditionedVars)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimerRotatedVelGlobal(binnedData):
    '''
    Another specialized version of binnedData for dimer example, that works with rotated velocities
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1,
                 binRotatedVelocity = False, numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, False, False, numBinnedAuxVars,
                         adjustPosVelBox)
        self.binRotatedVelocity = binRotatedVelocity

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()

        # Other important variables
        self.relPositionIndex = 11
        self.rotatedVelocityIndex = 14 # Index of component velocity in trajectory files (between dimer)

        # Obtain indexes in box array
        self.rotatedVelocityBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
        self.binningLabel = 'ri+1|'
        self.binningLabel2 = ''
        self.dimension = 0
        self.numConditionedVariables = 0
        if self.binPosition:
            self.binningLabel += 'qi,'
            self.binningLabel2 += 'qi'
            self.dimension +=6
            self.numConditionedVariables += 1
        if self.binVelocity:
            self.binningLabel += 'pi,'
            self.binningLabel2 += 'pi'
            self.dimension +=6
            self.numConditionedVariables += 1
        if self.binRotatedVelocity:
            self.binningLabel += 'vi,'
            self.binningLabel2 += 'vi'
            self.dimension +=6
            self.numConditionedVariables += 1
        for i in range(self.numBinnedAuxVars):
            self.dimension +=6
            self.numConditionedVariables += 1
            if i == 0:
                self.binningLabel += 'ri,'
                self.binningLabel2 += 'ri'
            else:
                self.binningLabel += 'ri-' +str(i) +','
                self.binningLabel2 += 'ri' + 'm'*i

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 6
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for component velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 6

        if self.binRotatedVelocity:
            self.rotatedVelocityBoxIndex = maxIndexSoFar
            self.auxBoxIndex = maxIndexSoFar + 6
        else:
            self.auxBoxIndex = maxIndexSoFar


    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'rotatedVelocity':
            trajIndex = self.rotatedVelocityIndex
            boxIndex = self.rotatedVelocityBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        else:
            print('Variable for adjustBox functions must be position, velocity, relDistance, relSpeed or velCenterMass')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            for j in range(numvars):
                if onlyPositive[j]:
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        if variable == 'position' or variable == 'velocity':
            for l in range(self.numparticles):
                for k in range(numvars):
                    currentBoxindex = boxIndex + numvars * l + k
                    self.boxsize[currentBoxindex] = (maxvec[k] - minvec[k])
                    voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                    self.bins[currentBoxindex] = np.arange(minvec[k], maxvec[k], voxeledge)
        else:
            for k in range(numvars):
                currentBoxindex = boxIndex + k
                self.boxsize[currentBoxindex] = (maxvec[k] - minvec[k])
                voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                self.bins[currentBoxindex] = np.arange(minvec[k], maxvec[k], voxeledge)

    def adjustBoxAux(self, trajs, nsigma=-1):
        '''
        Calculate boxlimits of auxiliary variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.auxBoxIndex correspond to the index of the x-coordinate of the first
        aux variable in the boxsize array; numAuxVars corresponds to the number of auxiliary
        variables, e.g. in ri+1|ri,ri-1, it would be two.
        '''
        # Adjust min and max of box
        minvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 6])
        maxvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 6])
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.auxIndex: self.auxIndex + 6]
                for j in range(6):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        # Don't take into account data beyond nsigma standard deviations
        if nsigma > 0:
            mean = trajectoryTools.calculateMean(trajs, [self.auxIndex,self.auxIndex + 6])
            stddev = trajectoryTools.calculateStdDev(trajs, [self.auxIndex,self.auxIndex + 6], mean)
            minvecAlt = mean - nsigma*stddev
            maxvecAlt = mean + nsigma*stddev
            for j in range(6):
                minvec[j] = max(minvec[j], minvecAlt[j])
                maxvec[j] = min(maxvec[j], maxvecAlt[j])
        # Adjust boxsize and bins accordingly
        for m in range(self.numBinnedAuxVars):
            for k in range(6):
                boxIndex = self.auxBoxIndex + k + 6 * m
                self.boxsize[boxIndex] = (maxvec[k] - minvec[k])
                voxeledge = self.boxsize[boxIndex] / self.numbins[boxIndex]
                self.bins[boxIndex] = np.arange(minvec[k], maxvec[k], voxeledge)

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
        if self.adjustPosVelBox and self.binRotatedVelocity:
            self.adjustBox(trajs, 'rotatedVelocity', nsigma)
        if self.numBinnedAuxVars > 0:
            self.adjustBoxAux(trajs, nsigma) # Adjust box limits for r variables
        # Loop over all data and load into dictionary
        print('Binning data for ' + self.binningLabel + ' ...')
        for k, traj in enumerate(trajs):
            for j in range(int((len(traj) / 2 - (self.numBinnedAuxVars + 1) * self.lagTimesteps))):
                i = j + self.numBinnedAuxVars * self.lagTimesteps
                conditionedVars = []
                if self.binPosition:
                    qi1 = traj[2 * i][self.posIndex:self.posIndex + 3]
                    qi2 = traj[2 * i + 1][self.posIndex:self.posIndex + 3]
                    conditionedVars.append(qi1)
                    conditionedVars.append(qi2)
                if self.binVelocity:
                    pi1 = traj[2 * i][self.velIndex:self.velIndex + 3]
                    pi2 = traj[2 * i + 1][self.velIndex:self.velIndex + 3]
                    conditionedVars.append(pi1)
                    conditionedVars.append(pi2)
                if self.binRotatedVelocity:
                    vi1 = traj[2*i][self.rotatedVelocityIndex:self.rotatedVelocityIndex + 3]
                    vi2 = traj[2*i+1][self.rotatedVelocityIndex:self.rotatedVelocityIndex + 3]
                    conditionedVars.append(vi1)
                    conditionedVars.append(vi2)
                for m in range(self.numBinnedAuxVars):
                    ii = i - m * self.lagTimesteps
                    ri1 = traj[2 * ii][self.auxIndex:self.auxIndex + 3]
                    ri2 = traj[2 * ii + 1][self.auxIndex:self.auxIndex + 3]
                    conditionedVars.append(ri1)
                    conditionedVars.append(ri2)
                conditionedVars = np.concatenate(conditionedVars)
                # Rotate ri+1
                deltaX = traj[i][self.relPositionIndex:self.relPositionIndex + 3]
                ii = i + self.lagTimesteps
                riplus1 = traj[2 * ii][self.auxIndex:self.auxIndex + 3]
                riplus2 = traj[2 * ii + 1][self.auxIndex:self.auxIndex + 3]
                riplus1 = trajectoryTools.rotateVec(deltaX,riplus1)
                riplus2 = trajectoryTools.rotateVec(deltaX, riplus2)
                riplus = np.concatenate((riplus1, riplus2))
                ijk = self.getBinIndex(conditionedVars)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimerConstrained1D(binnedData):
    '''
    Same as binnedData class, but constrained to 1D (x position)
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, binPosition = False,
                 binVelocity = False, binRelDistance = False, binRelSpeed = False, numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, binPosition, binVelocity, numBinnedAuxVars,
                         adjustPosVelBox)

        self.binRelDistance = binRelDistance
        self.binRelSpeed = binRelSpeed

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()

        # Other important variables
        self.relDistIndex = 11
        self.relSpeIndex = 12

        # Obtain indexes in box array
        self.relDistBoxIndex = None
        self.relSpeBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
        self.binningLabel = 'ri+1|'
        self.binningLabel2 = ''
        self.dimension = 0
        self.numConditionedVariables = 0
        if self.binPosition:
            self.binningLabel += 'qi,'
            self.binningLabel2 += 'qi'
            self.dimension +=1
            self.numConditionedVariables += 1
        if self.binVelocity:
            self.binningLabel += 'pi,'
            self.binningLabel2 += 'pi'
            self.dimension +=1
            self.numConditionedVariables += 1
        if self.binRelDistance:
            self.binningLabel += 'dqi,'
            self.binningLabel2 += 'dqi'
            self.dimension +=1
        if self.binRelSpeed:
            self.binningLabel += 'dpi,'
            self.binningLabel2 += 'dpi'
            self.dimension +=1
            self.numConditionedVariables += 1
        for i in range(self.numBinnedAuxVars):
            self.dimension +=1
            self.numConditionedVariables += 1
            if i == 0:
                self.binningLabel += 'ri,'
                self.binningLabel2 += 'ri'
            else:
                self.binningLabel += 'ri-' +str(i) +','
                self.binningLabel2 += 'ri' + 'm'*i

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 1
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for component velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 1

        # Index for relDistance and relSpeed
        if self.binRelDistance and self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            self.relSpeBoxIndex = maxIndexSoFar + 1
            maxIndexSoFar += 2
        elif not self.binRelDistance and self.binRelSpeed:
            self.relSpeBoxIndex = maxIndexSoFar
            maxIndexSoFar += 1
        elif self.binRelDistance and not self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            maxIndexSoFar = + 1

        self.auxBoxIndex = maxIndexSoFar


    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'relDistance':
            trajIndex = self.relDistIndex
            boxIndex = self.relDistBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'relSpeed':
            trajIndex = self.relSpeIndex
            boxIndex = self.relSpeBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        else:
            print('Variable for adjustBox functions must be position, velocity')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            for j in range(numvars):
                if onlyPositive[j]:
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        for k in range(numvars):
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
        # Adjust min and max of box
        minvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 1])
        maxvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 1])
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.auxIndex: self.auxIndex + 1]
                for j in range(1):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        # Don't take into account data beyond nsigma standard deviations
        if nsigma > 0:
            mean = trajectoryTools.calculateMean(trajs, [self.auxIndex,self.auxIndex + 1])
            stddev = trajectoryTools.calculateStdDev(trajs, [self.auxIndex,self.auxIndex + 1], mean)
            minvecAlt = mean - nsigma*stddev
            maxvecAlt = mean + nsigma*stddev
            for j in range(1):
                minvec[j] = max(minvec[j], minvecAlt[j])
                maxvec[j] = min(maxvec[j], maxvecAlt[j])
        # Adjust boxsize and bins accordingly
        for m in range(self.numBinnedAuxVars):
            for k in range(1):
                boxIndex = self.auxBoxIndex + k + m
                self.boxsize[boxIndex] = (maxvec[k] - minvec[k])
                voxeledge = self.boxsize[boxIndex] / self.numbins[boxIndex]
                self.bins[boxIndex] = np.arange(minvec[k], maxvec[k], voxeledge)


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
        if self.adjustPosVelBox and self.binRelDistance:
            self.adjustBox(trajs, 'relDistance', nsigma)
        if self.adjustPosVelBox and self.binRelSpeed:
            self.adjustBox(trajs, 'relSpeed', nsigma)
        if self.numBinnedAuxVars > 0:
            self.adjustBoxAux(trajs, nsigma) # Adjust box limits for r variables
        # Loop over all data and load into dictionary
        print('Binning data for ' + self.binningLabel + ' ...')
        for k, traj in enumerate(trajs):
            for j in range(len(traj) - self.numBinnedAuxVars * self.lagTimesteps):
                i = j + (self.numBinnedAuxVars - 1) * self.lagTimesteps
                conditionedVars = []
                if self.binPosition:
                    qi = traj[i][self.posIndex:self.posIndex + 1]
                    conditionedVars.append(qi)
                if self.binVelocity:
                    pi = traj[i][self.velIndex:self.velIndex + 1]
                    conditionedVars.append(pi)
                if self.binRelDistance:
                    dqi = traj[i][self.relDistIndex:self.relDistIndex + 1]
                    conditionedVars.append(dqi)
                if self.binRelSpeed:
                    dpi = traj[i][self.relSpeIndex:self.relSpeIndex + 1]
                    conditionedVars.append(dpi)
                for m in range(self.numBinnedAuxVars):
                    ri = traj[i - m * self.lagTimesteps][self.auxIndex:self.auxIndex + 1]
                    conditionedVars.append(ri)
                conditionedVars = np.concatenate(conditionedVars)
                riplus = traj[i + self.lagTimesteps][self.auxIndex:self.auxIndex + 1]
                ijk = self.getBinIndex(conditionedVars)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimerConstrained1DGlobal(binnedData):
    '''
    Same as binnedDataDimerConstrained1D class, but applied to global state (ri+1 is 6 dimensions instead of 3)
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, binPosition = False,
                 binVelocity = False, binRelDistance = False, binRelSpeed = False, binCMvelocity = False,
                 numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, binPosition, binVelocity, numBinnedAuxVars,
                         adjustPosVelBox)

        self.binRelDistance = binRelDistance
        self.binRelSpeed = binRelSpeed
        self.binCMvelocity = binCMvelocity

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()
        self.numparticles = 2

        # Other important variables
        self.relDistIndex = 11
        self.relSpeIndex = 12
        self.CMvelocityIndex = 13

        # Obtain indexes in box array
        self.relDistBoxIndex = None
        self.relSpeBoxIndex = None
        self.CMvelocityBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
        self.binningLabel = 'ri+1|'
        self.binningLabel2 = ''
        self.dimension = 0
        self.numConditionedVariables = 0
        if self.binPosition:
            self.binningLabel += 'qi,'
            self.binningLabel2 += 'qi'
            self.dimension +=2
            self.numConditionedVariables += 1
        if self.binVelocity:
            self.binningLabel += 'pi,'
            self.binningLabel2 += 'pi'
            self.dimension +=2
            self.numConditionedVariables += 1
        if self.binRelDistance:
            self.binningLabel += 'dqi,'
            self.binningLabel2 += 'dqi'
            self.dimension +=1
        if self.binRelSpeed:
            self.binningLabel += 'dpi,'
            self.binningLabel2 += 'dpi'
            self.dimension +=1
            self.numConditionedVariables += 1
        if self.binCMvelocity:
            self.binningLabel += 'vi,'
            self.binningLabel2 += 'vi'
            self.dimension +=2
            self.numConditionedVariables += 1
        for i in range(self.numBinnedAuxVars):
            self.dimension +=2
            self.numConditionedVariables += 1
            if i == 0:
                self.binningLabel += 'ri,'
                self.binningLabel2 += 'ri'
            else:
                self.binningLabel += 'ri-' +str(i) +','
                self.binningLabel2 += 'ri' + 'm'*i

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 2
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for component velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 2

        # Index for relDistance and relSpeed
        if self.binRelDistance and self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            self.relSpeBoxIndex = maxIndexSoFar + 1
            maxIndexSoFar += 2
        elif not self.binRelDistance and self.binRelSpeed:
            self.relSpeBoxIndex = maxIndexSoFar
            maxIndexSoFar += 1
        elif self.binRelDistance and not self.binRelSpeed:
            self.relDistBoxIndex = maxIndexSoFar
            maxIndexSoFar = + 1

        if self.binCMvelocity:
            self.CMvelocityBoxIndex = maxIndexSoFar
            maxIndexSoFar = + 1

        self.auxBoxIndex = maxIndexSoFar


    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'relDistance':
            trajIndex = self.relDistIndex
            boxIndex = self.relDistBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'relSpeed':
            trajIndex = self.relSpeIndex
            boxIndex = self.relSpeBoxIndex
            numvars = 1
            onlyPositive = [False]*numvars
        elif variable == 'CMvelocity':
            trajIndex = self.CMvelocityIndex
            boxIndex = self.CMvelocityBoxIndex
            numvars = 1
            onlyPositive = [False] * numvars
        else:
            print('Variable for adjustBox functions must be position, velocity')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            for j in range(numvars):
                if onlyPositive[j]:
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        if variable == 'position' or variable == 'velocity':
            for l in range(self.numparticles):
                for k in range(numvars):
                    currentBoxindex = boxIndex + numvars * l + k
                    self.boxsize[currentBoxindex] = (maxvec[k] - minvec[k])
                    voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                    self.bins[currentBoxindex] = np.arange(minvec[k], maxvec[k], voxeledge)
        else:
            for k in range(numvars):
                currentBoxindex = boxIndex + k
                self.boxsize[currentBoxindex] = (maxvec[k] - minvec[k])
                voxeledge = self.boxsize[currentBoxindex] / self.numbins[currentBoxindex]
                self.bins[currentBoxindex] = np.arange(minvec[k], maxvec[k], voxeledge)


    def adjustBoxAux(self, trajs, nsigma=-1):
        '''
        Calculate boxlimits of auxiliary variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.auxBoxIndex correspond to the index of the x-coordinate of the first
        aux variable in the boxsize array; numAuxVars corresponds to the number of auxiliary
        variables, e.g. in ri+1|ri,ri-1, it would be two.
        '''
        # Adjust min and max of box
        minvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 1])
        maxvec = np.array(trajs[0][0][self.auxIndex: self.auxIndex + 1])
        for traj in trajs:
            for i in range(len(traj)):
                condVar = traj[i][self.auxIndex: self.auxIndex + 1]
                for j in range(1):
                    minvec[j] = min(minvec[j], condVar[j])
                    maxvec[j] = max(maxvec[j], condVar[j])
        # Don't take into account data beyond nsigma standard deviations
        if nsigma > 0:
            mean = trajectoryTools.calculateMean(trajs, [self.auxIndex,self.auxIndex + 1])
            stddev = trajectoryTools.calculateStdDev(trajs, [self.auxIndex,self.auxIndex + 1], mean)
            minvecAlt = mean - nsigma*stddev
            maxvecAlt = mean + nsigma*stddev
            for j in range(1):
                minvec[j] = max(minvec[j], minvecAlt[j])
                maxvec[j] = min(maxvec[j], maxvecAlt[j])
        # Adjust boxsize and bins accordingly
        for m in range(self.numBinnedAuxVars):
            for k in range(2):
                kk = 0
                boxIndex = self.auxBoxIndex + k + m
                self.boxsize[boxIndex] = (maxvec[kk] - minvec[kk])
                voxeledge = self.boxsize[boxIndex] / self.numbins[boxIndex]
                self.bins[boxIndex] = np.arange(minvec[kk], maxvec[kk], voxeledge)


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
        if self.adjustPosVelBox and self.binRelDistance:
            self.adjustBox(trajs, 'relDistance', nsigma)
        if self.adjustPosVelBox and self.binRelSpeed:
            self.adjustBox(trajs, 'relSpeed', nsigma)
        if self.adjustPosVelBox and self.binCMvelocity:
            self.adjustBox(trajs, 'CMvelocity', nsigma)
        if self.numBinnedAuxVars > 0:
            self.adjustBoxAux(trajs, nsigma) # Adjust box limits for r variables
        # Loop over all data and load into dictionary
        print('Binning data for ' + self.binningLabel + ' ...')
        for k, traj in enumerate(trajs):
            for j in range(int( (len(traj)/2 - (self.numBinnedAuxVars + 1) * self.lagTimesteps) )):
                i = j + self.numBinnedAuxVars * self.lagTimesteps
                conditionedVars = []
                if self.binPosition:
                    qi1 = traj[2*i][self.posIndex:self.posIndex + 1]
                    qi2 = traj[2*i+1][self.posIndex:self.posIndex + 1]
                    conditionedVars.append(qi1)
                    conditionedVars.append(qi2)
                if self.binVelocity:
                    pi1 = traj[2*i][self.velIndex:self.velIndex + 1]
                    pi2 = traj[2*i+1][self.velIndex:self.velIndex + 1]
                    conditionedVars.append(pi1)
                    conditionedVars.append(pi2)
                if self.binRelDistance:
                    dqi = traj[2*i][self.relDistIndex:self.relDistIndex + 1]
                    conditionedVars.append(dqi)
                if self.binRelSpeed:
                    dpi = traj[2*i][self.relSpeIndex:self.relSpeIndex + 1]
                    conditionedVars.append(dpi)
                if self.binCMvelocity:
                    si = traj[2*i][self.CMvelocityIndex:self.CMvelocityIndex + 1]
                    conditionedVars.append(si)
                for m in range(self.numBinnedAuxVars):
                    ii = i - m * self.lagTimesteps
                    ri1 = traj[2*ii][self.auxIndex:self.auxIndex + 1]
                    ri2 = traj[2*ii+1][self.auxIndex:self.auxIndex + 1]
                    conditionedVars.append(ri1)
                    conditionedVars.append(ri2)
                conditionedVars = np.concatenate(conditionedVars)
                ii = i + self.lagTimesteps
                riplus1 = traj[2*ii][self.auxIndex:self.auxIndex + 1]
                riplus2 = traj[2*ii+1][self.auxIndex:self.auxIndex + 1]
                riplus = np.concatenate((riplus1, riplus2))
                ijk = self.getBinIndex(conditionedVars)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")
        self.updateDataStructures()
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )


class binnedDataDimer2(binnedData):
    '''
    Another specialized version of binnedData for dimer example
    '''

    def __init__(self, boxsize, numbins = 100, lagTimesteps = 1, binComponentVelocity = False,
                 numBinnedAuxVars = 1, adjustPosVelBox = True):
        super().__init__(boxsize, numbins, lagTimesteps, False, False, numBinnedAuxVars,
                         adjustPosVelBox)
        self.binComponentVelocity = binComponentVelocity

        # Calculate dimension and binning label
        self.calculateDimensionAndBinningLabel2()

        # Other important variables
        self.componentVelocityIndex = 11 # Index of component velocity in trajectory files (between dimer)

        # Obtain indexes in box array
        self.componentVelocityBoxIndex = None
        self.calculateBoxIndexes2()

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

    def calculateDimensionAndBinningLabel2(self):
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
        if self.binComponentVelocity:
            self.binningLabel += 'vi,'
            self.binningLabel2 += 'vi'
            self.dimension +=2
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

    def calculateBoxIndexes2(self):
        '''
        Determines the indexes correspo0nding to which variable in the box array. It assumes the
        variable are ordered first position, then velocity then aux variables.
        '''
        # Box indexes for position and velocity
        if self.binPosition and self.binVelocity:
            self.posBoxIndex = 0
            self.velBoxIndex = 3
        elif not self.binPosition and self.binVelocity:
            self.velBoxIndex = 0
        elif self.binPosition and not self.binVelocity:
            self.posBoxIndex = 0

        # Box indexes for component velocity and aux vars
        indexes = [self.posBoxIndex, self.velBoxIndex]
        indexes = list(filter(lambda ele: ele is not None, indexes)) # remove Nones
        if not indexes: #empty list
            maxIndexSoFar = 0
        else:
            maxIndexSoFar = max(indexes) + 3

        if self.binComponentVelocity:
            self.componentVelocityBoxIndex = maxIndexSoFar
            self.auxBoxIndex = maxIndexSoFar + 2
        else:
            self.auxBoxIndex = maxIndexSoFar


    def adjustBox(self, trajs, variable = 'position', nsigma=-1):
        '''
        Calculate boxlimits of position or velocity variables from trajectories for binning and
        adjust boxsize accordingly. If nsigma < 0, it creates a box around all data.
        If it is a numerical value, it includes up to nsigma standard deviations around the mean.
        The variable self.pos/velBoxIndex correspond to the index of the x-position/velocity in the
        boxsize array.
        '''
        onlyPositive = False
        if variable == 'position':
            trajIndex = self.posIndex
            boxIndex = self.posBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'velocity':
            trajIndex = self.velIndex
            boxIndex = self.velBoxIndex
            numvars = 3
            onlyPositive = [False]*numvars
        elif variable == 'componentVelocity':
            trajIndex = self.componentVelocityIndex
            boxIndex = self.componentVelocityBoxIndex
            numvars = 2
            onlyPositive = [False,True]
        else:
            print('Variable for adjustBox functions must be position, velocity, relDistance, relSpeed or velCenterMass')
        if nsigma < 0:
            minvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            maxvec = np.array(trajs[0][0][trajIndex: trajIndex + numvars])
            for traj in trajs:
                for i in range(len(traj)):
                    condVar = traj[i][trajIndex: trajIndex + numvars]
                    for j in range(numvars):
                        minvec[j] = min(minvec[j], condVar[j])
                        maxvec[j] = max(maxvec[j], condVar[j])
        else:
            mean = trajectoryTools.calculateMean(trajs, [trajIndex, trajIndex + numvars])
            stddev = trajectoryTools.calculateStdDev(trajs, [trajIndex, trajIndex + numvars], mean)
            minvec = mean - nsigma * stddev
            maxvec = mean + nsigma * stddev
            for j in range(numvars):
                if onlyPositive[j]:
                    minvec[j] = max(minvec[j], 0.0)
        # Adjust boxsize and bins accordingly
        for k in range(numvars):
            self.boxsize[boxIndex + k] = (maxvec[k] - minvec[k])
            voxeledge = self.boxsize[boxIndex + k] / self.numbins[boxIndex + k]
            self.bins[boxIndex + k] = np.arange(minvec[k], maxvec[k], voxeledge)


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
        if self.adjustPosVelBox and self.binComponentVelocity:
            self.adjustBox(trajs, 'componentVelocity', nsigma)
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
                if self.binVelocity:
                    pi = traj[i][self.velIndex:self.velIndex + 3]
                    conditionedVars.append(pi)
                if self.binComponentVelocity:
                    vi = traj[i][self.componentVelocityIndex:self.componentVelocityIndex + 2]
                    conditionedVars.append(vi)
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
        self.percentageOccupiedBins = 100.0 * len(self.occupiedTuplesArray)/np.product(self.numbins)
        sys.stdout.write("Loaded trajectories into bins. \r" )
        sys.stdout.write("\n" + "{:.2f}".format(self.percentageOccupiedBins) + "% of bins occupied. \n" )



