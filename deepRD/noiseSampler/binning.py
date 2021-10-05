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
        print("Binning data ...")
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

    def adjustBoxLimits(self,trajs):
        '''
        Calculate boxlimits from trajectories for binning and adjust boxsize accordingly
        '''
        rminvec = [0., 0., 0.]
        rmaxvec = [0., 0., 0.]
        for traj in trajs:
            for i in range(len(traj)):
                ri = traj[i][self.rIndex:]
                for j in range(3):
                    rminvec[j] = min(rminvec[j], ri[j])
                    rmaxvec[j] = max(rmaxvec[j], ri[j])
        rmin = np.floor(rminvec)
        rmax = np.ceil(rmaxvec)
        # Adjust boxsize and bins accordingly
        self.boxsize = (rmax - rmin)
        rvoxeledge = self.boxsize / self.numbins
        for i in range(self.dimension):
            self.bins[i] = np.arange(rmin[i], rmax[i], rvoxeledge[i])


    def loadData(self, trajs):
        '''
        Loads data into binning class
        '''
        self.adjustBoxLimits(trajs)
        self.createEmptyDictionary()
        print("Binning data ...")
        # Loop over all data and load into dictionary
        for k, traj in enumerate(trajs):
            for i in range(len(traj) - self.lagTimesteps):
                ri = traj[i][self.rIndex:]  #
                riplus = traj[i + self.lagTimesteps][self.rIndex:]
                ijk = self.getBinIndex(ri)
                try:
                    self.data[ijk].append(riplus)
                except KeyError:
                    self.data[ijk] = [riplus]
            sys.stdout.write("File " + str(k + 1) + " of " + str(len(trajs)) + " done." + "\r")

        self.updateDataStructures()



