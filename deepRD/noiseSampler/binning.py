import numpy as np
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

    def __init__(self, boxsize, numbins = 100, dimension = 3):
        self.dimension = dimension
        self.posIndex = 1  # Position of x coordinate in trajectory files
        self.rIndex = 5  # Position of x coordinate of r in trajectory files

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
        indexes = [None]*self.dimension
        try:
            for i in range(self.dimension):
                indexes[i] = np.max(np.where(self.bins[i] < conditionedVars[i]))
        except:
            print(conditionedVars)
            #raise Exception('Boxsize inadequate for data. Data range exceeds box size')
            for i in range(self.dimension):
                indexes[i] = -1
        return tuple(indexes)

    def createEmptyDictionary(self):
        '''
        Create dictionary with all possible keys but empty values
        '''
        iterable = (range(n) for n in self.numbins)
        for ijk in product(*iterable):  # ijk is a tuple
            self.data[ijk] = []
        trash = tuple([-1]*self.dimension)
        self.data[trash] = []




class binnedData_qi(binnedData):
    '''
    Class to bin data of positions (three-dimensional). Useful to
    simulate the stochastic closure model of Mori-Zwanzig dynamics
    using ri+1|qi
    '''

    def __init__(self, boxsize, numbins=100):
        dimension = 3
        super().__init__(boxsize, numbins, dimension)


    def loadData(self, trajs, timestepMultiplier = 1):
        '''
        Loads data into binning class
        '''
        self.createEmptyDictionary()
        # Loop over all data and load into dictionary
        for k, traj in enumerate(trajs):
            for i in range(len(traj) - timestepMultiplier):
                qi = traj[i][self.posIndex:self.posIndex + 3]
                riplus = traj[i + timestepMultiplier][self.rIndex:]
                ijk = self.getBinIndex(qi)
                self.data[ijk].append(riplus)
            print("File ", k + 1, " of ", len(trajs), " done.", end="\r")


class binnedData_ri(binnedData):
    '''
    Class to bin data of auxiliary variable r (three-dimensional). Useful to
    simulate the stochastic closure model of Mori-Zwanzig dynamics using ri+1|ri.
    Uses as base the binnedData_qi class
    '''
    def __init__(self, numbins=100):
        dimension = 3
        super().__init__(1, numbins, dimension)

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


    def loadData(self, trajs, timestepMultiplier = 1):
        '''
        Loads data into binning class
        '''
        self.adjustBoxLimits(trajs)
        self.createEmptyDictionary()
        # Loop over all data and load into dictionary
        for k, traj in enumerate(trajs):
            for i in range(len(traj) - timestepMultiplier):
                ri = traj[i][self.rIndex:]  #
                riplus = traj[i + timestepMultiplier][self.rIndex:]
                ijk = self.getBinIndex(ri)
                self.data[ijk].append(riplus)
            print("File ", k + 1, " of ", len(trajs), " done.", end="\r")


