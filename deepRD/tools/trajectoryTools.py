import h5py
import numpy as np

# Functions to load trajectories and manipulate them

def loadTrajectory(fnamebase, fnumber, fastload = False):
    '''
    Reads data from discrete trajectory and returns a simple np.array of
    integers representing the discrete trajectory. Assumes h5 file.
    :param fnamebase, base of the filename
    :param fnumber, filenumber
    :param fastload if true loads the H5 data, if false it converts the data to numpy.
    this however makes the loading very slow.
    :return: array of arrays representing the trajectory
    '''
    filename = fnamebase + str(fnumber).zfill(4) + '.h5'
    f = h5py.File(filename, 'r')

    # Get the data
    a_group_key = list(f.keys())[0]
    if fastload:
        data = f[a_group_key]
    else:
        data = np.array(f[a_group_key])
        #data = f[a_group_key][:] # equivalent

    return data


def writeTrajectory(traj, fnamebase, fnumber):
    '''
    Write trajectory into h5 file into the filename fnamebase+fnumber
    '''
    filename = fnamebase + str(fnumber).zfill(4) + '.h5'
    f = h5py.File(filename, 'w')
    f.create_dataset('trajectory', data=traj)
    f.close()

def loadDiscreteTrajectory(fnamebase, fnumber, fnamesuffix = '_discrete', filetype = 'h5'):
    '''
    Reads data from discrete trajectory and returns a simple np.array of
    integers representing the discrete trajectory. The file can be in the
    h5 or xyz format.
    :param fnamebase, base of the filename
    :param fnumber, filenumber
    :param fnamesuffix, suffix added at end of filename before the extension
    :param filetype, string indicating which format, h5 or xyz, is the file
    :return: array with integers representing the discrete trajectory
    '''
    if filetype == 'h5':
        filename = fnamebase + str(fnumber).zfill(4) + fnamesuffix + '.h5'
        f = h5py.File(filename, 'r')

        # Get the data
        a_group_key = list(f.keys())[0]
        array = f.get(a_group_key)
        nparray = np.array(array).transpose()[0]

        return nparray

    if filetype == 'xyz':
        filename = fnamebase + str(fnumber).zfill(4) + fnamesuffix + '.xyz'
        file = open(filename, "r")

        # Read file and save to array
        filelines = file.readlines()
        array = np.zeros(len(filelines), dtype = int)
        for i, line in enumerate(filelines):
            array[i] = int(float(line))
        return array



def listIndexSplit(inputList, *args):
    '''
    Function that splits inputList into smaller list by slicing in the indexes given by *args.
    :param inputList:
    :param args: int indexes where list should be splitted (Note to convert a
    list "mylist" into *args just do: *mylist)
    :return: list of sliced lists
    If extra arguments were passed prepend the 0th index and append the final
    # index of the passed list, in order toa v check for oid checking the start
    # and end of args in the loop. Also, add one in args for correct indexing.
    '''
    if args:
        args = (0,) + tuple(data+1 for data in args) + (len(inputList)+1,)
    # Slice list and return list of lists.
    slicedLists = []
    for start, end in zip(args, args[1:]):
        slicedLists.append(inputList[start:end-1])
    if slicedLists == []:
        slicedLists.append(inputList)
    return slicedLists



def splitDiscreteTrajs(discreteTrajs, unboundStateIndex = 0):
    '''
    Splits trajectories into smaller trajectories by cutting out
    all the states unboundStateindex (0)
    :param discreteTrajs: list of discrete trajectories
    :param unboundStateIndex: index of the unbound state used to
    decide where to cut the trajectories, normally we choose it to be
    zero.
    :return: List of sliced trajectories
    '''
    slicedDtrajs = []
    trajnum = 0
    for dtraj in discreteTrajs:
        # Slice trajectory using zeros as reference point
        indexZeros = np.where(dtraj==unboundStateIndex)
        slicedlist = listIndexSplit(dtraj, *indexZeros[0])
        # Remove the empty arrays
        for array in slicedlist:
            if array.size > 1:
                slicedDtrajs.append(array)
        trajnum += 1
        print("Slicing trajectory ", trajnum, " of ", len(discreteTrajs), " done.", end="\r")  
    return slicedDtrajs



def stitchTrajs(slicedDtrajs, minlength = 1000):
    '''
    Joins splitted trajectories into long trajectories of at least minlength. The trajectories that cannot
    be joined are left as they were.
    :param slicedDtrajs: list of discrete trajectories. Each discrete trajectory is a numpy array
    :param minlength: minimum length of stitched trajectory if any stititching is possible
    :return: list of stitched trajectories
    '''
    myslicedDtrajs = slicedDtrajs.copy()
    stitchedTrajs = []
    # Stitch trajectories until original sliced trajectory is empty
    while len(myslicedDtrajs) > 0:
        traj = myslicedDtrajs[0]
        del myslicedDtrajs[0]
        percentageDone = int(100.0*(1-len(myslicedDtrajs)/len(slicedDtrajs.copy())))
        print("Stitching trajectories: ", percentageDone, "% done   ", end="\r")
        # Try to keep all resulting trajectories over a certain length min length
        while traj.size <= minlength:
            foundTrajs = False
            for i in reversed(range(len(myslicedDtrajs))):
                # If end point and start point match, join trajectories
                if traj[-1] == myslicedDtrajs[i][0]:
                    if traj[-1] < 0:
                    	print("Error in stitching   ")
                    if myslicedDtrajs[i][0] < 0:
                    	print("Error in stitching   ")
                    foundTrajs = True
                    traj = np.concatenate([traj, myslicedDtrajs[i]])
                    del myslicedDtrajs[i]
            # If no possible trajectory to join is found, save trajectory and continue.
            if foundTrajs == False:
                break;
        stitchedTrajs.append(traj)   
    return stitchedTrajs

def convert2trajectory(timeArray, variableArrayList):
    '''
    Given a list of arrays, where each array corresponds to the trajectory of that variable, output
    a trajectory of the concatenated arrays as a single trajectory with more components. Note the variableArrayList
    is e.g. a list of array of positions, velocities, etc... each of this arrays stores in the each entry another
    array with the values (position/velocity) of all the particles in the simulation.
    '''
    traj = []
    trajLength = len(timeArray)
    varLength = len(variableArrayList[0][0])
    for i in range(trajLength):
        for j in range(varLength):
            time = np.array([timeArray[i]])
            trajElement = [time]
            for variableArray in variableArrayList:
                trajElement.append(variableArray[i][j])
            traj.append(np.concatenate((trajElement)))
    return np.array(traj)


def extractVariableFromTrajectory(trajs, variableIndex):
    '''
    Extracts the variable with index variableIndex from trajectories into an array
    '''
    variableArray = []
    for traj in trajs:
        for i in range(len(traj)):
            variableArray.append(traj[i][variableIndex])
    return np.array(variableArray)


def calculateMean(trajs, var = 'position'):
    '''
    Calculates mean of trajectories, var can be 'position' or 'velocity', assuming indexing in each element
    of a trajectory be (t,position,velocity).
    '''
    if var == 'position':
        index = 1
    elif var == 'velocity':
        index = 4
    meanPosition = np.zeros(3)
    totalSamples = 0
    for traj in trajs:
        for i in range(len(traj)):
            meanPosition += traj[i][index:index+3]
        totalSamples += len(traj)
    meanPosition = meanPosition/totalSamples
    return meanPosition

def calculateVariance(trajs, var = 'position', mean = None):
    '''
    Calculates variance of trajectories, var can be 'position' or 'velocity', assuming indexing in each element
    of a trajectory be (t,position,velocity). If mean is not given, it calls calulate mean.
    '''
    if var == 'position':
        index = 1
    elif var == 'velocity':
        index = 4
    if mean.any() == None:
        mean = calculateMean(trajs)
    variance = np.zeros(3)
    totalSamples = 0
    for traj in trajs:
        for i in range(len(traj)):
            devFromMean = traj[i][index:index+3] - mean
            variance += devFromMean*devFromMean
        totalSamples += len(traj)
    variance = variance/totalSamples
    return variance

def calculateStdDev(trajs, var = 'position', mean = None):
    variance = calculateVariance(trajs, var, mean)
    stddev = np.sqrt(variance)
    return stddev

def calculateAutoCorrelation(trajs, lagtimesteps, stride = 1, var = 'position', mean = None, variance = None):
    '''
    Calculates autocorrelation of trajectories, for a given stride. Variable var can be 'position' or 'velocity',
    assuming indexing in each element of a trajectory be (t,position,velocity). If mean and variance are not given,
    it calls calulate mean and calculate variance.
    '''
    if var == 'position':
        index = 1
    elif var == 'velocity':
        index = 4
    if mean.any() == None:
        mean = calculateMean(trajs)
    if variance.any() == None:
        variance = calculateVariance(trajs, mean)
    totalSamples = 0
    AC = 0.0
    for traj in trajs:
        for i in range(len(traj)-lagtimesteps*stride):
            devFromMean = traj[i][index:index+3] - mean
            devFromMean2 = traj[i + lagtimesteps*stride][index:index+3] - mean
            AC += np.dot(devFromMean, devFromMean2)
        totalSamples += len(traj)
    AC = AC/totalSamples
    AC = AC/variance
    return AC

def calculateAutoCorrelationFunction(trajs, lagtimesteps, stride = 1, var = 'position'):
    '''
    Calculates autocorrelation function of trajectories, for a given lagtimesteps (length of time interval in
    timesteps) and stride. Variable var can be 'position' or 'velocity', assuming indexing in each element of a
    trajectory be (t,position,velocity).
    '''
    ACF = []
    mean = calculateMean(trajs, var)
    # Calculate one dimensional variance
    if var == 'position':
        index = 1
    elif var == 'velocity':
        index = 4
    variance = 0
    totalSamples = 0
    for traj in trajs:
        for i in range(len(traj)):
            devFromMean = traj[i][index:index+3] - mean
            variance += np.dot(devFromMean,devFromMean)
        totalSamples += len(traj)
    variance = variance/totalSamples
    for lagtime in range(lagtimesteps):
        ACF.append(calculateAutoCorrelation(trajs, lagtime, stride, var, mean, variance))
        print('Computing ACF:', 100*(lagtime+1)/lagtimesteps , '% complete   ', end="\r")
    ACF = np.array(ACF)
    return ACF
