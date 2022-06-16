import os
import sys
import pickle
import numpy as np
from itertools import product
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler import binnedData

'''
Generates binned data structures on several different conditionings for the stochastic closure model.
Currently implemented on conditioning ri+1 on all the combinations qi,pi,ri,ri-1
'''

bsize = 5

parentDirectory = os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize)+ '/benchmark/'
fnamebase = parentDirectory + 'simMoriZwanzig_'
foldername = 'binnedData/'
binningDataDirectory = os.path.join(os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize) + '/', foldername)


try:
    os.makedirs(binningDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Load parameters from parameters file
parameterDictionary = analysisTools.readParameters(parentDirectory + "parameters")
# Parameters for loading continuous trajectories from files (from original simulation)
nfiles = parameterDictionary['numFiles']
dt = parameterDictionary['dt']
stride = parameterDictionary['stride']
totalTimeSteps = parameterDictionary['timesteps']
boxsize = parameterDictionary['boxsize']
boundaryType = parameterDictionary['boundaryType']

if bsize != boxsize:
    print('Requested boxsize does not match simulation')

# Load trajectory data from h5 files (only of distinguished particle)
trajs = []
print("Loading data ...")
for i in range(nfiles):
    traj = trajectoryTools.loadTrajectory(fnamebase, i)
    lentraj = np.shape([traj])[1]
    # Compute and add relativeDistance and relative velocity at end of trajectory
    relDistVel = np.zeros([lentraj,4])
    for j in range(int(lentraj/2)):
        relDist = np.linalg.norm(traj[2*j][1:4] - traj[2*j+1][1:4])
        relVel = traj[2*j+1][4:7] - traj[2*j][4:7]
        relDistVel[2*j][0] = relDist
        relDistVel[2*j+1][0] = relDist
        relDistVel[2*j][1:4] = relVel
        relDistVel[2*j+1][1:4] = -1*relVel
    newtraj = np.concatenate([traj,relDistVel], axis=1)
    trajs.append(newtraj)
    sys.stdout.write("File " + str(i+1) + " of " + str(nfiles) + " done." + "\r")
print("\nAll data loaded.")
print(' ')

# Parameters used for binnings:
lagTimesteps = 1  # Number of timesteps (from data) to look back in time
boxsizeBinning = boxsize # Overriden by default when loading trajectory data
numbins1 = 30 #50
numbins2 = 10 #50
numbins3 = 5 #50
nsigma1 = 3 # Only include up to nsigma standard deviations around mean of data. If no value given, includes all.
nsigma2 = 2
nsigma3 = 2

# Add elements to parameter dictionary
parameterDictionary['lagTimesteps'] = lagTimesteps

# List of possible combinations for binnings
#binPositionList = [False] #[False, True]
binRelativeDistanceList = [False,True]
binVelocitiesList = [False, True]
binRelativeVelocityList = [False, True]
numBinnedAuxVarsList = [2] #[0,1] #[0,1,2]

def getNumberConditionedVariables(binRelDist, binRelVelocity, binVelocity, numBinnedAuxVars):
    numConditionedVariables = 0
    if binRelDist:
        numConditionedVariables += 0 # One dimensional, so we assume it doesn't count
    if binRelVelocity:
        numConditionedVariables += 1
    if binVelocity:
        numConditionedVariables += 1
    for i in range(numBinnedAuxVars):
        numConditionedVariables += 1
    return numConditionedVariables

for parameterCombination in product(*[binRelativeDistanceList, binRelativeVelocityList, binVelocitiesList, numBinnedAuxVarsList]):
    if parameterCombination != (False,False,0):
        binRelDist, binRelVelocity, binVelocity, numBinnedAuxVars = parameterCombination
        numConditionedVariables = getNumberConditionedVariables(binRelDist, binRelVelocity, numBinnedAuxVars)
        if numConditionedVariables == 1:
            dataOnBins = binnedData(boxsizeBinning, numbins1, lagTimesteps, binPosition=False, binVelocity=binVelocity
                                    , numBinnedAuxVars=numBinnedAuxVars, binRelDistance = binRelDist,
                                    binRelVelocity = binRelVelocity)
            dataOnBins.loadData(trajs, nsigma1)
            parameterDictionary['numbins'] = numbins1
            parameterDictionary['nsigma'] = nsigma1
        elif numConditionedVariables == 2:
            dataOnBins = binnedData(boxsizeBinning, numbins2, lagTimesteps, binPosition=False, binVelocity=binVelocity
                                    , numBinnedAuxVars=numBinnedAuxVars, binRelDistance = binRelDist,
                                    binRelVelocity = binRelVelocity)
            dataOnBins.loadData(trajs, nsigma2)
            parameterDictionary['numbins'] = numbins2
            parameterDictionary['nsigma'] = nsigma2
        else:
            dataOnBins = binnedData(boxsizeBinning, numbins3, lagTimesteps, binPosition=False, binVelocity=binVelocity
                                    , numBinnedAuxVars=numBinnedAuxVars, binRelDistance = binRelDist,
                                    binRelVelocity = binRelVelocity)
            dataOnBins.loadData(trajs, nsigma3)
            parameterDictionary['numbins'] = numbins3
            parameterDictionary['nsigma'] = nsigma3
        parameterDictionary['percentageOccupiedBins'] = dataOnBins.percentageOccupiedBins
        dataOnBins.parameterDictionary = parameterDictionary

        # Dump qi binned data into pickle file and free memory
        print('Dumping data into pickle file ...')
        conditionedVarsString = dataOnBins.binningLabel2
        binnedDataFilename = binningDataDirectory + conditionedVarsString + 'BinnedData.pickle'
        pickle.dump(dataOnBins, open(binnedDataFilename, "wb"))
        print('Binning for ' + dataOnBins.binningLabel + ' done.')
        print(' ')
        del dataOnBins