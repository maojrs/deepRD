import numpy as np
import os
import sys
import pickle
from itertools import product
import matplotlib.pyplot as plt
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler import binnedData

'''
Generates binned data structures on several different conditionings for the stochastic closure model.
Currently implemented on conditioning ri+1 on all the combinations qi,pi,ri,ri-1
'''

bsize = 5

#parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/free/benchmark/'
parentDirectory = os.environ['DATA'] + 'stochasticClosure/free/boxsize' + str(bsize)+ '/benchmark/'

fnamebase = parentDirectory + 'simMoriZwanzig_'
foldername = 'binnedData/'
#binningDataDirectory = os.path.join('../../data/stochasticClosure/free/', foldername)
binningDataDirectory = os.path.join(os.environ['DATA'] + 'stochasticClosure/free/boxsize' + str(bsize) + '/', foldername)


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
    trajs.append(traj)
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
binPositionList = [False] #[False, True]
binVelocitiesList = [False, True]
numBinnedAuxVarsList = [0,1,2]

def getNumberConditionedVariables(binPosition, binVelocity, numBinnedAuxVars):
    numConditionedVariables = 0
    if binPosition:
        numConditionedVariables += 1
    if binVelocity:
        numConditionedVariables += 1
    for i in range(numBinnedAuxVars):
        numConditionedVariables += 1
    return numConditionedVariables

for parameterCombination in product(*[binPositionList, binVelocitiesList, numBinnedAuxVarsList]):
    if parameterCombination != (False,False,0):
        binPosition, binVelocity, numBinnedAuxVars = parameterCombination
        numConditionedVariables = getNumberConditionedVariables(binPosition, binVelocity, numBinnedAuxVars)
        if numConditionedVariables == 1:
            dataOnBins = binnedData(boxsizeBinning, numbins1, lagTimesteps, binPosition, binVelocity, numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma1)
            parameterDictionary['numbins'] = numbins1
            parameterDictionary['nsigma'] = nsigma1
        elif numConditionedVariables == 2:
            dataOnBins = binnedData(boxsizeBinning, numbins2, lagTimesteps, binPosition, binVelocity, numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma2)
            parameterDictionary['numbins'] = numbins2
            parameterDictionary['nsigma'] = nsigma2
        else:
            dataOnBins = binnedData(boxsizeBinning, numbins3, lagTimesteps, binPosition, binVelocity, numBinnedAuxVars)
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