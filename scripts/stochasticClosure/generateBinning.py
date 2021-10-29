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
Currently implemented for ri+1|qi; ri+1|ri; ri+1|qi,ri; ri+1|qi,ri,ri-1
'''

parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/benchmark/'
fnamebase = parentDirectory + 'simMoriZwanzig_'
foldername = 'binnedDataTest/'
binningDataDirectory = os.path.join('../../data/stochasticClosure/', foldername)

try:
    os.mkdir(binningDataDirectory)
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

# Load trajectory data from h5 files (only of distinguished particle)
trajs = []
print("Loading data ...")
for i in range(nfiles):
    traj = trajectoryTools.loadTrajectory(fnamebase, i)
    trajs.append(traj)
    sys.stdout.write("File " + str(i+1) + " of " + str(nfiles) + " done." + "\r")
print("All data loaded.")

# Parameters used for all binnings:
numbins = 50
lagTimesteps = 1  # Number of timesteps (from data) to look back in time
boxsizeBinning = boxsize

# List of possible combinations for binnings
binPositionList = [True, False]
binVelocitiesList = [True, False]
numBinnedAuxVarsList = [0,1,2]

for parameterCombination in product(*[binPositionList, binVelocitiesList, numBinnedAuxVarsList]):
    if parameterCombination != (False,False,0):
        binPosition, binVelocity, numBinnedAuxVars = parameterCombination
        dataOnBins = binnedData(boxsizeBinning, numbins, lagTimesteps, binPosition,binVelocity, numBinnedAuxVars)
        dataOnBins.loadData(trajs)
        dataOnBins.parameterDictionary = parameterDictionary

        # Dump qi binned data into pickle file and free memory
        print('Dumping data into pickle file ...')
        conditionedVarsString = dataOnBins.binningLabel2
        binnedDataFilename = binningDataDirectory + conditionedVarsString + 'BinnedData.pickle'
        pickle.dump(dataOnBins, open(binnedDataFilename, "wb"))
        print('Binning for ' + dataOnBins.binningLabel + ' done.')
        print(' ')
        del dataOnBins


# Old routines, but syntax still works

# # ----------------Binning for ri+1|qi--------------------------
#
# # Load binned data for ri+1|qi. Note one timestep from data equal parameters['dt'] * parameters['stride']
# qiBinnedData = binnedData(boxsizeBinning, numbins, lagTimesteps, binPosition = True,
#                           binVelocity = False, numBinnedAuxVars = 0)
# qiBinnedData.loadData(trajs)
# qiBinnedData.parameterDictionary = parameterDictionary
#
# # Dump qi binned data into pickle file and free memory
# print('Dumping data into pickle file ...')
# qiBinnedDataFilename = binningDataDirectory + 'qiBinnedData.pickle'
# pickle.dump(qiBinnedData, open(qiBinnedDataFilename, "wb" ))
# del qiBinnedData
# print('Binning for '+ qiBinnedData.binningLabel + ' done (1/4).')
#
# # ----------------Binning for ri+1|ri--------------------------
#
# # Load binned data for ri+1|ri
# riBinnedData = binnedData(1,numbins, lagTimesteps, binPosition = False,
#                           binVelocity = False, numBinnedAuxVars = 1)
# riBinnedData.loadData(trajs)
# riBinnedData.parameterDictionary = parameterDictionary
#
# # Dump ri binned data into pickle file and free memory
# print('Dumping data into pickle file ...')
# riBinnedDataFilename = binningDataDirectory + 'riBinnedData.pickle'
# pickle.dump(riBinnedData, open(riBinnedDataFilename, "wb" ))
# del riBinnedData
# print('Binning for '+ riBinnedData.binningLabel + ' done (2/4).')
#
# # ----------------Binning for ri+1|qi,ri--------------------------
#
# # Load binned data for ri+1|qi,ri
# qiriBinnedData = binnedData(boxsizeBinning, numbins, lagTimesteps, binPosition = True,
#                           binVelocity = False, numBinnedAuxVars = 1)
# qiriBinnedData.loadData(trajs)
# qiriBinnedData.parameterDictionary = parameterDictionary
#
# # Dump ri binned data into pickle file and free memory
# print('Dumping data into pickle file ...')
# qiriBinnedDataFilename = binningDataDirectory + 'qiriBinnedData.pickle'
# pickle.dump(qiriBinnedData, open(qiriBinnedDataFilename, "wb" ))
# del qiriBinnedData
# print('Binning for '+ qiriBinnedData.binningLabel + ' done (3/4).')
#
# # ----------------Binning for ri+1|qi,ri,ri-1 --------------------------
#
# # Load binned data for ri+1|qi,ri,ri-1
# qiririmBinnedData = binnedData(boxsizeBinning, numbins, lagTimesteps,binPosition = True,
#                           binVelocity = False, numBinnedAuxVars = 2)
# qiririmBinnedData.loadData(trajs)
# qiririmBinnedData.parameterDictionary = parameterDictionary
#
# # Dump ri binned data into pickle file and free memory
# print('Dumping data into pickle file ...')
# qiririmBinnedDataFilename = binningDataDirectory + 'qiririmBinnedData.pickle'
# pickle.dump(qiririmBinnedData, open(qiririmBinnedDataFilename, "wb" ))
# del qiririmBinnedData
# print('Binning for '+ qiririmBinnedData.binningLabel + ' done (4/4).')
