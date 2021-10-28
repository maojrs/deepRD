import numpy as np
import os
import sys
import pickle
from itertools import product
import matplotlib.pyplot as plt
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler import binnedData_qi, binnedData_ri, binnedData_qiri, binnedData_qiririm

'''
Generates binned data structures on several different conditionings for the stochastic closure model.
Currently implemented for ri+1|qi; ri+1|ri; ri+1|qi,ri; ri+1|qi,ri,ri-1
'''

parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/benchmark/'
fnamebase = parentDirectory + 'simMoriZwanzig_'
binningDataDirectory = '../../data/stochasticClosure/binnedData/'

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

# # ----------------Binning for ri+1|qi--------------------------
#
# # Load binned data for ri+1|qi. Note one timestep from data equal parameters['dt'] * parameters['stride']
# qiBinnedData = binnedData_qi(boxsizeBinning, numbins, lagTimesteps)
# qiBinnedData.loadData(trajs)
# qiBinnedData.parameterDictionary = parameterDictionary
#
# # Dump qi binned data into pickle file and free memory
# print("Dumping data into pickle file ...")
# qiBinnedDataFilename = binningDataDirectory + 'qiBinnedData.pickle'
# pickle.dump(qiBinnedData, open(qiBinnedDataFilename, "wb" ))
# del qiBinnedData
# print("Binning for ri+1|qi done (1/4).")
#
# # ----------------Binning for ri+1|ri--------------------------
#
# # Load binned data for ri+1|ri
# riBinnedData = binnedData_ri(numbins, lagTimesteps)
# riBinnedData.loadData(trajs)
# riBinnedData.parameterDictionary = parameterDictionary
#
# # Dump ri binned data into pickle file and free memory
# print("Dumping data into pickle file ...")
# riBinnedDataFilename = binningDataDirectory + 'riBinnedData.pickle'
# pickle.dump(riBinnedData, open(riBinnedDataFilename, "wb" ))
# del riBinnedData
# print("Binning for ri+1|ri done (2/4).")

# ----------------Binning for ri+1|qi,ri--------------------------

# Load binned data for ri+1|qi,ri
qiriBinnedData = binnedData_qiri(boxsizeBinning, numbins, lagTimesteps)
qiriBinnedData.loadData(trajs)
qiriBinnedData.parameterDictionary = parameterDictionary

# Dump ri binned data into pickle file and free memory
print("Dumping data into pickle file ...")
qiriBinnedDataFilename = binningDataDirectory + 'qiriBinnedData.pickle'
pickle.dump(qiriBinnedData, open(qiriBinnedDataFilename, "wb" ))
del qiriBinnedData
print("Binning for ri+1|qi,ri done (3/4).")

# ----------------Binning for ri+1|qi,ri,ri-1 --------------------------

# Load binned data for ri+1|qi,ri,ri-1
qiririmBinnedData = binnedData_qiririm(boxsizeBinning, numbins, lagTimesteps)
qiririmBinnedData.loadData(trajs)
qiririmBinnedData.parameterDictionary = parameterDictionary

# Dump ri binned data into pickle file and free memory
print("Dumping data into pickle file ...")
qiririmBinnedDataFilename = binningDataDirectory + 'qiririmBinnedData.pickle'
pickle.dump(qiririmBinnedData, open(qiririmBinnedDataFilename, "wb" ))
del qiririmBinnedData
print("Binning for ri+1|qi,ri,ri-1 done (4/4).")
