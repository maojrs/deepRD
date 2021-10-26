import numpy as np
import os
import sys
import pickle
from itertools import product
import matplotlib.pyplot as plt
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler import binnedData_qi, binnedData_ri, binnedData_qiri, binnedData_qiriri

parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/benchmark/'
fnamebase = parentDirectory + 'simMoriZwanzig_'
binningDataDirectory = '../../data/stochasticClosure/binnedData'

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

# Parameters used for all binnings:
numbins = 50
lagTimesteps = 1  # Number of timesteps (from data) to look back in time

# ----------------Binning for ri+1|qi--------------------------

# Load binned data for ri+1|qi. Note one timestep from data equal parameters['dt'] * parameters['stride']
boxsizeBinning = boxsize
qiBinnedData = binnedData_qi(boxsizeBinning, numbins, lagTimesteps)
qiBinnedData.loadData(trajs)
qiBinnedData.parameterDictionary = parameterDictionary

# Dump qi binned data into pickle file and free memory
qiBinnedDataFilename = binningDataDirectory + 'qiBinnedData.pickle'
pickle.dump(qiBinnedData, open(qiBinnedDataFilename, "wb" ))
del qiBinnedData
print("Binning for ri+1|1i (1/4) done")

# ----------------Binning for ri+1|ri--------------------------

# Load binned data for ri+1|ri
riBinnedData = binnedData_ri(numbins, lagTimesteps)
riBinnedData.loadData(trajs)
riBinnedData.parameterDictionary = parameterDictionary

# Dump ri binned data into pickle file and free memory
riBinnedDataFilename = binningDataDirectory + 'riBinnedData.pickle'
pickle.dump(riBinnedData, open(riBinnedDataFilename, "wb" ))
del riBinnedData
print("Binning for ri+1|ri (2/4) done")

# ----------------Binning for ri+1|qi,ri--------------------------

# Load binned data for ri+1|qi,ri
qiriBinnedData = binnedData_qiri(boxsizeBinning, numbins, lagTimesteps)
qiriBinnedData.loadData(trajs)
qiriBinnedData.parameterDictionary = parameterDictionary

# Dump ri binned data into pickle file and free memory
qiriBinnedDataFilename = binningDataDirectory + 'qiriBinnedData.pickle'
pickle.dump(qiriBinnedData, open(qiriBinnedDataFilename, "wb" ))
del qiriBinnedData
print("Binning for ri+1|qi,ri (3/4) done")

# ----------------Binning for ri+1|qi,ri,ri-1 --------------------------

# Load binned data for ri+1|qi,ri,ri-1
qiririBinnedData = binnedData_qiriri(boxsizeBinning, numbins, lagTimesteps)
qiririBinnedData.loadData(trajs)
qiririBinnedData.parameterDictionary = parameterDictionary

# Dump ri binned data into pickle file and free memory
qiririBinnedDataFilename = binningDataDirectory + 'riqiBinnedData.pickle'
pickle.dump(qiririBinnedData, open(qiririBinnedDataFilename, "wb" ))
del qiririBinnedData
print("Binning for ri+1|qi,ri,ri-1 (4/4) done")
