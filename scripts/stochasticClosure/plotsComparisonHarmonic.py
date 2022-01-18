import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from deepRD.noiseSampler import binnedData
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
matplotlib.rcParams.update({'font.size': 15})


# Benchmark data folder
parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/harmonic/benchmarkComparison/'
benchmarkfnamebase = parentDirectory + 'simMoriZwanzig_'
# Reduced models data folders
localDataDirectory = '../../data/stochasticClosure/harmonic/benchmarkReduced'
numModels = 8
redModelfnamebase = [localDataDirectory]*numModels
redModelfnamebase[0] += '_ri/simMoriZwanzigReduced_'
redModelfnamebase[1] += '_ririm/simMoriZwanzigReduced_'
redModelfnamebase[2] += '_qi/simMoriZwanzigReduced_'
redModelfnamebase[3] += '_qiri/simMoriZwanzigReduced_'
redModelfnamebase[4] += '_qiririm/simMoriZwanzigReduced_'
redModelfnamebase[5] += '_pi/simMoriZwanzigReduced_'
redModelfnamebase[6] += '_piri/simMoriZwanzigReduced_'
redModelfnamebase[7] += '_piririm/simMoriZwanzigReduced_'


# Read relevant parameters
parameterDictionary = analysisTools.readParameters(parentDirectory + "parameters")
numSimulations = 1 #6 #parameterDictionary['numFiles']
dt = parameterDictionary['dt'] 
integratorStride = parameterDictionary['stride']
totalTimeSteps = parameterDictionary['timesteps'] 
boxsize = parameterDictionary['boxsize']
boundaryType = parameterDictionary['boundaryType']
parameterDictionary


# ## Load benchmark and reduced model trajectory data

# Load benchmark trajectory data from h5 files (only of distinguished particle)
trajs_ref = []
print("Loading benchmark data ...")
for i in range(numSimulations):
    traj = trajectoryTools.loadTrajectory(benchmarkfnamebase, i)
    trajs_ref.append(traj)    
    print("File ", i+1, " of ", numSimulations, " done.", end="\r")
print("Benchmark data loaded.")


# Load reduced model trajectory data from h5 files (only of distinguished particle)
allTrajs = [None]*numModels
print("Loading reduced models data ...")
for i in range(numModels):
    try:
        iTraj = []
        for j in range(numSimulations):
            traj = trajectoryTools.loadTrajectory(redModelfnamebase[i], j)
            iTraj.append(traj)
            print("File ", i+1, " of ", numSimulations, " done.", end="\r")
        allTrajs[i] = iTraj
    except:
        continue
print("Reduced models data loaded.")


# ## Distribution plots comparisons


# Choose which reduced model to compare (just uncomment one)
conditionedList = ['ri', 'qiri', 'pi', 'piri'] #Possibilities 'qi', 'ri', 'qiri', 'qiririm'
bestCondition = 'piri' # The one that  best matches
trajIndexes = []
if 'ri' in conditionedList:
    trajIndexes.append(0)
if 'ririm' in conditionedList:
    trajIndexes.append(1)
if 'qi' in conditionedList:
    trajIndexes.append(2)
if 'qiri' in conditionedList:
    trajIndexes.append(3)
if 'qiririm' in conditionedList:
    trajIndexes.append(4)
if 'pi' in conditionedList:
    trajIndexes.append(5)
if 'piri' in conditionedList:
    trajIndexes.append(6)
if 'piririm' in conditionedList:
    trajIndexes.append(7)
numConditions = len(trajIndexes)
labelList = [r'$r_{i+1}|r_i$', r'$r_{i+1}|r_i, r_{i-1}$',
             r'$r_{i+1}|q_i$', r'$r_{i+1}|q_i,r_i$', r'$r_{i+1}|q_i, r_i, r_{i-1}$', 
             r'$r_{i+1}|p_i$', r'$r_{i+1}|p_i,r_i$', r'$r_{i+1}|p_i, r_i, r_{i-1}$']


# Extract variables to plot from trajectories (x components)
varIndex = 1 # 1=x, 2=y, 3=z
position = [None] * numConditions
velocity = [None] * numConditions
for i in range(numConditions):
    currentTrajs = allTrajs[trajIndexes[i]]
    position[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs, variableIndex = varIndex)
    velocity[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs, variableIndex = varIndex + 3)
position_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = varIndex)
velocity_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = varIndex + 3)



# Plot distributions comparisons for position and velocity
plotLines = True
numbins = 40
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4)) #fig, ax = plt.subplots()
# Plot reference distributions
posRef, binEdges = np.histogram(position_ref, bins=numbins, density = True)
binsPosRef = 0.5 * (binEdges[1:] + binEdges[:-1])
velRef, binEdges = np.histogram(velocity_ref, bins=numbins, density = True)
binsVelRef = 0.5 * (binEdges[1:] + binEdges[:-1])
if plotLines:
    ax1.plot(binsPosRef, posRef, '-', c='black', label = 'benchmark');
    ax2.plot(binsVelRef, velRef, '-', c='black', label = 'benchmark');
else:
    ax1.hist(position_ref, bins=numbins, density= True, alpha=0.5, label='benchmark');
    ax2.hist(velocity_ref, bins=numbins, density= True, alpha=0.5, label='benchmark');
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    pos, binEdges = np.histogram(position[i], bins=numbins, density = True)
    binsPos = 0.5 * (binEdges[1:] + binEdges[:-1])
    vel, binEdges = np.histogram(velocity[i], bins=numbins, density = True)
    binsVel = 0.5 * (binEdges[1:] + binEdges[:-1])
    if plotLines:
        if conditionedList[i] != bestCondition:
            ax1.plot(binsPos, pos, '--', label = labelList[index]);
            ax2.plot(binsVel, vel, '--', label = labelList[index]);
        else:
            ax1.plot(binsPos, pos, 'xk', label = labelList[index]);
            ax2.plot(binsVel, vel, 'xk', label = labelList[index]);
    else:
        ax1.hist(position[i], bins=numbins, density= True, alpha=0.5, label = labelList[i]);
        ax2.hist(velocity[i], bins=numbins, density= True, alpha=0.5, label = labelList[i]);


ax1.set_xlabel("position");
ax1.set_ylabel("distribution");
ax2.set_xlabel("velocity");
ax2.yaxis.tick_right()
ax2.legend(loc = 'lower left',  bbox_to_anchor=(-0.36, 0.5), framealpha = 1.0);

plt.savefig('distributions_comparisons_harmonic.pdf')
plt.clf()


# ## Autocorrelation function comparison


# Parameters for autocorrelation functions. Uses only a subset (mtrajs) of the
# total trajectories, since computing them with all is very slow
lagtimesteps = 40
mtrajs = 20
stridesPos = 50
stridesVel = 50


# Calculate reference autocorrelation functions
print('ACF for reference benchmark:')
ACF_ref_position = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs],
                                                                       lagtimesteps, stridesPos, 'position')
ACF_ref_velocity = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs],
                                                                       lagtimesteps, stridesVel, 'velocity')


# Calculate autocorrelation functions for reduced models
ACF_position = [None] * numConditions
ACF_velocity = [None] * numConditions
for i in range(numConditions):
    print('ACF conditioned on ' + conditionedList[i] + ' (' + str(i+1) + ' of ' + str(numConditions) + '):')
    currentTrajs = allTrajs[trajIndexes[i]]
    ACF_position[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs],
                                                                       lagtimesteps, stridesPos, 'position')
    ACF_velocity[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs],
                                                                       lagtimesteps, stridesVel, 'velocity')
    print('\nDone')


# Plot both autocorrelation functions at once
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), sharey=True) #fig, ax = plt.subplots()
time = dt*integratorStride*stridesPos*np.linspace(1,lagtimesteps,lagtimesteps)
ax1.plot(time, ACF_ref_position, '-k', label = 'benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    if conditionedList[i] != bestCondition:
        ax1.plot(time, ACF_position[i], '--', label = labelList[index])
    else:
        ax1.plot(time, ACF_position[i], 'xk', label = labelList[index])
ax1.set_xlabel('time(ns)')
ax1.set_ylabel('Position autocorrelation')

time = dt*integratorStride*stridesVel*np.linspace(1,lagtimesteps,lagtimesteps)
# Plot reference
ax2.plot(time, ACF_ref_velocity, '-k', label = 'benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    if conditionedList[i] != bestCondition:
        ax2.plot(time, ACF_velocity[i], '--', label = labelList[index])
    else:
        ax2.plot(time, ACF_velocity[i], 'xk', label = labelList[index])
ax2.set_xlabel('time(ns)')
ax2.set_ylabel('Velocity autocorrelation')
ax2.yaxis.tick_right()
plt.savefig('Autocorrelations_harmonic.pdf')
plt.clf()



