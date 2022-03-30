import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from deepRD.noiseSampler import binnedData
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
matplotlib.rcParams.update({'font.size': 15})
colorList = ['CC6677', '882255', 'AA4499','332288', '88CCEE', '44AA99','117733', '999933', 'DDCC77']
colorList2 = ['4477AA', 'EE6677', '228833', 'CCBB44', '66CCEE', 'AA3377', 'BBBBBB']
colorList3 = ['0077BB', '33BBEE', '009988', 'EE7733', 'CC3311', 'EE3377', 'BBBBBB']
colorList3alt = ['EE7733', 'A50026', '0077BB', '009988', '33BBEE', 'BBBBBB']
colorList3alt2 = ['A50026', '0077BB', '009988', '33BBEE', 'BBBBBB', 'EE7733']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colorList3alt2)

bsize = 5

# Benchmark data folder
#parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/harmonic/benchmarkComparison/'
parentDirectory = os.environ['DATA'] + 'stochasticClosure/harmonic/boxsize' + str(bsize) + '/benchmarkComparison/'
benchmarkfnamebase = parentDirectory + 'simMoriZwanzig_'
# Reduced models data folders
#localDataDirectory = '../../data/stochasticClosure/harmonic/benchmarkReduced'
localDataDirectory = os.environ['DATA'] + 'stochasticClosure/harmonic/boxsize' + str(bsize) + '/benchmarkReduced'
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

# Create plot directory
plotDirectory = os.environ['DATA'] + 'stochasticClosure/harmonic/boxsize' + str(bsize) + '/plots/'
try:
    os.makedirs(plotDirectory)
except OSError as error:
    print('Folder ' + plotDirectory + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Read relevant parameters
parameterDictionary = analysisTools.readParameters(parentDirectory + "parameters")
numSimulations = parameterDictionary['numFiles']
dt = parameterDictionary['dt'] 
integratorStride = parameterDictionary['stride']
totalTimeSteps = parameterDictionary['timesteps'] 
boxsize = parameterDictionary['boxsize']
boundaryType = parameterDictionary['boundaryType']
parameterDictionary

if bsize != boxsize:
    print('Requested boxsize does not match simulation')


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
            print("File ", j+1, " of ", numSimulations, " done.", end="\r")
        allTrajs[i] = iTraj
    except:
        continue
print("Reduced models data loaded.")


# ## Distribution plots comparisons


# Choose which reduced model to compare (just uncomment one)
#conditionedList = ['ri', 'qiri', 'pi', 'piri'] #Possibilities 'qi', 'ri', 'qiri', 'qiririm'
conditionedList = ['pi', 'piri', 'piririm']
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
# Note in plot labels x = q and v = p
labelList = [r'$\tilde{r}_{i+1}|\tilde{r}_i$', r'$\tilde{r}_{i+1}|\tilde{r}_i, \tilde{r}_{i-1}$',
             r'$\tilde{r}_{i+1}|\tilde{x}_i$', r'$\tilde{r}_{i+1}|\tilde{x}_i,\tilde{r}_i$',
             r'$\tilde{r}_{i+1}|\tilde{x}_i, \tilde{r}_i, \tilde{r}_{i-1}$',
             r'$\tilde{r}_{i+1}|\tilde{v}_i$', r'$\tilde{r}_{i+1}|\tilde{v}_i,\tilde{r}_i$',
             r'$\tilde{r}_{i+1}|\tilde{v}_i, \tilde{r}_i, \tilde{r}_{i-1}$']
#lineTypeList = [':', '-.', '--', 'xk']*2
#lwList = [4, 2, 2, 2]*2

lineTypeList = ['-.', '--', 'xk']*2
lwList = [2, 2, 2]*2




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
    ax1.plot(binsPosRef, posRef, '-k', label = 'benchmark');
    ax2.plot(binsVelRef, velRef, '-k', label = 'benchmark');
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
        ax1.plot(binsPos, pos, lineTypeList[i], lw = lwList[i], label = labelList[index]);
        ax2.plot(binsVel, vel, lineTypeList[i], lw = lwList[i], label = labelList[index]);
    else:
        ax1.hist(position[i], bins=numbins, density= True, alpha=0.5, label = labelList[i]);
        ax2.hist(velocity[i], bins=numbins, density= True, alpha=0.5, label = labelList[i]);


ax1.set_xlabel("position (nm)");
ax1.set_ylabel("distribution");
ax2.set_xlabel("velocity (nm/ns)");
ax2.yaxis.tick_right()
ax2.legend(loc = 'lower left',  bbox_to_anchor=(-0.36, 0.5), framealpha = 1.0);
plt.savefig(plotDirectory + 'distributions_comparisons_harmonic.pdf', bbox_inches='tight')
plt.clf()


# ## Autocorrelation function comparison

# Parameters for autocorrelation functions. Uses only a subset (mtrajs) of the 
# total trajectories, since computing them with all is very slow
lagtimesteps = 40
mtrajs = 20
stridesPos = 50
stridesVel = 50
stridesAux = 1


# Calculate reference autocorrelation functions
print('ACF for reference benchmark:')
ACF_ref_position = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs], 
                                                                       lagtimesteps, stridesPos, 'position')
ACF_ref_velocity = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs], 
                                                                       lagtimesteps, stridesVel, 'velocity')
ACF_ref_raux = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs],
                                                                       lagtimesteps, stridesAux, 'raux')


# Calculate autocorrelation functions for reduced models
ACF_position = [None] * numConditions
ACF_velocity = [None] * numConditions
ACF_raux = [None] * numConditions
for i in range(numConditions):
    print('\nACF conditioned on ' + conditionedList[i] + ' (' + str(i+1) + ' of ' + str(numConditions) + '):')
    currentTrajs = allTrajs[trajIndexes[i]]
    ACF_position[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs], 
                                                                       lagtimesteps, stridesPos, 'position')
    ACF_velocity[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs], 
                                                                       lagtimesteps, stridesVel, 'velocity')
    ACF_raux[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs],
                                                                       lagtimesteps, stridesAux, 'rauxReduced')
    print('\nDone')


# Plot three autocorrelation functions at once
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,4)) #fig, ax = plt.subplots()
time = dt*integratorStride*stridesPos*np.linspace(1,lagtimesteps,lagtimesteps)
ax1.plot(time, ACF_ref_position, '-k', label = 'benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    ax1.plot(time, ACF_position[i], lineTypeList[i], lw = lwList[i], label = labelList[index])
ax1.set_xlabel('time(ns)')
ax1.set_ylabel('Position autocorrelation')

time = dt*integratorStride*stridesVel*np.linspace(1,lagtimesteps,lagtimesteps)
# Plot reference
ax2.plot(time, ACF_ref_velocity, '-k', label = 'benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    ax2.plot(time, ACF_velocity[i], lineTypeList[i], lw = lwList[i], label = labelList[index])
ax2.set_xlabel('time(ns)')
ax2.set_ylabel('Velocity autocorrelation')
ax2.yaxis.tick_right()

time = dt*integratorStride*stridesAux*np.linspace(1,lagtimesteps,lagtimesteps)
ax3.plot(time, ACF_ref_raux, '-k', label = 'benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    ax3.plot(time, ACF_raux[i], lineTypeList[i], lw = lwList[i], label = labelList[index])
ax3.set_xlabel('time(ns)')
ax3.set_ylabel(r'$r$ autocorrelation')

plt.tight_layout()
plt.savefig(plotDirectory + 'Autocorrelations_harmonic.pdf')
plt.clf()