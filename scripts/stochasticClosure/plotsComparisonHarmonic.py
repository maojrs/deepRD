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
numSimulations = parameterDictionary['numFiles']
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
conditionedOn = 'qiri' #Possibilities 'qi', 'ri', 'qiri', 'qiririm'
if conditionedOn == 'ri':
    trajs = allTrajs[0] 
elif conditionedOn == 'ririm':
    trajs = allTrajs[1] 
elif conditionedOn == 'qi':
    trajs = allTrajs[2] 
elif conditionedOn == 'qiri':
    trajs = allTrajs[3] 
elif conditionedOn == 'qiririm':
    trajs = allTrajs[4] 
elif conditionedOn == 'pi':
    trajs = allTrajs[5] 
elif conditionedOn == 'piri':
    trajs = allTrajs[6] 
elif conditionedOn == 'piririm':
    trajs = allTrajs[7] 



# Extract variables to plot from tajectories (x components)
varIndex = 1 # 1=x, 2=y, 3=z
position = trajectoryTools.extractVariableFromTrajectory(trajs, variableIndex = varIndex)
velocity = trajectoryTools.extractVariableFromTrajectory(trajs, variableIndex = varIndex + 3)
position_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = varIndex)
velocity_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = varIndex + 3)



# Plot distribution comparison for position
plotLines = True
numbins = 40
pos, binEdges = np.histogram(position, bins=numbins, density = True)
binsPos = 0.5 * (binEdges[1:] + binEdges[:-1])
posRef, binEdges = np.histogram(position_ref, bins=numbins, density = True)
binsPosRef = 0.5 * (binEdges[1:] + binEdges[:-1])
fig, ax = plt.subplots()
if plotLines:
    ax.plot(binsPosRef, posRef, '-', c='black', label = 'benchmark');
    ax.plot(binsPos, pos, 'x', c='black', label = 'reduced');
else:
    ax.hist(position_ref, bins=numbins, density= True, alpha=0.5, label='benchmark');
    ax.hist(position, bins=numbins, density= True, alpha=0.5, label='reduced');
ax.set_xlabel("position");
ax.set_ylabel("distribution");
ax.legend();
plt.savefig('position_distribution_comparison_harmonic_' + conditionedOn +'.pdf')
plt.clf()


# Plot distirbution comparison for velocity
plotLines = True
numbins = 40
vel, binEdges = np.histogram(velocity, bins=numbins, density = True)
binsVel = 0.5 * (binEdges[1:] + binEdges[:-1])
velRef, binEdges = np.histogram(velocity_ref, bins=numbins, density = True)
binsVelRef = 0.5 * (binEdges[1:] + binEdges[:-1])
fig, ax = plt.subplots()
if plotLines:
    ax.plot(binsVelRef, velRef, '-', c='black', label = 'benchmark');
    ax.plot(binsVel, vel, 'x', c='black', label = 'reduced');
else:
    ax.hist(velocity_ref, bins=numbins, density= True, alpha=0.5, label='benchmark');
    ax.hist(velocity, bins=numbins, density= True, alpha=0.5, label='reduced');
ax.set_xlabel("velocity");
ax.set_ylabel("distribution");
ax.legend()
plt.savefig('velocity_distribution_comparison_harmonic_' + conditionedOn +'.pdf')
plt.clf()

# ## Autocorrelation function comparison


# Uses only a subset (mtrajs) of the total trajectories, since computing them with all is very slow
variables = ['position', 'velocity']
lagtimesteps = 40
mtrajs = 20
strides = [50,50]
ACF = [None]*2
ACF_ref = [None]*2
for i, var in enumerate(variables):
    mean = trajectoryTools.calculateMean(trajs[0:mtrajs], var)
    mean_ref = trajectoryTools.calculateMean(trajs_ref[0:mtrajs], var)
    variance = trajectoryTools.calculateVariance(trajs[0:mtrajs], var, mean)
    variance_ref = trajectoryTools.calculateVariance(trajs_ref[0:mtrajs], var, mean_ref)
    ACF[i] = trajectoryTools.calculateAutoCorrelationFunction(trajs[0:mtrajs], lagtimesteps, strides[i], var)
    ACF_ref[i] = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs], lagtimesteps, strides[i], var)




index = 0
time = dt*integratorStride*strides[index]*np.linspace(1,lagtimesteps,lagtimesteps)
plt.plot(time, ACF[index], 'xk', label = 'reduced')
plt.plot(time, ACF_ref[index], '-k', label = 'benchmark')
plt.xlabel('time(ns)')
plt.ylabel(variables[index] + ' autocorrelation')
plt.legend()
#plt.xlim([0,1500])
plt.savefig(variables[index]+ '_autocorrelation_harmonic_' + conditionedOn +'.pdf')
plt.clf()


index = 1
time = dt*integratorStride*strides[index]*np.linspace(1,lagtimesteps,lagtimesteps)
plt.plot(time, ACF[index], 'xk', label = 'reduced')
plt.plot(time, ACF_ref[index], '-k', label = 'benchmark')
plt.xlabel('time(ns)')
plt.ylabel(variables[index] + ' autocorrelation')
plt.legend()
#plt.xlim([0,1500])
plt.savefig(variables[index]+ '_autocorrelation_harmonic_' + conditionedOn +'.pdf')
plt.clf()



