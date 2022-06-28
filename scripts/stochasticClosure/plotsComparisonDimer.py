import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools

matplotlib.rcParams.update({'font.size': 15})
colorList = ['CC6677', '882255', 'AA4499', '332288', '88CCEE', '44AA99', '117733', '999933', 'DDCC77']
colorList2 = ['4477AA', 'EE6677', '228833', 'CCBB44', '66CCEE', 'AA3377', 'BBBBBB']
colorList3 = ['0077BB', '33BBEE', '009988', 'EE7733', 'CC3311', 'EE3377', 'BBBBBB']
colorList3alt = ['EE7733', 'A50026', '0077BB', '009988', '33BBEE', 'BBBBBB']
colorList3alt2 = ['A50026', '0077BB', '009988', '33BBEE', 'BBBBBB', 'EE7733']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colorList3alt2)

bsize = 5

# Benchmark data folder
parentDirectory = os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize) + '/benchmarkComparison/'
benchmarkfnamebase = parentDirectory + 'simMoriZwanzig_'
# Reduced models data folders
localDataDirectory = os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize) + '/benchmarkReduced'
numModels = 8
redModelfnamebase = [localDataDirectory] * numModels
redModelfnamebase[0] += '_ri/simMoriZwanzigReduced_'
redModelfnamebase[1] += '_ririm/simMoriZwanzigReduced_'
redModelfnamebase[2] += '_qi/simMoriZwanzigReduced_'
redModelfnamebase[3] += '_qiri/simMoriZwanzigReduced_'
redModelfnamebase[4] += '_qiririm/simMoriZwanzigReduced_'
redModelfnamebase[5] += '_pi/simMoriZwanzigReduced_'
redModelfnamebase[6] += '_piri/simMoriZwanzigReduced_'
redModelfnamebase[7] += '_piririm/simMoriZwanzigReduced_'

# Create plot directory
plotDirectory = os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize) + '/plots/'
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

# Load benchmark trajectory data from h5 files (only of distinguished particles)
trajs_ref = []
print("Loading benchmark data ...")
for i in range(numSimulations):
    traj = trajectoryTools.loadTrajectory(benchmarkfnamebase, i)
    trajs_ref.append(traj)
    print("File ", i + 1, " of ", numSimulations, " done.", end="\r")
print("Benchmark data loaded.")

# Extract reference trajectories of the two particles
trajs1_ref = trajectoryTools.extractParticleTrajectories(trajs_ref,0,2)
trajs2_ref = trajectoryTools.extractParticleTrajectories(trajs_ref,1,2)

# Load reduced model trajectory data from h5 files (only of distinguished particle)
allTrajs = [None] * numModels
print("Loading reduced models data ...")
for i in range(numModels):
    try:
        iTraj = []
        for j in range(numSimulations):
            traj = trajectoryTools.loadTrajectory(redModelfnamebase[i], j)
            iTraj.append(traj)
            print("File ", i + 1, " of ", numSimulations, " done.", end="\r")
        allTrajs[i] = iTraj
    except:
        continue
print("Reduced models data loaded.")

# Extract reduced model trajectories of the two particles
allTrajs1 = [None] * numModels
allTrajs2 = [None] * numModels
for i in range(numModels):
    allTrajs1[i] = trajectoryTools.extractParticleTrajectories(allTrajs[i],0,2)
    allTrajs2[i] = trajectoryTools.extractParticleTrajectories(allTrajs[i],1,2)


# Choose which reduced model to compare (just uncomment one)
# conditionedList = ['ri', 'qiri', 'pi', 'piri'] #Possibilities 'qi', 'ri', 'qiri', 'qiririm'
conditionedList = ['piri']  # ['pi', 'piri', 'piririm']
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
labelList = [r'$\tilde{r}^{n+1}|\tilde{r}^n$', r'$\tilde{r}^{n+1}|\tilde{r}^n, \tilde{r}^{n-1}$',
             r'$\tilde{r}^{n+1}|\tilde{x}^n$', r'$\tilde{r}^{n+1}|\tilde{x}^n,\tilde{r}^n$',
             r'$\tilde{r}^{n+1}|\tilde{x}^n, \tilde{r}^n, \tilde{r}^{n-1}$',
             r'$\tilde{r}^{n+1}|\tilde{v}^n$', r'$\tilde{r}^{n+1}|\tilde{v}^n,\tilde{r}^n$',
             r'$\tilde{r}^{n+1}|\tilde{v}^n, \tilde{r}^n, \tilde{r}^{n-1}$']
# lineTypeList = [':', '-.', '--', 'xk']*2
# lwList = [4, 2, 2, 2]*2

lineTypeList = ['-.', '--', 'xk'] * 2
lwList = [2, 2, 2] * 2

# Extract variables to plot from trajectories (x components)
# varIndex = 1 # 1=x, 2=y, 3=z
position1 = [None] * numConditions
position2 = [None] * numConditions
velocity1 = [None] * numConditions
velocity2 = [None] * numConditions
for i in range(numConditions):
    currentTrajs1 = allTrajs1[trajIndexes[i]]
    currentTrajs2 = allTrajs2[trajIndexes[i]]
    position1[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs1, variableIndex=[1, 4])
    position2[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs2, variableIndex=[1, 4])
    velocity1[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs1, variableIndex=[4, 7])
    velocity2[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs2, variableIndex=[4, 7])
position1_ref = trajectoryTools.extractVariableFromTrajectory(trajs1_ref, variableIndex=[1, 4])
position2_ref = trajectoryTools.extractVariableFromTrajectory(trajs2_ref, variableIndex=[1, 4])
velocity1_ref = trajectoryTools.extractVariableFromTrajectory(trajs1_ref, variableIndex=[4, 7])
velocity2_ref = trajectoryTools.extractVariableFromTrajectory(trajs2_ref, variableIndex=[4, 7])



# ______________Work up until here (belows needs modification) __________________#

# Distribution plots comparisons
# Create plot
numbins = 40
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
ax1, ax2 = gs.subplots()  # sharey='row')
xlabel = [r'$x$', r'$y$', r'$z$']
# zerolabel = [r'$y=z=0$', r'$x=z=0$', r'$x=y=0$']
print('Plotting distributions ...')

# Plot distribution comparisons
for i in range(3):
    posRef, binEdges = np.histogram(position_ref[:, i], bins=numbins, density=True)
    binsPosRef = 0.5 * (binEdges[1:] + binEdges[:-1])
    velRef, binEdges = np.histogram(velocity_ref[:, i], bins=numbins, density=True)
    binsVelRef = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax1[i].plot(binsPosRef, posRef, '-k', label='benchmark');
    ax2[i].plot(binsVelRef, velRef, '-k', label='benchmark');
    for j in range(numConditions):
        index = trajIndexes[j]
        pos, binEdges = np.histogram(position[j][:, i], bins=numbins, density=True)
        binsPos = 0.5 * (binEdges[1:] + binEdges[:-1])
        vel, binEdges = np.histogram(velocity[j][:, i], bins=numbins, density=True)
        binsVel = 0.5 * (binEdges[1:] + binEdges[:-1])
        ax1[i].plot(binsPos, pos, lineTypeList[j], lw=lwList[j], label=labelList[index]);
        ax2[i].plot(binsVel, vel, lineTypeList[j], lw=lwList[j], label=labelList[index]);

    ax1[i].set_xlim((-boxsize / 2, boxsize / 2))
    ax1[i].set_ylim((0, None))
    ax1[i].set_xlabel(xlabel[i] + '-position')
    # if i==0:
    #    ax1[i].yaxis.set_ticks(np.arange(0,0.1,0.02))
    # else:
    #    ax1[i].yaxis.set_ticks(np.arange(0,0.03,0.01))

    ax2[i].set_xlim((-1, 1))
    ax2[i].set_ylim((0, None))
    ax2[i].set_xlabel(xlabel[i] + '-velocity')  # + '\n('+ zerolabel[i] +')')
    # ax2[i].xaxis.set_ticks(np.arange(-0.5,0.6,0.5))

ax1[2].legend(bbox_to_anchor=(0.6, 0., 0.5, 1.0), framealpha=1.0)

# displaying plot
# plt.tight_layout()
plt.savefig(plotDirectory + 'distributions_comparison_bistable.pdf')
plt.clf()
print('Distributions plots done.')

# ## Plot auto-correlation functions comparison

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
    print('\nACF conditioned on ' + conditionedList[i] + ' (' + str(i + 1) + ' of ' + str(numConditions) + '):')
    currentTrajs = allTrajs[trajIndexes[i]]
    ACF_position[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs],
                                                                       lagtimesteps, stridesPos, 'position')
    ACF_velocity[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs],
                                                                       lagtimesteps, stridesVel, 'velocity')
    ACF_raux[i] = trajectoryTools.calculateAutoCorrelationFunction(currentTrajs[0:mtrajs],
                                                                   lagtimesteps, stridesAux, 'rauxReduced')
    print('\nDone')

# Plot three autocorrelation functions at once
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))  # fig, ax = plt.subplots()
time = dt * integratorStride * stridesPos * np.linspace(1, lagtimesteps, lagtimesteps)
ax1.plot(time, ACF_ref_position, '-k', label='benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    ax1.plot(time, ACF_position[i], lineTypeList[i], lw=lwList[i], label=labelList[index])
ax1.set_xlabel('time(ns)')
ax1.set_ylabel('Position autocorrelation')
ax1.yaxis.tick_right()

time = dt * integratorStride * stridesVel * np.linspace(1, lagtimesteps, lagtimesteps)
# Plot reference
ax2.plot(time, ACF_ref_velocity, '-k', label='benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    ax2.plot(time, ACF_velocity[i], lineTypeList[i], lw=lwList[i], label=labelList[index])
ax2.set_xlabel('time(ns)')
ax2.set_ylabel('Velocity autocorrelation')
ax2.yaxis.tick_right()

time = dt * integratorStride * stridesAux * np.linspace(1, lagtimesteps, lagtimesteps)
ax3.plot(time, ACF_ref_raux, '-k', label='benchmark')
# Plot reduced models
for i in range(numConditions):
    index = trajIndexes[i]
    ax3.plot(time, ACF_raux[i], lineTypeList[i], lw=lwList[i], label=labelList[index])
ax3.set_xlabel('time(ns)')
ax3.set_ylabel(r'$r$ autocorrelation')
ax3.yaxis.tick_right()

plt.tight_layout()
plt.savefig(plotDirectory + 'Autocorrelations_bistable.pdf')
plt.clf()