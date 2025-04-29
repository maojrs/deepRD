import numpy as np
import random
import os
import sys
import pickle
import deepRD
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from deepRD.diffusionIntegrators import langevinNoiseSampler
#from deepRD.diffusionIntegrators import langevinInteractionSampler
from deepRD.potentials import bistable
from deepRD.noiseSampler import cvaeSampler
#from deepRD.noiseSampler import binnedData
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools

import multiprocessing
from multiprocessing import Pool
from functools import partial

'''
Script to run benchmarks and plot directly for multiple models consequently.
Runs reduced model by stochastic closure with same parameters as benchmark for comparison. 
'''

## Units

# ### Boltzman constant
# - $k_B = 1.38064852 \times 10^{-23} \frac{m^2}{s^2} \frac{kg}{K} \left(= \frac{nm^2}{ns^2}\frac{kg}{K}\right)$.
#
# ### Basic units
# - Length (l): nanometers (nm)
# - Energy ($\epsilon$) : $k_B T = g/mol \frac{nm^2}{ns^2}$
# - Mass (m): gram/mol (g/mol)
#
# ### Derived units
# - Time: $l\sqrt{m/\epsilon} = ns$
# - Temperature: $\epsilon/k_B =$ Kelvin ($K$)
# - Force: $\epsilon/l = kg \frac{nm}{ns^2}$
#
# ### Reduced quantities (dimensionless)
# - Reduced pair potential: $U^* = U/\epsilon$
# - Reduced force: $F^* = F l /\epsilon$
# - Reduced distance: $r^* = r/l$
# - Reduced density: $\rho^*=\rho l^3$
# - Reduced Temperature: $T^* = k_B T/\epsilon$
# - Reduced Pressure: $P^* = Pl^3/\epsilon$
# - Reduced friction: $\sigma^2/time$

# Simulation parameters
#localDataDirectory = '../../data/stochasticClosure/'
localDataDirectory = os.environ['DATA'] + 'stochasticClosure/'
numSimulations = 100
bsize = 5 #5 #8 #10
systemType = 'bistable'
conditionedOn = 'piri' # Available conditionings: qi, pi, ri, qiri, piri, qiririm, piririm
outputAux = True #False

# Output data directory
foldername = 'bistable/boxsize' + str(bsize) + '/benchmarkReducedGenInt_' + conditionedOn
outputDataDirectory = os.path.join(localDataDirectory, foldername)

try:
    os.makedirs(outputDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Overwriting.')
# Loading parameter dictionary
parentDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize)+ '/benchmark/'
parameters = analysisTools.readParameters(parentDirectory + "parameters")
#print(parameters)

# Extract basic parameters
dt = parameters['dt']
Gamma = parameters['Gamma']
mass =  parameters['mass']
KbT = parameters['KbT']
boxsize = parameters['boxsize']
boundaryType = parameters['boundaryType']

if bsize != boxsize:
    print('Requested boxsize does not match simulation')

# Define noise sampler, n latent dims
localModelDirectory = 'deepRD/noiseSampler/models/modelWeights/model_state_'

# CVAE model settings
hidden_dims = [128, 64, 32]
latentDims = 3
batch_norm=False
dropout_rate=0
cutoff=False # only sample up to a certain radius in latent space
sampling_width=1
sampling_scale=1
normalize_data=False


# Each model to be trained is represented by a dictionary which contains all the parameters which are to be changed.
# Parameters which are unspecified in the dictionary will be set to default values as set above.

modelsToEvaluate = [
    {
        'modelName': 'N1_3e2',
        'outputModelName': 'N1_3e2_scale1',
        'latentDims': 3,
        'normalize_data': True,
        'sampling_scale': 1
    },
    {
        'modelName': 'N1_3e2',
        'outputModelName': 'N1_3e2_scale12',
        'latentDims': 3,
        'normalize_data': True,
        'sampling_scale': 1.2
    },
    {
        'modelName': 'N1_3e2',
        'outputModelName': 'N1_3e2_scale15',
        'latentDims': 3,
        'normalize_data': True,
        'sampling_scale': 1.5
    },
    {
        'modelName': 'N1_3e2',
        'outputModelName': 'N1_3e2_scale2',
        'latentDims': 3,
        'normalize_data': True,
        'sampling_scale': 2
    },
    {
        'modelName': 'N1_3e2',
        'outputModelName': 'N1_3e2_scale5',
        'latentDims': 3,
        'normalize_data': True,
        'sampling_scale': 5
    },
    {
        'modelName': 'N1_3e2',
        'outputModelName': 'N1_3e2_scale10',
        'latentDims': 3,
        'normalize_data': True,
        'sampling_scale': 10
    }
]

# Simulation wrapper for parallel runs
def runParallelSims(simnumber):
    #if simnumber % 2 == 0:
    #    sign = 1
    #else:
    #    sign= -1

    # Define particle list
    seed = int(simnumber)
    random.seed(seed)
    position = [0, 0, 0]
    velocity = [0, 0, 0]
    particle = deepRD.particle(position, velocity = velocity, mass=mass)
    particleList = deepRD.particleList([particle])

    # Define external potential
    bistablePotential = bistable(minimaDist, kconstants, scalefactor)

    diffIntegrator = langevinNoiseSampler(dt, integratorStride, tfinal, Gamma, nSampler, KbT,
                                          boxsize, boundaryType, equilibrationSteps, conditionedOn)

    #diffIntegrator = langevinInteractionSampler(dt, integratorStride, tfinal, Gamma, nSampler, KbT,
    #                                            boxsize, boundaryType, equilibrationSteps, conditionedOn)

    diffIntegrator.setExternalPotential(bistablePotential)

    # Integrate dynamics
    #t, X, V = diffIntegrator.propagate(particleList, outputAux = outputAux)
    t, X, V, Raux = diffIntegrator.propagate(particleList, outputAux = outputAux)


    # Write dynamics into trjactory
    #traj = trajectoryTools.convert2trajectory(t, [X, V])
    traj = trajectoryTools.convert2trajectory(t, [X, V, Raux])
    trajectoryTools.writeTrajectory(traj,basefilename,simnumber)

    print("Simulation " + str(simnumber) + ", done.")

# Parameters for external potential (will only acts on distinguished particles (type 1)
minimaDist = 1.5
kconstants = np.array([1.0, 1.0, 1.0])
scalefactor = 1

# Integrator parameters
integratorStride = 1 #50
tfinal = 10000
equilibrationSteps = 10000

# Create parameter dictionary to write to parameters reference file
#parameterfilename = os.path.join(outputDataDirectory, "parameters")
#parameterDictionary = {'numFiles' : numSimulations, 'dt' : dt, 'Gamma' : Gamma, 'KbT' : KbT,
#                    'mass' : mass, 'tfinal' : tfinal, 'stride' : integratorStride,
#                    'boxsize' : boxsize, 'boundaryType' : boundaryType,
#                    'equilibrationSteps' : equilibrationSteps, 'conditionedOn': conditionedOn}
#analysisTools.writeParameters(parameterfilename, parameterDictionary)


# Benchmark data folder
#parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/bistable/benchmarkComparison/'
parentDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize) + '/benchmarkComparison/'
benchmarkfnamebase = parentDirectory + 'simMoriZwanzig_'
# Reduced models data folders
#localDataDirectory = '../../data/stochasticClosure/bistable/benchmarkReduced'
localDataDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize) + '/benchmarkReducedGenInt'
numModels = 9
redModelfnamebase = [localDataDirectory]*numModels
redModelfnamebase[0] += '_ri/simMoriZwanzigReduced_'
redModelfnamebase[1] += '_ririm/simMoriZwanzigReduced_'
redModelfnamebase[2] += '_qi/simMoriZwanzigReduced_'
redModelfnamebase[3] += '_qiri/simMoriZwanzigReduced_'
redModelfnamebase[4] += '_qiririm/simMoriZwanzigReduced_'
redModelfnamebase[5] += '_pi/simMoriZwanzigReduced_'
redModelfnamebase[6] += '_piri/simMoriZwanzigReduced_'
redModelfnamebase[7] += '_piririm/simMoriZwanzigReduced_'
redModelfnamebase[8] += '_pipimri/simMoriZwanzigReduced_' # additional model

# Read relevant parameters
#parameterDictionary = analysisTools.readParameters(parentDirectory + "parameters")
#numSimulations = parameterDictionary['numFiles']
#dt = parameterDictionary['dt'] 
#integratorStride = parameterDictionary['stride']
#totalTimeSteps = parameterDictionary['timesteps'] 
#boxsize = parameterDictionary['boxsize']
#boundaryType = parameterDictionary['boundaryType']
#parameterDictionary

# Provides base filename (folder must exist (and preferably empty), otherwise H5 might fail)
basefilename = os.path.join(outputDataDirectory, "simMoriZwanzigReduced_")

# Here the part of the script responsible for plotting
matplotlib.rcParams.update({'font.size': 15})
colorList = ['CC6677', '882255', 'AA4499','332288', '88CCEE', '44AA99','117733', '999933', 'DDCC77']
colorList2 = ['4477AA', 'EE6677', '228833', 'CCBB44', '66CCEE', 'AA3377', 'BBBBBB']
colorList3 = ['0077BB', '33BBEE', '009988', 'EE7733', 'CC3311', 'EE3377', 'BBBBBB']
colorList3alt = ['EE7733', 'A50026', '0077BB', '009988', '33BBEE', 'BBBBBB']
colorList3alt2 = ['A50026', '0077BB', '009988', '33BBEE', 'BBBBBB', 'EE7733']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=colorList3alt2)

# Plot settings
plotDistributions = True
plotACFs = False
plotFPTs = False

# Choose which reduced model to compare (just uncomment one)
#conditionedList = ['ri', 'qiri', 'pi', 'piri'] #Possibilities 'qi', 'ri', 'qiri', 'qiririm'
conditionedList = ['piri']#, 'piririm', 'pipimri'] #['pi','piri','piririm'] #['pi', 'piri', 'piririm']
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
if 'pipimri' in conditionedList:
    trajIndexes.append(8)
numConditions = len(trajIndexes)
# Note in plot labels x = q and v = p
labelList = [r'$\tilde{r}^{n+1}|\tilde{r}^n$', r'$\tilde{r}^{n+1}|\tilde{r}^n, \tilde{r}^{n-1}$',
             r'$\tilde{r}^{n+1}|\tilde{x}^n$', r'$\tilde{r}^{n+1}|\tilde{x}^n,\tilde{r}^n$',
             r'$\tilde{r}^{n+1}|\tilde{x}^n, \tilde{r}^n, \tilde{r}^{n-1}$',
             r'$\tilde{r}^{n+1}|\tilde{v}^n$', r'$\tilde{r}^{n+1}|\tilde{v}^n,\tilde{r}^n$',
             r'$\tilde{r}^{n+1}|\tilde{v}^n, \tilde{r}^n, \tilde{r}^{n-1}$', 
             r'$\tilde{r}^{n+1}|\tilde{v}^n, \tilde{r}^n, \tilde{v}^{n-1}$']
#lineTypeList = [':', '-.', '--', 'xk']*2
#lwList = [4, 2, 2, 2]*2

lineTypeList = ['-.', '--', 'xk']*2
lwList = [2, 2, 2]*2

# Running benchmark simulations and plotting
for model in modelsToEvaluate:

     # Setting the variables specified in the model dictionary
    for key, val in model.items():
        print(key,' = ', val)
        exec(key + '=val')

    if not 'outputModelName' in model:
        outputModelName = modelName

    loadPretrained = localModelDirectory + conditionedOn + '_' + modelName + '.pt'

    if normalize_data:
        if systemType=='bistable':
            mean_input = np.array([2.0080e-07,  6.3679e-06, -2.9733e-06])
            std_input = np.array([1.6200e-02, 1.6212e-02, 1.6207e-02])
            mean_cond = np.array([2.0080e-07, 6.3679e-06, -2.9733e-06, -1.2232e-05, -7.2127e-05,  6.9702e-05])
            if conditionedOn=='piri':
                std_cond = np.array([1.6200e-02, 1.6212e-02, 1.6207e-02, 1.4253e-01, 1.4252e-01, 
                            1.4256e-01])

        norm_params = (mean_input, std_input, mean_cond, std_cond)
    else:
        norm_params = (0,1,0,1)


    nSampler = cvaeSampler.cvaeSampler(latentDims, loadPretrained, conditionedOn, 'bistable', hidden_dims, batch_norm=batch_norm, 
                                        dropout_rate=dropout_rate, norm_params=norm_params, sampling_width=sampling_width, 
                                        cutoff=cutoff, sampling_scale=sampling_scale)
                                        
    nSampler.eval()
    #nSampler = cvaeSampler.defaultSamplingModel()

    # Runs several simulations in parallel
    print('Simulation for ri+1|' + conditionedOn + ' begins ...')
    num_cores =  multiprocessing.cpu_count() - 1
    pool = Pool(processes=num_cores)
    iterator = [i for i in range(numSimulations)]
    pool.map(partial(runParallelSims), iterator)

    print('Starting plotting')

    # Create plot directory
    plotDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize) + '/plotsGen_' + outputModelName + '/'
    try:
        os.makedirs(plotDirectory)
    except OSError as error:
        print('Folder ' + plotDirectory + ' already exists. Overwriting.')

    # ## Load benchmark and reduced model trajectory data
    if (plotDistributions):
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

    if (plotDistributions):
        # Extract variables to plot from trajectories (x components)
        #varIndex = 1 # 1=x, 2=y, 3=z
        position = [None] * numConditions
        velocity = [None] * numConditions
        for i in range(numConditions):
            currentTrajs = allTrajs[trajIndexes[i]]
            position[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs, variableIndex = [1,4])
            velocity[i] = trajectoryTools.extractVariableFromTrajectory(currentTrajs, variableIndex = [4,7])
        position_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = [1,4])
        velocity_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = [4,7])


        # Distribution plots comparisons
        # Create plot
        numbins = 40
        fig = plt.figure(figsize=(15,8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
        ax1, ax2 = gs.subplots() #sharey='row')
        xlabel = [r'$x$', r'$y$', r'$z$']
        #zerolabel = [r'$y=z=0$', r'$x=z=0$', r'$x=y=0$']
        print('Plotting distributions ...')

        # Plot distribution comparisons
        for i in range(3):
            posRef, binEdges = np.histogram(position_ref[:,i], bins=numbins, density=True)
            binsPosRef = 0.5 * (binEdges[1:] + binEdges[:-1])
            velRef, binEdges = np.histogram(velocity_ref[:,i], bins=numbins, density=True)
            binsVelRef = 0.5 * (binEdges[1:] + binEdges[:-1])
            ax1[i].plot(binsPosRef, posRef, '-k', label='benchmark');
            ax2[i].plot(binsVelRef, velRef, '-k', label='benchmark');
            for j in range(numConditions):
                index = trajIndexes[j]
                pos, binEdges = np.histogram(position[j][:,i], bins=numbins, density=True)
                binsPos = 0.5 * (binEdges[1:] + binEdges[:-1])
                vel, binEdges = np.histogram(velocity[j][:,i], bins=numbins, density=True)
                binsVel = 0.5 * (binEdges[1:] + binEdges[:-1])
                ax1[i].plot(binsPos, pos, lineTypeList[j], lw=lwList[j], label=labelList[index]);
                ax2[i].plot(binsVel, vel, lineTypeList[j], lw=lwList[j], label=labelList[index]);

            ax1[i].set_xlim((-boxsize/2,boxsize/2))
            ax1[i].set_ylim((0,None))
            ax1[i].set_xlabel(xlabel[i] + '-position')
            #if i==0:
            #    ax1[i].yaxis.set_ticks(np.arange(0,0.1,0.02))
            #else:
            #    ax1[i].yaxis.set_ticks(np.arange(0,0.03,0.01))

            ax2[i].set_xlim((-1,1))
            ax2[i].set_ylim((0,None))
            ax2[i].set_xlabel(xlabel[i] + '-velocity') # + '\n('+ zerolabel[i] +')')
            #ax2[i].xaxis.set_ticks(np.arange(-0.5,0.6,0.5))

        ax1[2].legend(bbox_to_anchor=(0.6, 0., 0.5, 1.0), framealpha=1.0) # for box8
        #ax1[2].legend(bbox_to_anchor=(0.7, 0., 0.5, 1.0), framealpha=1.0) #for box5

        # displaying plot
        #plt.tight_layout()
        plt.savefig(plotDirectory + 'distributions_comparison_bistable.pdf')
        plt.clf()
        print('Distributions plots done.')


    if (plotACFs):
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
        ax1.yaxis.tick_right()


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
        ax3.yaxis.tick_right()

        plt.tight_layout()
        plt.savefig(plotDirectory + 'Autocorrelations_bistable.pdf')
        plt.clf()


