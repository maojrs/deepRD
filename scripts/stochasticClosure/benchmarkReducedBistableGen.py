import numpy as np
import random
import os
import sys
import pickle
import deepRD
import torch
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
conditionedOn = 'piririm' # Available conditionings: qi, pi, ri, qiri, piri, qiririm, piririm
outputAux = True #False

# Output data directory
#foldername = 'bistable/boxsize' + str(bsize) + '/benchmarkReduced_' + conditionedOn
foldername = 'bistable/boxsize' + str(bsize) + '/benchmarkReducedGen_' + conditionedOn
outputDataDirectory = os.path.join(localDataDirectory, foldername)
# Create folder for data
try:
    os.makedirs(outputDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

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
loadPretrained = localModelDirectory + conditionedOn + '_E81.pt'

batch_norm=False
dropout_rate=0
cutoff=False # only sample up to a certain radius in latent space
normalize_data=True

if normalize_data:
    mean_input = 0
    std_input = np.array([0.0162, 0.0162, 0.0162])
    mean_cond = 0
    if conditionedOn=='piri':
        std_cond = np.array([0.0162, 0.0162, 0.0162, 0.1425, 0.1425, 0.1426])

    norm_params = 
else:
    norm_params = (0,1,0,1)

#norm_params_m2 = (np.array([4.9424e-05, 1.1248e-05, 7.7914e-05]), np.array([0.0163, 0.0163, 0.0162]), 
#                np.array([ 4.9414e-05,  1.1577e-05,  7.7847e-05,  5.8293e-04, -1.0487e-04, -2.5263e-04]),
#                np.array([0.0163, 0.0163, 0.0162, 0.1413, 0.1438, 0.1410]))
#norm_params_N1 = (np.array([ 5.2723e-06,  3.6534e-06, -4.3195e-06]), np.array([0.0162, 0.0162, 0.0162]), np.array([ 5.1132e-06,  3.7120e-06, -4.3634e-06,  1.0060e-04, -9.1961e-05,
#         1.0791e-05]), np.array([0.0162, 0.0162, 0.0162, 0.1425, 0.1426, 0.1425]))
#norm_params_N2 = (np.array([-2.1970e-06,  5.2476e-06, -2.8618e-06]), np.array([0.0162, 0.0162, 0.0162]), np.array([-2.2653e-06,  5.3580e-06, -2.9306e-06,  2.5850e-05, -4.7011e-05,
#         3.8208e-05]), np.array([0.0162, 0.0162, 0.0162, 0.1425, 0.1425, 0.1427]))
#N3 tensor([-4.5942e-07, -1.3397e-06, -5.2720e-07]) tensor([0.0162, 0.0162, 0.0162]) tensor([-5.1237e-07, -1.3432e-06, -6.0386e-07,  2.7447e-05, -1.2348e-04,
#         4.0647e-05]) tensor([0.0162, 0.0162, 0.0162, 0.1427, 0.1425, 0.1424])

hidden_dims = [128, 64, 32]
nSampler = cvaeSampler.cvaeSampler(8, loadPretrained, conditionedOn, 'bistable', hidden_dims, batch_norm=batch_norm, 
                                        dropout_rate=dropout_rate, norm_params=norm_params, sampling_width=1.5, cutoff=cutoff)
nSampler.eval()
#nSampler = cvaeSampler.defaultSamplingModel()


# Parameters for external potential (will only acts on distinguished particles (type 1)
minimaDist = 1.5
kconstants = np.array([1.0, 1.0, 1.0])
scalefactor = 1

# Integrator parameters
integratorStride = 1 #50
tfinal = 10000
equilibrationSteps = 10000

# Create parameter dictionary to write to parameters reference file
parameterfilename = os.path.join(outputDataDirectory, "parameters")
parameterDictionary = {'numFiles' : numSimulations, 'dt' : dt, 'Gamma' : Gamma, 'KbT' : KbT,
                       'mass' : mass, 'tfinal' : tfinal, 'stride' : integratorStride,
                       'boxsize' : boxsize, 'boundaryType' : boundaryType,
                       'equilibrationSteps' : equilibrationSteps, 'conditionedOn': conditionedOn}
analysisTools.writeParameters(parameterfilename, parameterDictionary)

# Provides base filename (folder must exist (and preferably empty), otherwise H5 might fail)
basefilename = os.path.join(outputDataDirectory, "simMoriZwanzigReduced_")

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


# Runs several simulations in parallel
print('Simulation for ri+1|' + conditionedOn + ' begins ...')
num_cores =  multiprocessing.cpu_count() - 1
pool = Pool(processes=num_cores)
iterator = [i for i in range(numSimulations)]
pool.map(partial(runParallelSims), iterator)

## Serial test
#for i in range(numSimulations):
#    runParallelSims(i)
