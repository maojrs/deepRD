import numpy as np
import random
import os
import sys
import pickle
import deepRD
from deepRD.diffusionIntegrators import langevinNoiseSampler
#from deepRD.diffusionIntegrators import langevinInteractionSampler
from deepRD.potentials import bistable
from deepRD.noiseSampler import noiseSampler
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
localDataDirectory = os.environ['DATA'] + 'stochasticClosureGen/'
numSimulations = 100
bsize = 8 #5 #8 #10
conditionedOn = 'piri' # Available conditionings: qi, pi, ri, qiri, piri, qiririm, piririm
outputAux = True #False

# Output data directory
#foldername = 'bistable/boxsize' + str(bsize) + '/benchmarkReduced_' + conditionedOn
foldername = 'bistable/boxsize' + str(bsize) + '/benchmarkReduced_' + conditionedOn
outputDataDirectory = os.path.join(localDataDirectory, foldername)
# Create folder for data
try:
    os.makedirs(outputDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Load binning sampling models
print("Loading binned data ...")
#binnedDataFilename = localDataDirectory + 'bistable/boxsize' + str(bsize) + '/binnedData/' + conditionedOn + 'BinnedData.pickle'
binnedDataFilename = localDataDirectory + 'bistable/boxsize' + str(bsize) + '/binnedData/' + conditionedOn + 'BinnedData.pickle'
dataOnBins = pickle.load(open(binnedDataFilename, "rb" ))
parameters = dataOnBins.parameterDictionary
print('Binned data loaded')
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

# Extract binning parameters
numbins = parameters['numbins']
lagTimesteps = parameters['lagTimesteps']
nsigma = parameters['nsigma']

# Define noise sampler
nSampler = noiseSampler(dataOnBins)

# Parameters for external potential (will only acts on distinguished particles (type 1))
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
                       'equilibrationSteps' : equilibrationSteps, 'conditionedOn': conditionedOn,
                       'numbins': numbins, 'lagTimesteps': lagTimesteps, 'nsigma': nsigma}
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
