import numpy as np
import random
import os
import sys
import pickle
import deepRD
from deepRD.diffusionIntegrators import langevinNoiseSampler
from deepRD.potentials import bistable
from deepRD.noiseSampler import noiseSampler
from deepRD.noiseSampler import binnedData
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools

import multiprocessing
from multiprocessing import Pool, Process, Manager
from multiprocessing.managers import BaseManager
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
numSimulations = 10000 #100
bsize= 5
conditionedOn = 'piri' # Available conditionings: qi, pi, ri, qiri, piri, qiririm, piririm

# Output data directory
foldername = 'bistable/boxsize' + str(bsize) + '/benchmarkFPTreduced'
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
binnedDataFilename = localDataDirectory + 'bistable/boxsize' + str(bsize) + '/binnedData/' + conditionedOn + 'BinnedData.pickle'
#binnedDataFilename = localDataDirectory + 'binnedData/riBinnedData.pickle'
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

# Extract binning parameters
numbins = parameters['numbins']
lagTimesteps = parameters['lagTimesteps']
nsigma = parameters['nsigma']

if bsize != boxsize:
    print('Requested boxsize does not match simulation')

# Define noise sampler
nSampler = noiseSampler(dataOnBins)

# Parameters for external potential (will only acts on distinguished particles (type 1))
minimaDist = 1.5
kconstants = np.array([1.0, 1.0, 1.0])
scalefactor = 1

# Integrator parameters
integratorStride = 1 #50
tfinal = 10000
equilibrationSteps = 0

# Parameters for FPT calculations
initialPosition = np.array([-1.0*minimaDist, 0., 0.])
finalPosition = np.array([1.0*minimaDist, 0., 0.])
minimaThreshold = 0.3

# Create parameter dictionary to write to parameters reference file
parameterfilename = os.path.join(outputDataDirectory, "parameters")
parameterDictionary = {'numFiles' : numSimulations, 'dt' : dt, 'Gamma' : Gamma, 'KbT' : KbT,
                       'mass' : mass, 'tfinal' : tfinal, 'stride' : integratorStride,
                       'boxsize' : boxsize, 'boundaryType' : boundaryType,
                       'conditionedOn': conditionedOn, 'numbins': numbins, 'lagTimesteps': lagTimesteps,
                       'nsigma': nsigma}
analysisTools.writeParameters(parameterfilename, parameterDictionary)

# Provides base filename (folder must exist (and preferably empty), otherwise H5 might fail)
basefilename = os.path.join(outputDataDirectory, "simMoriZwanzigReduced_")

# Create empty files to save the data in parallel algorithm
filename = outputDataDirectory  + '/simMoriZwanzigFPTs_' + conditionedOn + '_box' + str(boxsize) + '_nsims' + str(numSimulations) + '.xyz'

# Simulation wrapper for parallel runs
def runParallelSims(simnumber):
    # Define particle list
    seed = int(simnumber)
    random.seed(seed)
    position = initialPosition
    velocity = [0, 0, 0]
    particle = deepRD.particle(position, velocity = velocity, mass=mass)
    particleList = deepRD.particleList([particle])

    # Define external potential
    bistablePotential = bistable(minimaDist, kconstants, scalefactor)

    diffIntegrator = langevinNoiseSampler(dt, integratorStride, tfinal, Gamma, nSampler, KbT,
                                          boxsize, boundaryType, equilibrationSteps, conditionedOn)
    diffIntegrator.setExternalPotential(bistablePotential)

    # Integrate dynamics
    result, FPT = diffIntegrator.propagateFPT(particleList, finalPosition, minimaThreshold)

    return result, FPT


def multiprocessingHandler():
    '''
    Handles parallel processing of simulationFPT and writes to same file in parallel
    '''
    # Runs several simulations in parallel
    num_cores = multiprocessing.cpu_count() - 1
    pool = Pool(processes=num_cores)
    trajNumList = list(range(numSimulations))
    with open(filename, 'w') as file:
        for index, result in enumerate(pool.imap(runParallelSims, trajNumList)):
            status, time = result
            if status == 'success':
                file.write(str(time) + '\n')
                print("Simulation " + str(index) + ", done. Success!")
            else:
                print("Simulation " + str(index) + ", done. Failed :(")

# Run parallel code
multiprocessingHandler()

## Serial test
#for i in range(numSimulations):
#    runParallelSims(i)
