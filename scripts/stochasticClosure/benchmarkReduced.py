import numpy as np
import random
import os
import sys
import pickle
import deepRD
from deepRD.diffusionIntegrators import langevinNoiseSampler
from deepRD.potentials import harmonic
from deepRD.noiseSampler import noiseSampler
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools

import multiprocessing
from multiprocessing import Pool
from functools import partial

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
localDataDirectory = '../../data/stochasticClosure/'
numSimulations = 100 #100
conditionedOn = 'qi' # Available conditionings: qi, ri, qiri, qiririm

# Output data directory
foldername = 'benchmarkReduced_' + conditionedOn
outputDataDirectory = os.path.join(localDataDirectory, foldername)
# Create folder for data
try:
    os.mkdir(outputDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Load binning sampling models
print("Loading binned data ...")
binnedDataFilename = localDataDirectory + 'binnedData/' + conditionedOn + 'BinnedData.pickle'
#binnedDataFilename = localDataDirectory + 'binnedData/riBinnedData.pickle'
binnedData = pickle.load(open(binnedDataFilename, "rb" ))
parameters = binnedData.parameterDictionary
print('Binned data loaded')
#print(parameters)

# Extract basic parameters
dt = parameters['dt']
D = parameters['D']
mass =  parameters['mass']
KbT = parameters['KbT']
boxsize = parameters['boxsize']
boundaryType = parameters['boundaryType']

# External potential parameters
kconstant = 0.3

# Integrator parameters
integratorStride = 50
tfinal = 10000
equilibrationSteps = 10000

# Create parameter dictionary to write to parameters reference file
parameterfilename = os.path.join(outputDataDirectory, "parameters")
parameterDictionary = {'numFiles' : numSimulations, 'dt' : dt, 'D' : D, 'KbT' : KbT,
                       'mass' : mass, 'tfinal' : tfinal, 'stride' : integratorStride,
                       'boxsize' : boxsize, 'boundaryType' : boundaryType,
                       'equilibrationSteps' : equilibrationSteps}
analysisTools.writeParameters(parameterfilename, parameterDictionary)

# Provides base filename (folder must exist (and preferably empty), otherwise H5 might fail)
basefilename = os.path.join(outputDataDirectory, "simMoriZwanzigReduced")

# Simulation wrapper for parallel runs
def runParallelSims(simnumber):

    # Define particle list
    seed = int(simnumber)
    random.seed(seed)
    position = [0, 0, 0]
    velocity = [0, 0, 0]
    particle = deepRD.particle(position, D, velocity, mass)
    particleList = deepRD.particleList([particle])

    # Define noise sampler
    nSampler = noiseSampler(binnedData)

    # Define external potential
    harmonicPotential = harmonic(kconstant)

    diffIntegrator = langevinNoiseSampler(dt, integratorStride, tfinal, nSampler, KbT,
                                          boxsize, boundaryType, equilibrationSteps)
    diffIntegrator.setExternalPotential(harmonicPotential)

    # Integrate dynamics
    t, X, V = diffIntegrator.propagate(particleList)

    # Write dynamics into trjactory
    traj = trajectoryTools.convert2trajectory(t, [X, V])
    trajectoryTools.writeTrajectory(traj,basefilename,simnumber)

    print("Simulation " + str(simnumber) + ", done.")


# Runs several simulations in parallel
print("Simulation begins ...")
num_cores = multiprocessing.cpu_count() - 1
pool = Pool(processes=num_cores)
iterator = [i for i in range(numSimulations)]
pool.map(partial(runParallelSims), iterator)