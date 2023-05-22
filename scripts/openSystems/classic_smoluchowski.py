import os
import numpy as np
import random
import time
import deepRD
from deepRD.diffusionIntegrators import smoluchowski
from deepRD.tools import analysisTools

import multiprocessing
from multiprocessing import Pool

localDataDirectory = os.environ['DATA'] + 'openSystems/'
numSimulations = 10

# Define parameters
simID = 0
D = 0.1 #diffusion coefficient
stride = 1 # timesteps stride for output
tfinal = 10
kappa = 10.0 # intrinsic Reaction rate (ala Colins and Kimball)
sigma = 0.0 # Reaction radius
R = 1.0 # Far-field boundary
deltar = 0.05 # Width of boundary layer next to reservoir
cR = 10.0 # Concentration of reservoir
dt = deltar*deltar/(2.0*D) # Largest possible timestep
equilibrationSteps = 0

# Output data directory
foldername = 'classic_smoluchowski_' + str(simID)
outputDataDirectory = os.path.join(localDataDirectory, foldername)
# Create folder for data
try:
    os.makedirs(outputDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Create parameter dictionary to write to parameters reference file
parameterfilename = os.path.join(outputDataDirectory, "parameters_" + str(simID))
parameterDictionary = {'numSimulations' : numSimulations, 'D' : D, 'dt' : dt, 'stride' : stride, 'tfinal' : tfinal,
                       'kappa' : kappa, 'sigma' : sigma, 'R' : R, 'deltar' : deltar,
                       'cR' : cR, 'equilibrationSteps' : equilibrationSteps}
analysisTools.writeParameters(parameterfilename, parameterDictionary)


# Create empty files to save the data in parallel algorithm
filename = outputDataDirectory  + '/classic_smol_nsims' + str(numSimulations) + '.xyz'

# Simulation wrapper for parallel runs
def runParallelSims(simnumber):

    seed = int( (time.time()) * random.random())
    np.random.seed(seed)

    # Define empty particle list
    particleList = deepRD.particleList([])

    # Define integrator
    diffIntegrator = smoluchowski(dt, stride, tfinal, D, kappa, sigma, R, deltar, cR, equilibrationSteps)

    # Propagate simulation
    t, positionsArrays = diffIntegrator.propagate(particleList)

    # Transform position array of last time step to radial distances array
    distancesArray = np.linalg.norm(positionsArrays[-1], axis=1)

    return particleList.countParticles(), distancesArray



def multiprocessingHandler():
    '''
    Handles parallel processing of classic_Smoluchowski and writes to same file in parallel
    '''
    # Runs several simulations in parallel
    num_cores = multiprocessing.cpu_count() - 1
    pool = Pool(processes=num_cores)
    trajNumList = list(range(numSimulations))
    with open(filename, 'w') as file:
        for index, result in enumerate(pool.imap(runParallelSims, trajNumList)):
            numParticles, distArray = result
            file.write(np.array2string(distArray) + '\n')
            print("Simulation " + str(index) + ", done. Final number particles: " + str(numParticles))


## Run parallel code
multiprocessingHandler()

## Serial test
#for i in range(numSimulations):
#    runParallelSims(i)
#    print('')