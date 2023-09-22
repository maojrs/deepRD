import os
import sys
import numpy as np
import random
import time
import deepRD
from deepRD.diffusionIntegrators import smoluchowski
from deepRD.tools import analysisTools
np.set_printoptions(threshold=sys.maxsize) # To print np arrays into file

import multiprocessing
from multiprocessing import Pool

localDataDirectory = os.environ['DATA'] + 'openSystems/'
numSimulations = 1000 #10000

# Define parameters
simID = '0036'
D = 0.5 #diffusion coefficient
dt = 0.001 #0.001 # timestep
stride = 1 # timesteps stride for output
tfinal = 100
kappa = 10.0 #100000 #10.0 # intrinsic Reaction rate (ala Colins and Kimball)
sigma = 1.0 # Reaction radius
R = 5.0 #10.0 # Far-field boundary
cR = 1.0 # Concentration of reservoir
equilibrationSteps = 0
tauleapSubsteps = 10
secondOrderPARB = False

# Output data directory
foldername = 'classicSmoluchowski_' + simID
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
parameterfilename = os.path.join(outputDataDirectory, "parameters_" + simID)
parameterDictionary = {'numSimulations' : numSimulations, 'D' : D, 'dt' : dt, 'stride' : stride, 'tfinal' : tfinal,
                       'kappa' : kappa, 'sigma' : sigma, 'R' : R, 'cR' : cR, 'equilibrationSteps' : equilibrationSteps,
                       'tauleapSubsteps' : tauleapSubsteps, 'secondOrderPARB' : secondOrderPARB}
analysisTools.writeParameters(parameterfilename, parameterDictionary)


# Create empty files to save the data in parallel algorithm
filename = outputDataDirectory  + '/classicSmol_nsims' + str(numSimulations) + '.xyz'

# Simulation wrapper for parallel runs
def runParallelSims(simnumber):

    seed = int( (time.time()) * random.random())
    np.random.seed(seed)

    # Define empty particle list
    particleList = deepRD.particleList([])

    # Define integrator
    diffIntegrator = smoluchowski(dt, stride, tfinal, D, kappa, sigma, R, cR, equilibrationSteps, tauleapSubsteps, secondOrderPARB)

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
    num_cores = 1 #multiprocessing.cpu_count() - 1
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
