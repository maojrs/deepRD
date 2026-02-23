import numpy as np
import random
import os
import sys
import pickle
import deepRD
import torch
import joblib
#from deepRD.diffusionIntegrators import langevinNoiseSampler, langevinNoiseSamplerDimer, \
#from deepRD.diffusionIntegrators import langevinNoiseSamplerDimer2, langevinNoiseSamplerDimer3
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal #langevinNoiseSamplerDimerConstrained1DGlobal #, langevinNoiseSamplerDimerConstrained1D
#from deepRD.potentials import pairBistable
from deepRD.potentials import pairBistableBias
from deepRD.noiseSampler import cvaeSampler, defaultSamplingModel
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools

import multiprocessing
from multiprocessing import Pool
from functools import partial
import hashlib

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
numSimulations = 100 #10 #100
bsize = 5 #5 #8 #10
# Available conditionings: dqi, dpi, vi,  ri, dqiri, dpiri, dqiririm, dpiririm, etc...
# dqi:=relative distance between dimer particles, dpi:= relative velocity along dimer axis,
# vi:=center of mass velocity (first component norm along axis, second one norm of perpendicular part)
conditionedOn = 'pipimdqidpiririm' #'pi'
outputAux = True #False
nbins = 5

# Output data directory
foldername = 'dimerGlobal/boxsize' + str(bsize) + '/benchmarkReducedGen_' + conditionedOn
outputDataDirectory = os.path.join(localDataDirectory, foldername)
# Create folder for data
try:
    os.makedirs(outputDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Define noise sampler
localModelDirectory = 'notebooks/stochasticClosureCVAE/dev/'
systemType='dimer'

# Model weights and scaler filepath
model_state_path = localModelDirectory + f"ckpts/cvae_checkpoint_{systemType}_{conditionedOn}_stride1.pt"
normalizers_path = localModelDirectory + f"normalizers/normalizers_{systemType}_{conditionedOn}_stride1.pkl"

# Loading parameter dictionary
parentDirectory = os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize)+ '/benchmark/'
parameters = analysisTools.readParameters(parentDirectory + "parameters")


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
#numbins = parameters['numbins']
#lagTimesteps = parameters['lagTimesteps']
#nsigma = parameters['nsigma']
norm_params = (0,1,0,1)

# Define noise sampler, n latent dims

# For testing
#defaultSampler = defaultSamplingModel(mean=0, covariance=0.005)
#nSampler = noiseSampler(defaultSampler)

# Parameters for pair potential that will only actsgit on distinguished particles (type 1)
particleDiameter = 0.5
x0 = 1.0*particleDiameter # location of first minima
rad = 1.0*particleDiameter # half the distance between minimas
scalefactor = 2

# Integrator parameters
integratorStride = 1 #50
tfinal = 10000
equilibrationSteps = 10000
calculateRelPosVel = True

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

    # Loading Sampling Model
    zdim = 3
    nSampler = cvaeSampler.CVAE(zdim=zdim, system_type=systemType, cond_type=conditionedOn)
    nSampler.eval()
    ckpt = torch.load(model_state_path, map_location="cpu", weights_only=True)
    nSampler.load_state_dict(ckpt['model_state'])
    scalers = joblib.load(normalizers_path)
    nSampler.attach_normalizers(**scalers)
    nSampler.set_temps(Tr=1, Tz=1.5)

    # Define particle list
    seed = int(simnumber)
    random.seed(seed)
    position1 = [0.0, 0.0, 0.0]
    position2 = [x0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    particle1 = deepRD.particle(position1, velocity = velocity, mass=mass)
    particle2 = deepRD.particle(position2, velocity = velocity, mass=mass)
    particleList = deepRD.particleList([particle1, particle2])

    # Define pair potential
    #pairBistablePotential = pairBistable(x0, rad, scalefactor)
    pairBistablePotential = pairBistableBias(x0, rad, scalefactor)

    #diffIntegrator = langevinNoiseSampler(dt, integratorStride, tfinal, Gamma, nSampler, KbT, boxsize,
    #                                      boundaryType, equilibrationSteps, conditionedOn)
    #diffIntegrator = langevinNoiseSamplerDimer(dt, integratorStride, tfinal, Gamma, nSampler, KbT, boxsize,
    #                                      boundaryType, equilibrationSteps, conditionedOn)
    #diffIntegrator = langevinNoiseSamplerDimer2(dt, integratorStride, tfinal, Gamma, nSampler, KbT, boxsize,
    #                                      boundaryType, equilibrationSteps, conditionedOn)
    #diffIntegrator = langevinNoiseSamplerDimer3(dt, integratorStride, tfinal, Gamma, nSampler, KbT, boxsize,
    #                                            boundaryType, equilibrationSteps, conditionedOn)
    diffIntegrator = langevinNoiseSamplerDimerGlobal(dt, integratorStride, tfinal, Gamma, nSampler, KbT, boxsize,
                                          boundaryType, equilibrationSteps, conditionedOn)

    diffIntegrator.setPairPotential(pairBistablePotential)

    # Integrate dynamics
    #t, X, V = diffIntegrator.propagate(particleList, outputAux = outputAux)
    t, X, V, Raux = diffIntegrator.propagate(particleList, outputAux = outputAux)


    # Write dynamics into trjactory
    #traj = trajectoryTools.convert2trajectory(t, [X, V])
    traj = trajectoryTools.convert2trajectory(t, [X, V, Raux])
    trajectoryTools.writeTrajectory(traj,basefilename,simnumber)

    print("Simulation " + str(simnumber) + ", done.")


## Runs several simulations in parallel
print('Simulation for ri+1|' + conditionedOn + ' begins ...')
num_cores =  multiprocessing.cpu_count() - 1
pool = Pool(processes=num_cores)
iterator = [i for i in range(numSimulations)]
pool.map(partial(runParallelSims), iterator)

## Serial test
#for i in range(numSimulations):
#    runParallelSims(i)
