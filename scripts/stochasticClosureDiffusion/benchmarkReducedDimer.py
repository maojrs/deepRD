import numpy as np
import random
import os
import sys
import deepRD
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal
from deepRD.potentials import pairBistableBias
from AdaLN_Gating_Multihead_dimer import diffusionSampler
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools

import multiprocessing
from multiprocessing import Pool
from functools import partial
import torch

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


CONDITION_TO_MODEL_PATH = {
    'pidqiri': 'noiseSampler/Diffusion_checkpoints/AdaLN_pidqiri_dimer_T40.pt',
    'piabsdqiri': 'noiseSampler/Diffusion_checkpoints/AdaLN_piabsdqiri_dimer_T40.pt',
    'piabsdqiririm': 'noiseSampler/Diffusion_checkpoints/AdaLN_piabsdqiririm_dimer_T40.pt',
    'pipimabsdqiririm': 'noiseSampler/Diffusion_checkpoints/AdaLN_pipimabsdqiririm_dimer_T40.pt',
    'piabsdqiririmrimm': 'noiseSampler/Diffusion_checkpoints/AdaLN_piabsdqiririmrimm_dimer_T40.pt',
}


# Simulation parameters
if 'DATA' not in os.environ:
    raise ValueError('Missing DATA environment variable.')

numSimulations = 100
bsize = 5
# Available conditionings: dqi, dpi, vi, ri, dqiri, dpiri, dqiririm, dpiririm, etc...
conditionedOn = 'piabsdqiririm'
outputAux = True

localDataDirectory = os.environ['DATA'] + 'stochClosureDiffusion/'
parentDirectory = os.environ['DATA'] + 'stochasticClosure/Dimer/boxsize' + str(bsize) + '/benchmark/'
parameters = analysisTools.readParameters(os.path.join(parentDirectory, 'parameters'))

normPath = os.path.abspath('normal_file.npz')


# Output data directory
foldername = 'Dimer/boxsize' + str(bsize) + '/benchmarkReduced_' + conditionedOn
outputDataDirectory = os.path.join(localDataDirectory, foldername)
# Create folder for data
if os.path.isdir(outputDataDirectory):
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input().strip().lower()
    if proceed != 'y':
        sys.exit()
else:
    os.makedirs(outputDataDirectory)


# Extract basic parameters
dt = parameters['dt']
Gamma = parameters['Gamma']
mass = parameters['mass']
KbT = parameters['KbT']
boxsize = parameters['boxsize']
boundaryType = parameters['boundaryType']

if bsize != boxsize:
    print('Requested boxsize does not match simulation')


# Parameters for pair potential that will only act on distinguished particles (type 1)
particleDiameter = 0.5
x0 = 1.0 * particleDiameter  # location of first minimum
rad = 1.0 * particleDiameter  # half the distance between minima
scalefactor = 2

# Integrator parameters
integratorStride = 1
tfinal = 1000
equilibrationSteps = 5000

model_path = CONDITION_TO_MODEL_PATH.get(conditionedOn)
if model_path is None:
    raise ValueError(f'Unknown condition: {conditionedOn}')

# Create parameter dictionary to write to parameters reference file
parameterfilename = os.path.join(outputDataDirectory, 'parameters')
parameterDictionary = {
    'numFiles': numSimulations,
    'dt': dt,
    'Gamma': Gamma,
    'KbT': KbT,
    'mass': mass,
    'tfinal': tfinal,
    'stride': integratorStride,
    'boxsize': boxsize,
    'boundaryType': boundaryType,
    'equilibrationSteps': equilibrationSteps,
    'conditionedOn': conditionedOn,
    'model_path': model_path,
}
analysisTools.writeParameters(parameterfilename, parameterDictionary)

# Provides base filename (folder must exist (and preferably empty), otherwise H5 might fail)
basefilename = os.path.join(outputDataDirectory, 'simMoriZwanzigReduced_')


# Simulation wrapper for parallel runs
def runParallelSims(simnumber, norm_path=normPath):
    # Load normalization arrays in each worker to avoid shared NpzFile handles.
    with np.load(norm_path) as f:
        norm_params = {k: f[k].copy() for k in f.files}

    # Define particle list
    seed = int(simnumber)
    random.seed(seed)
    torch.manual_seed(seed)
    position1 = [0.0, 0.0, 0.0]
    position2 = [x0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    particle1 = deepRD.particle(position1, velocity=velocity, mass=mass)
    particle2 = deepRD.particle(position2, velocity=velocity, mass=mass)
    particleList = deepRD.particleList([particle1, particle2])

    # Define pair potential
    pairBistablePotential = pairBistableBias(x0, rad, scalefactor)

    # Define NN-based noise sampler
    nSampler = diffusionSampler(model_path, conditionedOn, device='cuda', normalize=True, norm_params=norm_params)

    diffIntegrator = langevinNoiseSamplerDimerGlobal(
        dt,
        integratorStride,
        tfinal,
        Gamma,
        nSampler,
        KbT,
        boxsize,
        boundaryType,
        equilibrationSteps,
        conditionedOn,
    )
    diffIntegrator.setPairPotential(pairBistablePotential)

    # Integrate dynamics
    t, X, V, Raux = diffIntegrator.propagate(particleList, outputAux=outputAux)

    # Write dynamics into trajectory
    traj = trajectoryTools.convert2trajectory(t, [X, V, Raux])
    trajectoryTools.writeTrajectory(traj, basefilename, simnumber)

    print('Simulation ' + str(simnumber) + ', done.')


## Runs several simulations in parallel
print('Simulation for ri+1|' + conditionedOn + ' begins ...')
num_cores = max(multiprocessing.cpu_count() - 1, 1)
pool = Pool(processes=num_cores)
iterator = [i for i in range(numSimulations)]

for _ in pool.imap_unordered(partial(runParallelSims, norm_path=normPath), iterator, chunksize=1):
    pass

pool.close()
pool.join()
