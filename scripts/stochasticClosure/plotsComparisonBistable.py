import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from deepRD.noiseSampler import binnedData
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
matplotlib.rcParams.update({'font.size': 15})

bsize = 5

# Benchmark data folder
#parentDirectory = os.environ.get('MSMRD') + '/data/MoriZwanzig/bistable/benchmarkComparison/'
parentDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize) + '/benchmarkComparison/'
benchmarkfnamebase = parentDirectory + 'simMoriZwanzig_'
# Reduced models data folders
#localDataDirectory = '../../data/stochasticClosure/bistable/benchmarkReduced'
localDataDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize) + '/benchmarkReduced'
numModels = 8
redModelfnamebase = [localDataDirectory]*numModels
redModelfnamebase[0] += '_ri/simMoriZwanzigReduced_'
redModelfnamebase[1] += '_ririm/simMoriZwanzigReduced_'
redModelfnamebase[2] += '_qi/simMoriZwanzigReduced_'
redModelfnamebase[3] += '_qiri/simMoriZwanzigReduced_'
redModelfnamebase[4] += '_qiririm/simMoriZwanzigReduced_'
redModelfnamebase[5] += '_pi/simMoriZwanzigReduced_'
redModelfnamebase[6] += '_piri/simMoriZwanzigReduced_'
redModelfnamebase[7] += '_piririm/simMoriZwanzigReduced_'

# Create plot directory
plotDirectory = os.environ['DATA'] + 'stochasticClosure/bistable/boxsize' + str(bsize) + '/plots/'
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
            print("File ", i+1, " of ", numSimulations, " done.", end="\r")
        allTrajs[i] = iTraj
    except:
        continue
print("Reduced models data loaded.")


# ## Kernel density estimation from data


# Choose which reduced model to compare (just uncomment one)
conditionedOn = 'piri' #Possibilities 'qi', 'ri', 'qiri', 'qiririm'

if conditionedOn == 'ri':
    trajs = allTrajs[0]
    texlabel = r'$(r_{i+1}|r_i)$'
elif conditionedOn == 'ririm':
    trajs = allTrajs[1] 
    texlabel = r'$(r_{i+1}|r_i,r_{i-1})$'
elif conditionedOn == 'qi':
    trajs = allTrajs[2] 
    texlabel = r'$(r_{i+1}|x_i)$'
elif conditionedOn == 'qiri':
    trajs = allTrajs[3] 
    texlabel = r'$(r_{i+1}|x_i,r_i)$'
elif conditionedOn == 'qiririm':
    trajs = allTrajs[4] 
    texlabel = r'$(r_{i+1}|x_i,r_i,r_{i-1})$'
elif conditionedOn == 'pi':
    trajs = allTrajs[5] 
    texlabel = r'$(r_{i+1}|v_i)$'
elif conditionedOn == 'piri':
    trajs = allTrajs[6] 
    texlabel = r'$(r_{i+1}|v_i,r_i)$'
elif conditionedOn == 'piririm':
    trajs = allTrajs[7]
    texlabel = r'$(r_{i+1}|v_i,r_i,r_{i-1})$'


# Extract variables to plot from tajectories
position = trajectoryTools.extractVariableFromTrajectory(trajs, variableIndex = [1,4])
velocity = trajectoryTools.extractVariableFromTrajectory(trajs, variableIndex = [4,7])
position_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = [1,4])
velocity_ref = trajectoryTools.extractVariableFromTrajectory(trajs_ref, variableIndex = [4,7])


#  Obtain bandwidth for kernel density estimation through cross validation
numsamples = 50000
crossValidation = False
if crossValidation:
    # Sample random points from original data (both positions and velocities)
    idx = np.random.randint(len(position_ref), size=numsamples)
    sampled_positions = position_ref[idx,:]
    sampled_velocities = velocity_ref[idx,:]
    
    # Run cross validations for positions
    gridPos = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.1, 1.0, 31)},
                        cv=5,  # 5-fold cross-validation
                        verbose=2)
    gridPos.fit(sampled_positions)
    print(gridPos.best_params_)
    print(gridPos.best_estimator_)
    
    # Run cross validations for velocities
    gridVel = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.01, 0.2, 31)},
                        cv=5,  # 5-fold cross-validation
                        verbose=2)
    gridVel.fit(sampled_velocities) #(sampled_positions) 
    print(gridVel.best_params_)
    print(gridVel.best_estimator_)


# Estimate the densities using Gaussian kernel density estimation
bandwidthPos = 0.190 # Obtained through cross-validation 0.19(Main) but also 0.22 and 0.28 #0.77 #0.2
bandwidthVel = 0.036 # Obtained through cross-validation 0.036(Main) but also 0.1
# Use "epanechnikov" for fast tests, "gaussian" for final plots
kernelType = "gaussian" #"epanechnikov" # "epanechnikov", "tophat" "gaussian" # Sample requires Guassian/tophat
rtol=1E-3 # Default value zero, sacrifices minor accuracy for faster computation
kdePosition = KernelDensity(kernel=kernelType, bandwidth=bandwidthPos, rtol=rtol).fit(position)
kdePosition_ref = KernelDensity(kernel=kernelType, bandwidth=bandwidthPos, rtol=rtol).fit(position_ref)
kdeVelocity = KernelDensity(kernel=kernelType, bandwidth=bandwidthVel, rtol=rtol).fit(velocity)
kdeVelocity_ref = KernelDensity(kernel=kernelType, bandwidth=bandwidthVel, rtol=rtol).fit(velocity_ref)
def calculateKernelDensity(x, variable = 'position_ref'):
    if variable == 'position':
        log_dens = kdePosition.score_samples(x)
    elif variable == 'position_ref':
        log_dens = kdePosition_ref.score_samples(x)
    elif variable == 'velocity':
        log_dens = kdeVelocity.score_samples(x)
    elif variable == 'velocity_ref':
        log_dens = kdeVelocity_ref.score_samples(x)
    return np.exp(log_dens)

def sampleKernelDensity(numSamples, variable = 'position_ref'):
    ''' Only available for kernel density estimation using gaussian or tophat'''
    if variable == 'position':
        return kdePosition.sample(numSamples)
    elif variable == 'position_ref':
        return kdePosition_ref.sample(numSamples)
    elif variable == 'velocity':
        return kdeVelocity.sample(numSamples)
    elif variable == 'velocity_ref':
        return kdeVelocity_ref.sample(numSamples)
    
# Sample 3D values from estimated reference density. It can only sample 
# if Gaussian or tophat kernels are being used
#numsamples = 50000
#values = sampleKernelDensity(numsamples, variable)
#values_ref = sampleKernelDensity(numsamples, variable + '_ref')


# Calculate kernel density output for a certain one dimensional cut going through the origin.

# Obtain x, y, or z cut of the distribution
xxPos = np.arange(-2.5,2.5,0.1)
xxVel = np.arange(-0.6,0.6,0.015)
ww = np.zeros(len(xxPos))
ww2 = np.zeros(len(xxVel))
xyzcutPos = [None]*3
xyzcutVel = [None]*3
distributionPos = [None]*3
distributionPos_ref = [None]*3 
distributionVel = [None]*3
distributionVel_ref = [None]*3
xlabel = [r'$x$', r'$y$', r'$z$']
zerolabel = [r'$y=z=0$', r'$x=z=0$', r'$x=y=0$']

# Calculate distributions for xcut (y=z=0)
xyzcutPos[0] = np.array(list(zip(xxPos,ww,ww))).reshape(-1, 3)
xyzcutVel[0] = np.array(list(zip(xxVel,ww2,ww2))).reshape(-1, 3)
distributionPos[0] = calculateKernelDensity(xyzcutPos[0], 'position')
distributionPos_ref[0] = calculateKernelDensity(xyzcutPos[0], 'position_ref')
distributionVel[0] = calculateKernelDensity(xyzcutVel[0], 'velocity')
distributionVel_ref[0] = calculateKernelDensity(xyzcutVel[0], 'velocity_ref')
print("Calculations of x-cut distributions finished.")

# Calculate distributions for ycut (x=z=0)
xyzcutPos[1] = np.array(list(zip(ww,xxPos,ww))).reshape(-1, 3)
xyzcutVel[1] = np.array(list(zip(ww2,xxVel,ww2))).reshape(-1, 3)
distributionPos[1] = calculateKernelDensity(xyzcutPos[1], 'position')
distributionPos_ref[1] = calculateKernelDensity(xyzcutPos[1], 'position_ref')
distributionVel[1] = calculateKernelDensity(xyzcutVel[1], 'velocity')
distributionVel_ref[1] = calculateKernelDensity(xyzcutVel[1], 'velocity_ref')
print("Calculations of y-cut distributions finished.")

# Calculate distributions for zcut (x=y=0)
xyzcutPos[2] = np.array(list(zip(ww,ww,xxPos))).reshape(-1, 3)
xyzcutVel[2] = np.array(list(zip(ww2,ww2,xxVel))).reshape(-1, 3)
distributionPos[2] = calculateKernelDensity(xyzcutPos[2], 'position')
distributionPos_ref[2] = calculateKernelDensity(xyzcutPos[2], 'position_ref')
distributionVel[2] = calculateKernelDensity(xyzcutVel[2], 'velocity')
distributionVel_ref[2] = calculateKernelDensity(xyzcutVel[2], 'velocity_ref')
print("Calculations of z-cut distributions finished.")


# ## Distribution plots comparisons

# Plot distribution comparisons

# Create plot
fig = plt.figure(figsize=(15,8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
ax1, ax2 = gs.subplots() #sharey='row')

# Plot position distribution
for i in range(3):
    ax1[i].plot(xxPos,distributionPos_ref[i], '-k', lw = 0.5)
    ax1[i].fill_between(xxPos,distributionPos_ref[i], color='dodgerblue', alpha = 0.15, label = "benchmark")
    ax1[i].plot(xxPos,distributionPos[i], 'xk', label = 'reduced ' + texlabel)
    #ax1[i].set_xlim((-4,4))
    ax1[i].set_ylim((0,None))
    ax1[i].set_xlabel(xlabel[i] + '-position')
    if i==0:
        ax1[i].yaxis.set_ticks(np.arange(0,0.1,0.02)) 
    else:
        ax1[i].yaxis.set_ticks(np.arange(0,0.03,0.01)) 
    

    # Plot velocity distribution
    ax2[i].plot(xxVel,distributionVel_ref[i], '-k', lw = 0.5)
    ax2[i].fill_between(xxVel,distributionVel_ref[i], color='dodgerblue', alpha = 0.15, label = "benchmark")
    ax2[i].plot(xxVel,distributionVel[i], 'xk', label = 'reduced ' + texlabel)
    #ax2[i].set_xlim((-0.6,0.6))
    ax2[i].set_ylim((0,None))
    ax2[i].set_xlabel(xlabel[i] + '-velocity' + '\n('+ zerolabel[i] +')')
    ax2[i].xaxis.set_ticks(np.arange(-0.5,0.6,0.5)) 
    #ax2[i].yaxis.set_ticks(np.arange(0,1.5,0.5)) 

ax1[2].legend(bbox_to_anchor=(0.6, 0., 0.5, 1.0), framealpha=1.0)
    
# displaying plot
plt.tight_layout()
plt.savefig(plotDirectory + 'distributions_comparison_bistable_'+ conditionedOn +'.pdf')
plt.clf()

# Plot x-position distribution vs samples from original data

# Create plot
fig, (ax1, ax2) = plt.subplots(figsize=(8,6), nrows=2, sharex=True)

# Plot x-position distribution
ax1.plot(xxPos,distributionPos_ref[0], '-k', lw = 0.5)
ax1.fill_between(xxPos,distributionPos_ref[0], color='dodgerblue', alpha = 0.15, label = "benchmark (kde)")
#ax1.plot(xx,distributionPos[0], 'xk', label = 'reduced ' + texlabel)
ax1.set_ylim((0,0.11)) #None))
ax1.yaxis.set_ticks(np.arange(0,0.15,0.05)) 
ax1.legend(bbox_to_anchor=(0.5, 0., 0.5, 1.02)) #, framealpha=1.0, borderpad=0.2)

# Plot velocity distribution
numsamples = 50000
idx = np.random.randint(len(position_ref), size=numsamples)
sampledPosRef = position_ref[idx,:]
sampledVelRef = velocity_ref[idx,:]

ax2.scatter(sampledPosRef[:,0],sampledPosRef[:,1], marker='o', s=0.1,label='random data samples')

ax2.set_xlabel('position ' + xlabel[0])
ax2.set_ylabel(r'$y$')
ax2.set_xlim([-2.5,2.5])
ax2.set_ylim([-1.5,1.5])
ax2.xaxis.set_ticks(np.arange(-2,3,1))
ax2.yaxis.set_ticks(np.arange(-1,2,1)) 
ax2.legend(loc="upper right", markerscale=20, borderpad=0.1)

plt.subplots_adjust(hspace=0)
#plt.savefig('kernel_density_estimation.pdf')
plt.savefig(plotDirectory + 'distributions_n_sampleddata_bistable_'+ conditionedOn +'.pdf')
plt.clf()


# ## Plot auto-correlation functions comparison


# Uses only a subset (mtrajs) of the total trajectories, since computing them with all is very slow
variables = ['position', 'velocity']
lagtimesteps = [40,40]
mtrajs = 1
strides = [200,5] #[30,1]
ACF = [None]*2
ACF_ref = [None]*2
for i, var in enumerate(variables):
    #mean = trajectoryTools.calculateMean(trajs[0:mtrajs], var)
    #mean_ref = trajectoryTools.calculateMean(trajs_ref[0:mtrajs], var)
    #variance = trajectoryTools.calculateVariance(trajs[0:mtrajs], var, mean)
    #variance_ref = trajectoryTools.calculateVariance(trajs_ref[0:mtrajs], var, mean_ref)
    ACF[i] = trajectoryTools.calculateAutoCorrelationFunction(trajs[0:mtrajs], lagtimesteps[i], strides[i], var)
    ACF_ref[i] = trajectoryTools.calculateAutoCorrelationFunction(trajs_ref[0:mtrajs], lagtimesteps[i], strides[i], var)


index = 0
time = dt*integratorStride*strides[index]*np.linspace(1,lagtimesteps[index],lagtimesteps[index])
plt.plot(time/1000, ACF[index], 'xk', label = 'reduced ' + texlabel)
plt.plot(time/1000, ACF_ref[index], '-k', label = 'benchmark')
plt.xlabel('time ' +r'$(\mu s)$')
plt.ylabel(variables[index] + ' autocorrelation')
plt.legend()
#plt.xlim([0,1500])
plt.subplots_adjust(bottom=0.15)
plt.savefig(plotDirectory + variables[index] + '_autocorrelation_bistable_'+ conditionedOn +'.pdf', bbox_inches='tight')
plt.clf()

index = 1
time = dt*integratorStride*strides[index]*np.linspace(1,lagtimesteps[index],lagtimesteps[index])
plt.plot(time, ACF[index], 'xk', label = 'reduced ' + texlabel)
plt.plot(time, ACF_ref[index], '-k', label = 'benchmark')
plt.xlabel('time ' + r'$(ns)$')
plt.ylabel(variables[index] + ' autocorrelation')
plt.legend()
#plt.xlim([0,1500])
plt.subplots_adjust(bottom=0.15)
plt.savefig(plotDirectory + variables[index]+ '_autocorrelation_bistable_'+ conditionedOn +'.pdf', bbox_inches='tight')
plt.clf()
