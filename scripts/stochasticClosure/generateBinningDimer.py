import os
import sys
import pickle
import numpy as np
from itertools import product
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler import binnedDataDimer

'''
Generates binned data structures on several different conditionings for the stochastic closure model.
Currently implemented on conditioning ri+1 on all the combinations qi,pi,ri,ri-1
'''

bsize = 5

parentDirectory = os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize)+ '/benchmark/'
fnamebase = parentDirectory + 'simMoriZwanzig_'
foldername = 'binnedData/'
binningDataDirectory = os.path.join(os.environ['DATA'] + 'stochasticClosure/dimer/boxsize' + str(bsize) + '/', foldername)


try:
    os.makedirs(binningDataDirectory)
except OSError as error:
    print('Folder ' + foldername + ' already exists. Previous data files might be overwritten. Continue, y/n?')
    proceed = input()
    if proceed != 'y':
        sys.exit()

# Load parameters from parameters file
parameterDictionary = analysisTools.readParameters(parentDirectory + "parameters")
# Parameters for loading continuous trajectories from files (from original simulation)
nfiles = parameterDictionary['numFiles']
dt = parameterDictionary['dt']
stride = parameterDictionary['stride']
totalTimeSteps = parameterDictionary['timesteps']
boxsize = parameterDictionary['boxsize']
boundaryType = parameterDictionary['boundaryType']

if bsize != boxsize:
    print('Requested boxsize does not match simulation')


def calculateAdditionalConditionings(x1,x2,v1,v2):
    deltaX = x2 - x1
    deltaV = v2 - v1
    velCM = 0.5*(v1 + v2)
    normDeltaX = np.linalg.norm(deltaX)
    unitDeltaX = deltaX/normDeltaX
    axisRelVel = np.dot(deltaV, unitDeltaX)
    axisVelCM = np.dot(velCM, unitDeltaX)
    normAxisVelCM = np.linalg.norm(axisVelCM)
    vecAxisVelCM = axisVelCM * unitDeltaX
    orthogonalVelCM = velCM - vecAxisVelCM
    normOrthogonalVelCM = np.linalg.norm(orthogonalVelCM)
    return normDeltaX, axisRelVel, normAxisVelCM, normOrthogonalVelCM


# Load trajectory data from h5 files (only of distinguished particle)
trajs = []
print("Loading data ...")
for i in range(nfiles):
    traj = trajectoryTools.loadTrajectory(fnamebase, i)
    lentraj = np.shape([traj])[1]
    # Compute and add relativeDistance and relative velocity at end of trajectory
    additionalCondtionings = np.zeros([lentraj,4])
    for j in range(int(lentraj/2)):
        x1 = traj[2*j][1:4]
        x2 = traj[2*j+1][1:4]
        v1 = traj[2*j][4:7]
        v2 = traj[2*j+1][4:7]
        normDeltaX, axisRelVel, normAxisVelCM, normTangVelCM = calculateAdditionalConditionings(x1,x2,v1,v2)
        additionalCondtionings[2*j][0] = normDeltaX
        additionalCondtionings[2*j][1] = axisRelVel
        additionalCondtionings[2*j][2] = normAxisVelCM
        additionalCondtionings[2*j][3] = normTangVelCM
        additionalCondtionings[2*j+1][0] = 1 * normDeltaX
        additionalCondtionings[2*j+1][1] = 1 * axisRelVel
        additionalCondtionings[2*j+1][2] = -1 * normAxisVelCM
        additionalCondtionings[2*j+1][3] = 1 * normTangVelCM
    newtraj = np.concatenate([traj,additionalCondtionings], axis=1)
    trajs.append(newtraj)
    sys.stdout.write("File " + str(i+1) + " of " + str(nfiles) + " done." + "\r")
print("\nAll data loaded.")
print(' ')

# Parameters used for binnings:
lagTimesteps = 1  # Number of timesteps (from data) to look back in time
boxsizeBinning = boxsize # Overriden by default when loading trajectory data
numbins1 = 30 #50
numbins2 = 10 #50
numbins3 = 5 #50
nsigma1 = 3 # Only include up to nsigma standard deviations around mean of data. If no value given, includes all.
nsigma2 = 2
nsigma3 = 2

# Add elements to parameter dictionary
parameterDictionary['lagTimesteps'] = lagTimesteps

# List of possible combinations for binnings
#binPositionList = [False] #[False, True]
binRelativeDistanceList = [False,True]
binRelSpeedList = [False, True]
binVelCenterMassList = [False, True]
numBinnedAuxVarsList = [2] #[0,1] #[0,1,2]

def getNumberConditionedVariables(binRelDist, binRelSpeed, binVelCenterMass, numBinnedAuxVars):
    numConditionedVariables = 0
    if binRelDist:
        numConditionedVariables += 1 # One dimensional, so we assume it doesn't count
    if binRelSpeed:
        numConditionedVariables += 1
    if binVelCenterMass:
        numConditionedVariables += 2
    for i in range(numBinnedAuxVars):
        numConditionedVariables += 3
    return numConditionedVariables

for parameterCombination in product(*[binRelativeDistanceList, binRelSpeedList, binVelCenterMassList, numBinnedAuxVarsList]):
    if parameterCombination != (False,False,0):
        binRelDist, binRelSpeed, binVelCenterMass, numBinnedAuxVars = parameterCombination
        numConditionedVariables = getNumberConditionedVariables(binRelDist, binRelSpeed, binVelCenterMass, numBinnedAuxVars)
        if numConditionedVariables <= 4:
            dataOnBins = binnedDataDimer(boxsizeBinning, numbins1, lagTimesteps, binRelDistance = binRelDist,
                                    binRelSpeed = binRelSpeed, binVelCenterMass = binVelCenterMass,
                                         numBinnedAuxVars=numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma1)
            parameterDictionary['numbins'] = numbins1
            parameterDictionary['nsigma'] = nsigma1
        elif numConditionedVariables <= 7:
            dataOnBins = binnedDataDimer(boxsizeBinning, numbins2, lagTimesteps, binRelDistance = binRelDist,
                                         binRelSpeed = binRelSpeed, binVelCenterMass = binVelCenterMass,
                                         numBinnedAuxVars=numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma2)
            parameterDictionary['numbins'] = numbins2
            parameterDictionary['nsigma'] = nsigma2
        else:
            dataOnBins = binnedDataDimer(boxsizeBinning, numbins3, lagTimesteps, binRelDistance = binRelDist,
                                    binRelSpeed = binRelSpeed, binVelCenterMass = binVelCenterMass,
                                         numBinnedAuxVars=numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma3)
            parameterDictionary['numbins'] = numbins3
            parameterDictionary['nsigma'] = nsigma3
        parameterDictionary['percentageOccupiedBins'] = dataOnBins.percentageOccupiedBins
        dataOnBins.parameterDictionary = parameterDictionary

        # Dump qi binned data into pickle file and free memory
        print('Dumping data into pickle file ...')
        conditionedVarsString = dataOnBins.binningLabel2
        binnedDataFilename = binningDataDirectory + conditionedVarsString + 'BinnedData.pickle'
        pickle.dump(dataOnBins, open(binnedDataFilename, "wb"))
        print('Binning for ' + dataOnBins.binningLabel + ' done.')
        print(' ')
        del dataOnBins