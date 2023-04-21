import os
import sys
import pickle
import numpy as np
from itertools import product
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler import  binnedDataDimerConstrained1DGlobal



'''
Generates binned data structures on several different conditionings for the stochastic closure model.
Currently implemented on conditioning ri+1 on all the combinations dqi,dpi,vi,ri,ri-1
'''

bsize = 8

parentDirectory = os.environ['DATA'] + 'stochasticClosure/dimer1D/boxsize' + str(bsize)+ '/benchmark/'
fnamebase = parentDirectory + 'simMoriZwanzig_'
foldername = 'binnedData/'
binningDataDirectory = os.path.join(os.environ['DATA'] + 'stochasticClosure/dimer1DGlobal/boxsize' + str(bsize) + '/', foldername)


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


def calculateRelDistance(x1,x2):
     deltaX = trajectoryTools.relativePosition(x1,x2,boundaryType, boxsize)
     normDeltaX = np.linalg.norm(deltaX)
     return normDeltaX


def calculateRelVelocity(v1, v2):
    relVelocity = v2 - v1
    return np.linalg.norm(relVelocity)

def calculateCMvelocity(v1, v2):
    CMvelocity = (v1 + v2)/2.0
    return np.linalg.norm(CMvelocity)



# def calculateAdditionalConditionings(x1,x2,v1,v2):
#     deltaX = trajectoryTools.relativePosition(x1,x2,boundaryType, boxsize)
#     deltaV = v2 - v1
#     velCM = 0.5*(v1 + v2)
#     normDeltaX = np.linalg.norm(deltaX)
#     unitDeltaX = deltaX/normDeltaX
#     axisRelVel = np.dot(deltaV, unitDeltaX)
#     axisVelCM = np.dot(velCM, unitDeltaX)
#     normAxisVelCM = np.linalg.norm(axisVelCM)
#     vecAxisVelCM = axisVelCM * unitDeltaX
#     orthogonalVelCM = velCM - vecAxisVelCM
#     normOrthogonalVelCM = np.linalg.norm(orthogonalVelCM)
#     return normDeltaX, axisRelVel, normAxisVelCM, normOrthogonalVelCM

# def calculateAdditionalConditionings(x1,x2,v1,v2):
#     deltaX = trajectoryTools.relativePosition(x1,x2,boundaryType, boxsize)
#     normDeltaX = np.linalg.norm(deltaX)
#     unitDeltaX = deltaX/normDeltaX
#     axisVel1 = np.dot(v1, unitDeltaX)
#     axisVel2 = np.dot(v2, -1.0*unitDeltaX)
#     orthogonalVel1 = v1 - axisVel1
#     orthogonalVel2 = v2 - axisVel2
#     normOrthogonalVel1 = np.linalg.norm(orthogonalVel1)
#     normOrthogonalVel2 = np.linalg.norm(orthogonalVel2)
#     return axisVel1, axisVel2, normOrthogonalVel1, normOrthogonalVel2

def calculateAdditionalConditionings(x1,x2,v1,v2):
    deltaX = trajectoryTools.relativePosition(x1,x2,boundaryType, boxsize)
    normDeltaX = np.linalg.norm(deltaX)
    unitDeltaX = deltaX/normDeltaX
    rotatedVel1 = trajectoryTools.rotateVec(unitDeltaX,v1)
    rotatedVel2 = trajectoryTools.rotateVec(unitDeltaX,v2)
    return deltaX, -1*deltaX, rotatedVel1, rotatedVel2


# Load trajectory data from h5 files (only of distinguished particle)
trajs = []
print("Loading data ...")
for i in range(nfiles):
    traj = trajectoryTools.loadTrajectory(fnamebase, i)
    # Compute and add relativeDistance and rotated velocity at end of trajectory
    lentraj = np.shape([traj])[1]
    additionalCondtionings = np.zeros([lentraj, 3])
    for j in range(int(lentraj / 2)):
        x1 = traj[2 * j][1:4]
        x2 = traj[2 * j + 1][1:4]
        v1 = traj[2 * j][4:7]
        v2 = traj[2 * j + 1][4:7]
        deltaX = calculateRelDistance(x1, x2)
        deltaV = calculateRelVelocity(v1,v2)
        CMvel = calculateCMvelocity(v1,v2)
        additionalCondtionings[2 * j][0:1] = deltaX
        additionalCondtionings[2 * j][1:2] = deltaV
        additionalCondtionings[2 * j][2:3] = CMvel
        additionalCondtionings[2 * j][3:4] = 0.0
        additionalCondtionings[2 * j + 1][0:1] = -1 * deltaX
        additionalCondtionings[2 * j + 1][1:2] = -1*deltaV
        additionalCondtionings[2 * j + 1][2:3] = CMvel
        additionalCondtionings[2 * j + 1][3:4] = 0.0
    newtraj = np.concatenate([traj, additionalCondtionings], axis=1)
    trajs.append(newtraj)
    sys.stdout.write("File " + str(i+1) + " of " + str(nfiles) + " done." + "\r")
print("\nAll data loaded.")
print(' ')

# Parameters used for binnings:
lagTimesteps = 1  # Number of timesteps (from data) to look back in time
boxsizeBinning = boxsize # Overriden by default when loading trajectory data
numbins1 = 20 #50
numbins2 = 20 #10 #50
numbins3 = 20 #50
nsigma1 = 3 # Only include up to nsigma standard deviations around mean of data. If no value given, includes all.
nsigma2 = 3 #2
nsigma3 = 3

# Add elements to parameter dictionary
parameterDictionary['lagTimesteps'] = lagTimesteps

# List of possible combinations for binnings
binPositionList = [False] #[False, True]
#binVelocitiesList = [True]
numBinnedVelVarsList = [1,2]
binRelativeDistanceList = [False]
binRelativeSpeedList = [False]
binCMvelocityList = [False]
numBinnedAuxVarsList = [0,1,2] #[0,1] #[0,1,2]

def getNumberConditionedVariables(binPosition, numBinnedVelVars, binRelDistance,
                                  binRelSpeed, binCMvelocity, numBinnedAuxVars):
    numConditionedVariables = 0
    if binPosition:
        numConditionedVariables += 2 #1
    for i in range(numBinnedVelVars):
        numConditionedVariables += 2 #1
    if binRelDistance:
        numConditionedVariables += 1
    if binRelSpeed:
        numConditionedVariables += 1
    if binCMvelocity:
        numConditionedVariables += 2 #1
    for i in range(numBinnedAuxVars):
        numConditionedVariables += 2 #1
    return numConditionedVariables


for parameterCombination in product(*[binPositionList, numBinnedVelVarsList, binRelativeDistanceList,
                                      binRelativeSpeedList, binCMvelocityList, numBinnedAuxVarsList]):
    if parameterCombination != (False, 0, False, False, False, 0):
        binPosition, numBinnedVelVars, binRelDistance, binRelSpeed, binCMvelocity, numBinnedAuxVars = parameterCombination
        numConditionedVariables = getNumberConditionedVariables(binPosition, numBinnedVelVars, binRelDistance,
                                                                binRelSpeed, binCMvelocity, numBinnedAuxVars)
        if numConditionedVariables <= 2: #3:
            #if numConditionedVariables == 1:
            #dataOnBins = binnedData(boxsizeBinning, numbins1, lagTimesteps, binPosition, binVelocity,
            #                        numBinnedAuxVars)
            dataOnBins = binnedDataDimerGlobal(boxsizeBinning, numbins1, lagTimesteps, binPosition, numBinnedVelVars,
                                    binRelDistance, binRelSpeed, binCMvelocity, numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma1)
            parameterDictionary['numbins'] = numbins1
            parameterDictionary['nsigma'] = nsigma1
        elif numConditionedVariables <= 4: #6:
            #elif numConditionedVariables == 2:
            #dataOnBins = binnedData(boxsizeBinning, numbins2, lagTimesteps, binPosition, binVelocity,
            #                        numBinnedAuxVars)
            dataOnBins = binnedDataDimerGlobal(boxsizeBinning, numbins2, lagTimesteps, binPosition,
                                                            numBinnedVelVars, binRelDistance,
                                                            binRelSpeed, binCMvelocity, numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma2)
            parameterDictionary['numbins'] = numbins2
            parameterDictionary['nsigma'] = nsigma2
        else:
            #dataOnBins = binnedData(boxsizeBinning, numbins3, lagTimesteps, binPosition, binVelocity,
            #                        numBinnedAuxVars)
            dataOnBins = binnedDataDimerGlobal(boxsizeBinning, numbins3, lagTimesteps, binPosition,
                                                            numBinnedVelVars, binRelDistance,
                                                            binRelSpeed, binCMvelocity, numBinnedAuxVars)
            dataOnBins.loadData(trajs, nsigma3)
            parameterDictionary['numbins'] = numbins3
            parameterDictionary['nsigma'] = nsigma3
        parameterDictionary['percentageOccupiedBins'] = dataOnBins.percentageOccupiedBins
        dataOnBins.parameterDictionary = parameterDictionary

        # Dump qi binned data into pickle file and free memory
        print('Dumping data into pickle file ...')
        conditionedVarsString = dataOnBins.binningLabel2
        binnedDataFilename = binningDataDirectory + conditionedVarsString + 'BinnedData_' + str(parameterDictionary['numbins']) + 'bins.pickle'
        pickle.dump(dataOnBins, open(binnedDataFilename, "wb"))
        print('Binning for ' + dataOnBins.binningLabel + ' done.')
        print(' ')
        del dataOnBins