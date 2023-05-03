import numpy as np
import sys
from .langevin import langevin
from .. import trajectoryTools


class langevinNoiseSampler(langevin):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle following the Langevin equation,
    where the noise term and interaction terms in the velocity equation are sampled from an alternate data-based
    model. Takes same input as Langevin integrator with the addition of a noise sampler. The noise sampler
    takes certain input and outputs a corresponding noise term. The variable particle.aux1 and particle.aux2 store
    the values of the noise/interactions small term r for previous time steps, t-dt and t-2dt,
    respectively.
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'qi'):
        self.noiseSampler = noiseSampler
        self.conditionedOn = conditionedOn
        self.integratorType = "dataDrivenABOBA"
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, kBT, boxsize, boundary, equilibrationSteps)


    def prepareSimulation(self, particleList):
        '''
        Overrides parent method to set up additional variables at the start of the routine.
        See parent method for more details. The variable particle.aux1 and particle.aux2 store
        the values of the noise/interactions small term r for previous time steps, t-dt and t-2dt,
        respectively.
        '''
        for part in particleList:
            part.aux1 = np.zeros(particleList.dimension)
            part.aux2 = np.zeros(particleList.dimension)
            part.aux3 = np.zeros(particleList.dimension)
        self.currentOrNext = 'next'
        self.calculateForceField(particleList)
        self.firstRun = False

    def getConditionedVars(self, particle, index = None):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return (particle.aux1)
        elif self.conditionedOn == 'ririm':
            return np.concatenate((particle.aux1, particle.aux2))
        elif self.conditionedOn == 'qi':
            return (particle.nextPosition)
        elif self.conditionedOn == 'qiri':
            return np.concatenate((particle.nextPosition, particle.aux1))
        elif self.conditionedOn == 'qiririm':
            return np.concatenate((particle.nextPosition, particle.aux1, particle.aux2))
        elif self.conditionedOn == 'pi':
            return (particle.nextVelocity)
        elif self.conditionedOn == 'piri':
            return np.concatenate((particle.nextVelocity, particle.aux1))
        elif self.conditionedOn == 'piririm':
            return np.concatenate((particle.nextVelocity, particle.aux1, particle.aux2))
        elif self.conditionedOn == 'qipi':
            return ((particle.nextPosition, particle.nextVelocity))
        elif self.conditionedOn == 'qipiri':
            return np.concatenate((particle.nextPosition, particle.nextVelocity, particle.aux1))
        elif self.conditionedOn == 'qipiririm':
            return np.concatenate((particle.nextPosition, particle.nextVelocity, particle.aux1, particle.aux2))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.integrateBOB(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()


    def integrateBOB(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        for i, particle in enumerate(particleList):
            # Calculate friction term and external potential term
            expterm = np.exp(-dt * self.Gamma / particle.mass)
            frictionForceTerm = particle.nextVelocity * expterm
            frictionForceTerm += (1 + expterm) * self.forceField[i] * dt/(2*particle.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars(particle, i)
            interactionNoiseTerm = self.noiseSampler.sample(conditionedVars)

            ## For testing and consistency.
            #xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            #interactionNoiseTerm = xi / particle.mass * np.random.normal(0., 1, particle.dimension)

            particle.aux2 = 1.0 * particle.aux1
            particle.aux1 = 1.0 * interactionNoiseTerm
            particle.nextVelocity = frictionForceTerm + interactionNoiseTerm

    def propagate(self, particleList, showProgress = False, outputAux = False):
        '''
        Same as Langevin propagator, but this one can output particle.aux1, corresponding to the 'r' variables.
        '''
        if self.firstRun:
            self.prepareSimulation(particleList)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
        # Begins integration
        time = 0.0
        tTraj = [time]
        xTraj = [particleList.positions]
        vTraj = [particleList.velocities]
        if outputAux:
            rTraj = [particleList.aux1List]
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                tTraj.append(time)
                xTraj.append(particleList.positions)
                vTraj.append(particleList.velocities)
                if outputAux:
                    rTraj.append(particleList.aux1List)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time/ self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        self.firstRun = True
        if outputAux:
            return np.array(tTraj), np.array(xTraj), np.array(vTraj), np.array(rTraj)
        else:
            return np.array(tTraj), np.array(xTraj), np.array(vTraj)

    def propagateFPT(self, particleList, finalPosition, threshold):
        '''
        Same as propagate, but also takes a finalPosition argument. If the final position is reached the propagation
        is stopped and the time is printed.
        '''
        if self.firstRun:
            self.prepareSimulation(particleList)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
        # Begins integration
        time = 0.0
        condition = True
        while (condition):
            self.integrateOne(particleList)
            # Update variables
            time = time + self.dt
            if time > self.tfinal:
                condition = False
            elif np.linalg.norm(particleList[0].position - finalPosition) < threshold:
                condition = False
                return 'success', time
        return 'failed', time


class langevinNoiseSamplerDimer(langevinNoiseSampler):
    '''
    Specialized version of the langevinNoiseSampler class to integrate the dynamics of a dimer bonded by
    some potential (e.g. bistable or harmonic).
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'dqi'):
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, noiseSampler, kBT, boxsize,
                 boundary, equilibrationSteps, conditionedOn)
        self.relDistance = None # Between dimer particles
        self.axisRelVelocity = None # Along axis connecting particles
        self.centerMassVelocity = None # Divided into norm along axis and norm along perepndicular



    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.calculateRelDistanceVelocity(particleList)
        self.integrateBOB(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()

    def calculateRelDistanceVelocity(self, particleList):
        numParticles = len(particleList)
        self.relDistance = np.zeros(numParticles)
        self.axisRelVelocity = np.zeros(numParticles)
        self.centerMassVelocity = np.zeros([numParticles,2])
        for i in range(int(numParticles/2)):
            relPos = trajectoryTools.relativePosition(particleList[2 * i].nextPosition, particleList[2 * i + 1].nextPosition,
                                                       self.boundary, self.boxsize)
            relVel = particleList[2 * i + 1].nextVelocity - particleList[2 * i].nextVelocity

            velCM = 0.5 * (particleList[2 * i].nextVelocity + particleList[2 * i + 1].nextVelocity)
            normRelPos = np.linalg.norm(relPos)
            unitRelPos = relPos / normRelPos
            axisRelVel = np.dot(relVel, unitRelPos)
            axisVelCM = np.dot(velCM, unitRelPos)
            normAxisVelCM = np.linalg.norm(axisVelCM)
            vecAxisVelCM = axisVelCM * unitRelPos
            orthogonalVelCM = velCM - vecAxisVelCM
            normOrthogonalVelCM = np.linalg.norm(orthogonalVelCM)
            self.relDistance[2*i] = normRelPos
            self.relDistance[2*i+1] = normRelPos
            self.axisRelVelocity[2*i] = axisRelVel
            self.axisRelVelocity[2*i+1] = -1*axisRelVel
            self.centerMassVelocity[2*i] = np.array([normAxisVelCM, normOrthogonalVelCM])
            self.centerMassVelocity[2*i+1] = np.array([normAxisVelCM, normOrthogonalVelCM])

    def getConditionedVars(self, particle, index = None):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return (particle.aux1)
        elif self.conditionedOn == 'ririm':
            return np.concatenate((particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dqi':
            return (self.relDistance[index])
        elif self.conditionedOn == 'dqiri':
            return np.concatenate((np.array([self.relDistance[index]]), particle.aux1))
        elif self.conditionedOn == 'dqiririm':
            return np.concatenate((np.array([self.relDistance[index]]), particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dpi':
            return (self.axisRelVelocity[index])
        elif self.conditionedOn == 'dpiri':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle.aux1))
        elif self.conditionedOn == 'dpiririm':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dqidpi':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]])))
        elif self.conditionedOn == 'dqidpiri':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), particle.aux1))
        elif self.conditionedOn == 'dqidpiririm':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), particle.aux1, particle.aux2))
        elif self.conditionedOn == 'vi':
            return (self.centerMassVelocity[index])
        elif self.conditionedOn == 'viri':
            return np.concatenate((self.centerMassVelocity[index], particle.aux1))
        elif self.conditionedOn == 'viririm':
            return np.concatenate((self.centerMassVelocity[index], particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dqivi':
            return np.concatenate((np.array([self.relDistance[index]]), self.centerMassVelocity[index]))
        elif self.conditionedOn == 'dqiviri':
            return np.concatenate((np.array([self.relDistance[index]]), self.centerMassVelocity[index], particle.aux1))
        elif self.conditionedOn == 'dqiviririm':
            return np.concatenate((np.array([self.relDistance[index]]), self.centerMassVelocity[index], particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dpivi':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), self.centerMassVelocity[index,0]))
        elif self.conditionedOn == 'dpiviri':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), self.centerMassVelocity[index,0], particle.aux1))
        elif self.conditionedOn == 'dpiviririm':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), self.centerMassVelocity[index,0], particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dqidpivi':
            return ((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), self.centerMassVelocity[index]))
        elif self.conditionedOn == 'dqidpiviri':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), self.centerMassVelocity[index], particle.aux1))
        elif self.conditionedOn == 'dqidpiviririm':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), self.centerMassVelocity[index], particle.aux1, particle.aux2))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def propagateFPT(self, particleList, initialSeparation, finalSeparation, threshold):
        '''
        Same as propagate, but also takes a finalPosition argument. If the final position is reached the propagation
        is stopped and the time is printed.
        '''
        if self.firstRun:
            self.prepareSimulation(particleList)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
        # Begins integration
        time = 0.0
        condition = True
        while (condition):
            self.integrateOne(particleList)
            # Update variables
            time = time + self.dt
            relPos = trajectoryTools.relativePosition(particleList[0].position,
                                                      particleList[1].position,
                                                      self.boundary, self.boxsize)
            relDistance = np.linalg.norm(relPos)
            if time > self.tfinal:
                condition = False
            elif np.abs(relDistance - finalSeparation) <= threshold:
                condition = False
                return 'success', time
        return 'failed', time

class langevinNoiseSamplerDimer2(langevinNoiseSamplerDimer):
    '''
    Alternative specialized version of the langevinNoiseSampler class to integrate the dynamics of a dimer bonded by
    some potential (e.g. bistable or harmonic).
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'vi'):
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, noiseSampler, kBT, boxsize,
                 boundary, equilibrationSteps, conditionedOn)
        self.componentVelocity = None # Divided into norm along axis and norm along perepndicular



    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.calculateRelDistanceVelocity(particleList)
        self.calculateComponentVelocity(particleList)
        self.integrateBOB(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()

    def calculateComponentVelocity(self, particleList):
        numParticles = len(particleList)
        self.componentVelocity = np.zeros([numParticles,2])
        for i in range(int(numParticles/2)):
            relPos = trajectoryTools.relativePosition(particleList[2 * i].nextPosition, particleList[2 * i + 1].nextPosition,
                                                       self.boundary, self.boxsize)
            normRelPos = np.linalg.norm(relPos)
            unitRelPos = relPos / normRelPos
            vel1 = particleList[2 * i].nextVelocity
            vel2 = particleList[2 * i + 1].nextVelocity
            axisVel1 = np.dot(vel1, unitRelPos)
            axisVel2 = np.dot(vel2, -1.0*unitRelPos)
            orthogonalVel1 = vel1 - axisVel1
            orthogonalVel2 = vel2 - axisVel2
            normOrthogonalVel1 = np.linalg.norm(orthogonalVel1)
            normOrthogonalVel2 = np.linalg.norm(orthogonalVel2)
            self.componentVelocity[2*i] = np.array([axisVel1, normOrthogonalVel1])
            self.componentVelocity[2*i+1] = np.array([axisVel2, normOrthogonalVel2])

    def getConditionedVars(self, particle, index = None):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return (particle.aux1)
        elif self.conditionedOn == 'ririm':
            return np.concatenate((particle.aux1, particle.aux2))
        elif self.conditionedOn == 'vi':
            return (self.componentVelocity[index])
        elif self.conditionedOn == 'viri':
            return np.concatenate((self.componentVelocity[index], particle.aux1))
        elif self.conditionedOn == 'viririm':
            return np.concatenate((self.componentVelocity[index], particle.aux1, particle.aux2))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


class langevinNoiseSamplerDimer3(langevinNoiseSampler):
    '''
    Alternative specialized version of the langevinNoiseSampler class to integrate the dynamics of a dimer bonded by
    some potential (e.g. bistable or harmonic).
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'v2i'):
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, noiseSampler, kBT, boxsize,
                 boundary, equilibrationSteps, conditionedOn)
        self.relPosition = None
        self.rotatedVelocity = None



    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.calculateRotatedVelocity(particleList)
        self.integrateBOB2(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()

    def integrateBOB2(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        for i, particle in enumerate(particleList):
            # Calculate friction term and external potential term
            expterm = np.exp(-dt * self.Gamma / particle.mass)
            frictionForceTerm = particle.nextVelocity * expterm
            frictionForceTerm += (1 + expterm) * self.forceField[i] * dt/(2*particle.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars(particle, i)
            rotatedInteractionNoiseTerm = self.noiseSampler.sample(conditionedVars)

            # Rotate back riplus
            interactionNoiseTerm = trajectoryTools.rotateVecInverse(self.relPosition[i], rotatedInteractionNoiseTerm)

            ## For testing and consistency.
            #xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            #interactionNoiseTerm = xi / particle.mass * np.random.normal(0., 1, particle.dimension)

            particle.aux2 = 1.0 * particle.aux1
            particle.aux1 = 1.0 * interactionNoiseTerm
            particle.nextVelocity = frictionForceTerm + interactionNoiseTerm


    def calculateRotatedVelocity(self, particleList):
        numParticles = len(particleList)
        self.relPosition = np.zeros([numParticles,3])
        self.rotatedVelocity = np.zeros([numParticles,3])
        for i in range(int(numParticles/2)):
            relPos = trajectoryTools.relativePosition(particleList[2 * i].nextPosition,
                                                      particleList[2 * i + 1].nextPosition,
                                                      self.boundary, self.boxsize)
            relPosUnit = relPos/np.linalg.norm(relPos)
            self.relPosition[2*i] = relPos
            self.relPosition[2*i+1] = -1*relPos
            self.rotatedVelocity[2*i] = trajectoryTools.rotateVec(relPosUnit, particleList[2*i].nextVelocity)
            self.rotatedVelocity[2*i+1] = trajectoryTools.rotateVec(-1*relPosUnit, particleList[2*i+1].nextVelocity)

    def getConditionedVars(self, particle, index = None):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return (particle.aux1)
        elif self.conditionedOn == 'ririm':
            return np.concatenate((particle.aux1, particle.aux2))
        elif self.conditionedOn == 'vi':
            return (self.rotatedVelocity[index])
        elif self.conditionedOn == 'viri':
            return np.concatenate((self.rotatedVelocity[index], particle.aux1))
        elif self.conditionedOn == 'viririm':
            return np.concatenate((self.rotatedVelocity[index], particle.aux1, particle.aux2))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


class langevinNoiseSamplerDimerConstrained1D(langevinNoiseSamplerDimer):
    '''
    Specialized version of the langevinNoiseSampler class to integrate the dynamics of a dimer bonded by
    some potential (e.g. bistable or harmonic).
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'dqi'):
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, noiseSampler, kBT, boxsize,
                 boundary, equilibrationSteps, conditionedOn)

    def getConditionedVars1D(self, particle, index = None, j = 0):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return np.array([particle.aux1[j]])
        elif self.conditionedOn == 'ririm':
            return np.array([particle.aux1[j], particle.aux2[j]])
        elif self.conditionedOn == 'dqi':
            return (self.relDistance[index])
        elif self.conditionedOn == 'dqiri':
            return np.concatenate((np.array([self.relDistance[index]]), particle.aux1))
        elif self.conditionedOn == 'dqiririm':
            return np.concatenate((np.array([self.relDistance[index]]), particle.aux1, particle.aux2))
        elif self.conditionedOn == 'dpi':
            return (self.axisRelVelocity[index])
        elif self.conditionedOn == 'dpiri':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle.aux1))
        elif self.conditionedOn == 'dpiririm':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle.aux1, particle.aux2))
        elif self.conditionedOn == 'qi':
            return np.array([particle.nextPosition[j]])
        elif self.conditionedOn == 'qiri':
            return np.array([particle.nextPosition[j], particle.aux1[j]])
        elif self.conditionedOn == 'qiririm':
            return np.array([particle.nextPosition[j], particle.aux1[j], particle.aux2[j]])
        elif self.conditionedOn == 'pi':
            return np.array([particle.nextVelocity[j]])
        elif self.conditionedOn == 'piri':
            return np.array([particle.nextVelocity[j], particle.aux1[j]])
        elif self.conditionedOn == 'piririm':
            return np.array([particle.nextVelocity[j], particle.aux1[j], particle.aux2[j]])
        elif self.conditionedOn == 'qipi':
            return np.array([particle.nextPosition[j], particle.nextVelocity[j]])
        elif self.conditionedOn == 'qipiri':
            return np.array([particle.nextPosition[j], particle.nextVelocity[j], particle.aux1[j]])
        elif self.conditionedOn == 'qipiririm':
            return np.array([particle.nextPosition[j], particle.nextVelocity[j], particle.aux1[j], particle.aux2[j]])
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.calculateRelDistanceVelocity(particleList)
        self.integrateBOB1D(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocitiesIndex(0)

    def integrateBOB1D(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        for i, particle in enumerate(particleList):
            # Calculate friction term and external potential term
            expterm = np.exp(-dt * self.Gamma / particle.mass)
            frictionForceTerm = particle.nextVelocity * expterm
            frictionForceTerm += (1 + expterm) * self.forceField[i] * dt/(2*particle.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars1D(particle, i, 0)
            interactionNoiseTerm = self.noiseSampler.sample(conditionedVars)

            ## For testing and consistency.
            #xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            #interactionNoiseTerm = xi / particle.mass * np.random.normal(0., 1, particle.dimension)

            particle.aux2 = 1.0* particle.aux1
            particle.aux1 = 1.0 * interactionNoiseTerm
            interactionNoiseTerm = np.append(interactionNoiseTerm, np.array([0.,0]))
            particle.nextVelocity = frictionForceTerm + interactionNoiseTerm


class langevinNoiseSamplerDimerConstrained1DGlobal(langevinNoiseSamplerDimer):
    '''
    Specialized version of the langevinNoiseSampler class to integrate the dynamics of a dimer bonded by
    some potential (e.g. bistable or harmonic).
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'dqi'):
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, noiseSampler, kBT, boxsize,
                 boundary, equilibrationSteps, conditionedOn)

    def getConditionedVars1D(self, particle1, particle2, index = None, j = 0):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return np.array([particle1.aux1[j], particle2.aux1[j]])
        elif self.conditionedOn == 'ririm':
            return np.array([particle1.aux1[j], particle2.aux1[j], particle1.aux2[j], particle2.aux2[j]])
        elif self.conditionedOn == 'dqi':
            return (self.relDistance[index])
        elif self.conditionedOn == 'dqiri':
            return np.concatenate((np.array([self.relDistance[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dqiririm':
            return np.concatenate((np.array([self.relDistance[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'dpi':
            return (self.axisRelVelocity[index])
        elif self.conditionedOn == 'dpiri':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dpiririm':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'dpivi':
            return np.array([self.axisRelVelocity[index], self.centerMassVelocity[index,0]])
        elif self.conditionedOn == 'dpiviri':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), np.array([self.centerMassVelocity[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dpiviririm':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), np.array([self.centerMassVelocity[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'dqidpi':
            return np.array([self.relDistance[index], self.axisRelVelocity[index]])
        elif self.conditionedOn == 'dqidpiri':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dqidpiririm':
            return np.concatenate((np.array([self.relDistance[index]]),np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'qi':
            return np.array([particle1.nextPosition[j], particle2.nextPosition[j]])
        elif self.conditionedOn == 'qiri':
            return np.array([particle1.nextPosition[j], particle2.nextPosition[j], particle1.aux1[j], particle2.aux1[j]])
        elif self.conditionedOn == 'qiririm':
            return np.array([particle1.nextPosition[j], particle2.nextPosition[j], particle1.aux1[j], particle2.aux1[j], particle1.aux2[j], particle2.aux2[j]])
        elif self.conditionedOn == 'pi':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j]])
        elif self.conditionedOn == 'piri':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux1[j], particle2.aux1[j]])
        elif self.conditionedOn == 'piririm':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux1[j], particle2.aux1[j], particle1.aux2[j], particle2.aux2[j]])
        elif self.conditionedOn == 'pidqi':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j]], self.relDistance[index])
        elif self.conditionedOn == 'pidqiri':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], self.relDistance[index], particle1.aux1[j], particle2.aux1[j]])
        elif self.conditionedOn == 'pidqiririm':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], self.relDistance[index], particle1.aux1[j], particle2.aux1[j], particle1.aux2[j], particle2.aux2[j]])
        elif self.conditionedOn == 'pipim':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux3[j], particle2.aux3[j]])
        elif self.conditionedOn == 'pipimri':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux3[j], particle2.aux3[j], particle1.aux1[j], particle2.aux1[j]])
        elif self.conditionedOn == 'pipimririm':
            return np.array([particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux3[j], particle2.aux3[j],particle1.aux1[j], particle2.aux1[j], particle1.aux2[j], particle2.aux2[j]])
        elif self.conditionedOn == 'qipi':
            return np.array([particle1.nextPosition[j], particle2.nextPosition[j], particle1.nextVelocity[j], particle2.nextVelocity[j]])
        elif self.conditionedOn == 'qipiri':
            return np.array([particle1.nextPosition[j], particle2.nextPosition[j], particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux1[j], particle2.aux1[j]])
        elif self.conditionedOn == 'qipiririm':
            return np.array([particle1.nextPosition[j], particle2.nextPosition[j], particle1.nextVelocity[j], particle2.nextVelocity[j], particle1.aux1[j], particle2.aux1[j], particle1.aux2[j], particle2.aux2[j]])
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.calculateRelDistanceVelocity(particleList)
        self.integrateBOB1D(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocitiesIndex(0)

    def integrateBOB1D(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        for i in range(int(len(particleList)/2)):
            particle1 = particleList[2*i]
            particle2 = particleList[2*i+1]
            # Calculate friction term and external potential term
            expterm1 = np.exp(-dt * self.Gamma / particle1.mass)
            expterm2 = np.exp(-dt * self.Gamma / particle2.mass)
            frictionForceTerm1 = particle1.nextVelocity * expterm1
            frictionForceTerm2 = particle2.nextVelocity * expterm2
            frictionForceTerm1 += (1 + expterm1) * self.forceField[2*i] * dt/(2*particle1.mass)
            frictionForceTerm2 += (1 + expterm2) * self.forceField[2*i+1] * dt/(2*particle2.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars1D(particle1, particle2, 0)
            interactionNoiseTerm = self.noiseSampler.sample(conditionedVars)

            ## For testing and consistency.
            #xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            #interactionNoiseTerm = xi / particle.mass * np.random.normal(0., 1, particle.dimension)

            particleList[2 * i].aux3 = 1.0 * particleList[2 * i].nextVelocity
            particleList[2 * i + 1].aux3 = 1.0 * particleList[2 * i + 1].nextVelocity
            particleList[2*i].aux2 = 1.0 * particleList[2*i].aux1
            particleList[2*i+1].aux2 = 1.0 * particleList[2*i+1].aux1
            particleList[2*i].aux1 = np.array([1.0 * interactionNoiseTerm[0]])
            particleList[2*i+1].aux1 = np.array([1.0 * interactionNoiseTerm[1]])

            interactionNoiseTerm1 = np.append(interactionNoiseTerm[0], np.array([0.,0]))
            interactionNoiseTerm2 = np.append(interactionNoiseTerm[1], np.array([0.,0]))
            particleList[2*i].nextVelocity = frictionForceTerm1 + interactionNoiseTerm1
            particleList[2*i+1].nextVelocity = frictionForceTerm2 + interactionNoiseTerm2


class langevinNoiseSamplerDimerGlobal(langevinNoiseSamplerDimer):
    '''
    Specialized version of the langevinNoiseSampler class to integrate the dynamics of a dimer bonded by
    some potential (e.g. bistable or harmonic).
    '''

    def __init__(self, dt, stride, tfinal, Gamma, noiseSampler, kBT=1, boxsize = None,
                 boundary = 'periodic', equilibrationSteps = 0, conditionedOn = 'dqi'):
        # inherit methods from parent class
        super().__init__(dt, stride, tfinal, Gamma, noiseSampler, kBT, boxsize,
                 boundary, equilibrationSteps, conditionedOn)
        self.relPosition = None
        self.rotatedVelocity = None

    def calculateRotatedVelocity(self, particleList):
        numParticles = len(particleList)
        self.relPosition = np.zeros([numParticles,3])
        self.rotatedVelocity = np.zeros([numParticles,3])
        for i in range(int(numParticles/2)):
            relPos = trajectoryTools.relativePosition(particleList[2 * i].nextPosition,
                                                      particleList[2 * i + 1].nextPosition,
                                                      self.boundary, self.boxsize)
            relPosUnit = relPos/np.linalg.norm(relPos)
            self.relPosition[2*i] = relPos
            self.relPosition[2*i+1] = -1*relPos
            self.rotatedVelocity[2*i] = trajectoryTools.rotateVec(relPosUnit, particleList[2*i].nextVelocity)
            self.rotatedVelocity[2*i+1] = trajectoryTools.rotateVec(relPosUnit, particleList[2*i+1].nextVelocity)

    def getConditionedVars(self, particle1, particle2, index = None):
        '''
        Returns variable upon which the binning is conditioned for the integration. Can extend to
        incorporate conditioning on velocities.
        '''
        if self.conditionedOn == 'ri':
            return np.concatenate((particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'ririm':
            return np.concatenate((particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'dqi':
            return (self.relDistance[index])
        elif self.conditionedOn == 'dqiri':
            return np.concatenate((np.array([self.relDistance[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dqiririm':
            return np.concatenate((np.array([self.relDistance[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'dpi':
            return (self.axisRelVelocity[index])
        elif self.conditionedOn == 'dpiri':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dpiririm':
            return np.concatenate((np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'dqidpi':
            return np.array([self.relDistance[index], self.axisRelVelocity[index]])
        elif self.conditionedOn == 'dqidpiri':
            return np.concatenate((np.array([self.relDistance[index]]), np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'dqidpiririm':
            return np.concatenate((np.array([self.relDistance[index]]),np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'qi':
            return np.concatenate((particle1.nextPosition, particle2.nextPosition))
        elif self.conditionedOn == 'qiri':
            return np.concatenate((particle1.nextPosition, particle2.nextPosition, particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'qiririm':
            return np.concatenate((particle1.nextPosition, particle2.nextPosition, particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'pi':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity))
        elif self.conditionedOn == 'piri':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'piririm':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'pidqi':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, np.array([self.axisRelVelocity[index]])))
        elif self.conditionedOn == 'pidqiri':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'pidqiririm':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, np.array([self.axisRelVelocity[index]]), particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'pipim':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, particle1.aux3, particle2.aux3))
        elif self.conditionedOn == 'pipimri':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, particle1.aux3, particle2.aux3, particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'pipimririm':
            return np.concatenate((particle1.nextVelocity, particle2.nextVelocity, particle1.aux3, particle2.aux3, particle1.aux1, particle2.aux1,particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'qipi':
            return np.concatenate((particle1.nextPosition, particle2.nextPosition, particle1.nextVelocity, particle2.nextVelocity))
        elif self.conditionedOn == 'qipiri':
            return np.concatenate((particle1.nextPosition, particle2.nextPosition, particle1.nextVelocity, particle2.nextVelocity, particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'qipiririm':
            return np.concatenate((particle1.nextPosition, particle2.nextPosition, particle1.nextVelocity, particle2.nextVelocity, particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        elif self.conditionedOn == 'vi':
            return np.array([self.rotatedVelocity[index], self.rotatedVelocity[index+1]])
        elif self.conditionedOn == 'viri':
            return np.concatenate((self.rotatedVelocity[index], self.rotatedVelocity[index+1], particle1.aux1, particle2.aux1))
        elif self.conditionedOn == 'viririm':
            return np.concatenate((self.rotatedVelocity, particle1.aux1, particle2.aux1, particle1.aux2, particle2.aux2))
        else:
            sys.stdout.write("Unknown conditioned variables, check getConditionedVars in langevinNoiseSampler.\r")


    def integrateOne(self, particleList):
        ''' Integrates one time step of data-driven version of ABOBA '''
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        self.calculateForceField(particleList)
        self.calculateRotatedVelocity(particleList)
        self.calculateRelDistanceVelocity(particleList)
        self.integrateBOBGlobal(particleList, self.dt)
        self.integrateA(particleList, self.dt/2.0)
        self.enforceBoundary(particleList)
        particleList.updatePositionsVelocities()

    def integrateBOBGlobal(self, particleList, dt):
        '''Integrates BOB integrations step at once. This is required to separate the noise Sampler from the
        external potential. '''
        for i in range(int(len(particleList)/2)):
            particle1 = particleList[2*i]
            particle2 = particleList[2*i+1]
            # Calculate friction term and external potential term
            expterm1 = np.exp(-dt * self.Gamma / particle1.mass)
            expterm2 = np.exp(-dt * self.Gamma / particle2.mass)
            frictionForceTerm1 = particle1.nextVelocity * expterm1
            frictionForceTerm2 = particle2.nextVelocity * expterm2
            frictionForceTerm1 += (1 + expterm1) * self.forceField[2*i] * dt/(2*particle1.mass)
            frictionForceTerm2 += (1 + expterm2) * self.forceField[2*i+1] * dt/(2*particle2.mass)
            # Calculate interaction and noise term from noise sampler
            conditionedVars = self.getConditionedVars(particle1, particle2, 0)

            # Sample interaction noisterm
            interactionNoiseTerm = self.noiseSampler.sample(conditionedVars)
            #rotatedInteractionNoiseTerm = self.noiseSampler.sample(conditionedVars)

            # Rotate back riplus if needed
            interactionNoiseTerm1 = interactionNoiseTerm[0:3]
            interactionNoiseTerm2 = interactionNoiseTerm[3:6]
            #interactionNoiseTerm1 = trajectoryTools.rotateVecInverse(self.relPosition[2*i], rotatedInteractionNoiseTerm[0:3])
            #interactionNoiseTerm2 = trajectoryTools.rotateVecInverse(self.relPosition[2*i], rotatedInteractionNoiseTerm[3:6])

            ## For testing and consistency.
            #xi = np.sqrt(self.kBT * particle.mass * (1 - np.exp(-2 * self.Gamma * dt / particle.mass)))
            #interactionNoiseTerm = xi / particle.mass * np.random.normal(0., 1, particle.dimension)

            particleList[2 * i].aux3 = 1.0 * particleList[2*i].nextVelocity
            particleList[2 * i + 1].aux3 = 1.0 * particleList[2*i+1].nextVelocity
            particleList[2*i].aux2 = 1.0 * particleList[2*i].aux1
            particleList[2*i+1].aux2 = 1.0 * particleList[2*i+1].aux1
            particleList[2*i].aux1 = interactionNoiseTerm1
            particleList[2*i+1].aux1 = interactionNoiseTerm2

            particleList[2*i].nextVelocity = frictionForceTerm1 + interactionNoiseTerm1
            particleList[2*i+1].nextVelocity = frictionForceTerm2 + interactionNoiseTerm2