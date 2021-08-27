try:
    from joblib import Parallel, delayed
    import multiprocessing
except:
    print("Multiprocessing libraries not found, parallel data generation will not work.")

class reactionModel:
    '''
    Parent (abstract) class for all reaction models
    '''

    def __init__(self):
        # Define default simulation parameters
        self.dt = 0.0001
        self.stride = 1
        self.tfinal = 1000
        self.timesteps = int(self.tfinal/self.dt)

    def setSimulationParameters(self, dt, stride, tfinal):
        '''
        Function to set simulation parameters. This will be inherited
        and used by child classes
        '''
        self.dt = dt
        self.stride = stride
        self.tfinal = tfinal
        self.timesteps = int(self.tfinal/self.dt)

    def populateReactionVectors(self):
        '''
        'Abstract' method needed to populates reaction vectors
        of a given reaction system. Each reaction vectors correspond to
        the copy numbe change of all the species involved
        for the corresponding reaction. Note this will need a reaction vectors
        variablev (reactionVectors) of dimension corresponding to the number
        of reactions times the number of chemical species involved.
        '''
        raise NotImplementedError("Please Implement populateReactionVectors method")

    def updatePropensities(self):
        '''
        'Abstract' method to update the propensities, since propensities
        depend on the copy number of the chemical species, they need to be
        updated every time the number of chemical species changes. The method
        can be made more efficient by only updating the propensities that
        changed. Note this will require a propensities variable (propensities)
        of dimension corresponding to the number of all possible reactions.
        '''
        raise NotImplementedError("Please Implement updatePropensities method")