import numpy as np

class noiseSampler:
    '''
    Main class to sample noise from data-based models. The sample function takes
    conditionedVariables as input, which is a list with all the relevant variable for
    a given implementation. The sampling model can be arbitrary, it only needs to have
    a function called sample that takes conditioned variables as input.
    '''
    def __init__(self, samplingModel= None):
        self.samplingModel = samplingModel

    def sample(self, conditionedVariables):
        try:
            return self.samplingModel.sample(conditionedVariables)
        except:
            return np.random.normal(0., 1, 3)
