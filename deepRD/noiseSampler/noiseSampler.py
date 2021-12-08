import numpy as np

class noiseSampler:
    '''
    Main class to sample noise from data-based models. The sample function takes
    conditionedVariables as input, which is a list with all the relevant variable for
    a given implementation. The sampling model can be arbitrary, it only needs to have
    a function called sample that takes conditioned variables as input.
    '''
    def __init__(self, samplingModel= None, defaultVariance = 1):
        self.samplingModel = samplingModel
        self.defaultVariance = defaultVariance

    def sample(self, conditionedVariables):
        try:
            return self.samplingModel.sample(conditionedVariables)
        except:
            return np.random.normal(0., self.defaultVariance, 3)
