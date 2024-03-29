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
            raise NotImplementedError("Please Implement sample method for given conditioned variables")

class defaultSamplingModel:
    '''
    Default sampler to be fed into noise sampler for testing cases
    '''
    def __init__(self, mean = [0,0,0], covariance = [[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]]):
        self.mean = mean
        self.covariance = covariance


    def sample(self, conditionedVariables):
        if isinstance(self.mean, list) and isinstance(self.covariance, list):
            return np.random.multivariate_normal(self.mean, self.covariance)
        else:
            return np.random.normal(self.mean, self.covariance)