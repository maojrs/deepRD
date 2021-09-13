import numpy as np

class noiseSampler:
    '''
    Main class to sample noise from data-based models. The sample function takes
    conditionedVariables as input. This is a list with all the relevant variable for
    a given implementation
    '''
    def __init__(self, dimension = 3):
        self.dimension = dimension

    def sample(self, conditionedVariables):
        return np.random.normal(0., 1, self.dimension)
