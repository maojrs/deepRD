import numpy as np
from .noiseSampler import noiseSampler

class noiseSamplerBinnedData(noiseSampler):
    '''
    Sample noise from binned data conditioned on position (ri+1|qi). The sample function
    takes conditionedVariables as input. This is a list with all the relevant variable for
    a given implementation. The binned data uses the classes defined on binning.py
    '''
    def __init__(self, binnedData, sampleDimension = 3):
        # inherit all methods from parent class
        super().__init__(sampleDimension)
        self.binnedData = binnedData

    def sample(self, conditionedVariables):
        return self.binnedData.sample(conditionedVariables)