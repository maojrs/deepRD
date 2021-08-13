from abc import ABC, abstractmethod

class reactionModel(ABC):

    @abstractmethod
    def populateReactionVectors(self):
        """ Abstract method needed to populates reaction vectors
        of a given reaction system. Each reaction vectors correspond to
        the copy numbe change of all the species involved
        for the corresponding reaction. Note this will need a reaction vectors
        variablev (reactionVectors) of dimension corresponding to the number
        of reactions times the number of chemical species involved. """
        pass

    @abstractmethod
    def updatePropensities(self):
        """ Abstract method to update the propensities, since propensities
        depend on the copy number of the chemical species, they need to be
        updated every time the number of chemical species changes. The method
        can be made more efficient by only updating the propensities that
        changed. Note this will require a propensities variable (lambdas)
        of dimension corresponding to the number of all possible reactions."""
        pass