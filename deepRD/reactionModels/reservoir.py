from .reactionModel import reactionModel


class reservoir(reactionModel):

    def __init__(self, perParticleJumpRate, reservoirVolume, reservoirConcentation):
        # inherit all methods from parent class
        super().__init__()

        # Define default initial conditions and names
        self.X = np.array([0])
        self.perParticleJumpRate = perParticleJumpRate
        self.reservoirVolume = reservoirVolume
        self.reservoirConcentation = reservoirConcentation
        self.reservoirNumParticles = self.reservoirConcentation * self.reservoirVolume
        self.names = ['injection_events']

        # Define base paramters, based on ODE model and data generation parameters
        self.nreactions = 1
        self.reactionVectors = np.zeros([self.nreactions, len(self.X)])
        self.propensities = np.zeros(self.nreactions)
        self.populateReactionVectors()
        self.updatePropensities()

    def setModelParameters(self, perParticleJumpRate, reservoirVolume, reservoirConcentation):
        self.perParticleJumpRate = perParticleJumpRate
        self.reservoirVolume = reservoirVolume
        self.reservoirConcentation = reservoirConcentation
        self.reservoirNumParticles = self.reservoirConcentation * self.reservoirVolume

    def populateReactionVectors(self):
        self.reactionVectors[0] = [1]  # Each injection event adds one

    def updatePropensities(self):
        x = self.X[0]
        self.propensities[0] = self.perParticleJumpRate * (self.reservoirNumParticles - x)