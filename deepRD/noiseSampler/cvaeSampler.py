import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
import os
from torch import nn
from collections import OrderedDict
#from torch.utils.data import DataLoader


class cvaeSampler(nn.Module):

    """
    Model for sampling of the interaction noise term in reduced simulations. 
    The model is a Conditional Variational Autoencoder (CVAE) architecture.

    Parameters:
    latentDims (int): dimensionality of latent (encoded) space
    loadPretrained (str): filepath of pre-trained model to be loaded
    conditionedOn (str): ['piri', 'pipimri', etc.] - set of the variables used for conditioning of the decoding/encoding
    systemType (str): ['bistable', 'dimer'] 
    hidden_dims (1-D array of int): e.g. [128, 64, 32] ; determines the encoder/decoder structure given as subsequent hidden layers
    """

    def __init__(self, latentDims, loadPretrained, conditionedOn, systemType, hidden_dims=None, batch_norm=False, dropout_rate=0, fhl=20,):
        super().__init__()
        self.conditionedOn = conditionedOn
        self.systemType = systemType
        self.latentDims = latentDims
        self.loadPretrained = loadPretrained
        self.conditionDims = self.getConditionDims()
        self.inputDims = self.getInputDims()
        self.G = torch.distributions.Normal(0, 1)

        if hidden_dims==None:
            # Initialising template network architecture
            self.fhl = fhl

            self.encoder = nn.Sequential(
                nn.Linear(self.inputDims+self.conditionDims, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 40),
                nn.ReLU(),
                nn.Linear(40, 20),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(self.latentDims+self.conditionDims, 20),
                nn.ReLU(),
                nn.Linear(20, 40),
                nn.ReLU(),
                nn.Linear(40, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.inputDims)
            )

        else:
            # Building custom network architecture

            self.fhl = hidden_dims[-1]

            if dropout_rate>0 and batch_norm==True:
                print('Use either BN or Dropout.')

            elif batch_norm: # Using Batch Norm only 
                self.encoder = nn.Sequential(
                nn.Linear(self.inputDims+self.conditionDims, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU()
                )

                self.decoder = nn.Sequential(
                nn.Linear(self.latentDims+self.conditionDims, hidden_dims[-1]),
                nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU()
                )

            else: # No BN
                
                self.encoder = nn.Sequential(
                nn.Linear(self.inputDims+self.conditionDims, hidden_dims[0]),
                nn.ReLU()
                )   

                self.decoder = nn.Sequential(
                nn.Linear(self.latentDims+self.conditionDims, hidden_dims[-1]),
                nn.ReLU()
                )

            for i in range(1, len(hidden_dims)):

                if dropout_rate>0:
                    self.encoder.append(nn.Dropout(dropout_rate))
                self.encoder.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                if batch_norm:
                    self.encoder.append(nn.BatchNorm1d(hidden_dims[i]))
                self.encoder.append(nn.ReLU())


                if dropout_rate>0:
                    self.decoder.append(nn.Dropout(dropout_rate))
                self.decoder.append(nn.Linear(hidden_dims[-i], hidden_dims[-(i+1)]))
                if batch_norm:
                    self.decoder.append(nn.BatchNorm1d(hidden_dims[-(i+1)]))
                self.decoder.append(nn.ReLU())

            self.decoder.append(nn.Linear(hidden_dims[0], self.inputDims))

        self.linear1 = nn.Linear(self.fhl, self.latentDims)
        self.linear2 = nn.Linear(self.fhl, self.latentDims)
        self.load_model()


    def load_model(self):
        if self.loadPretrained==None:
            print('Untrained model initialized. Conditioned on:', conditionedOn)
        else:
            print('Loading pretrained model: ' + self.loadPretrained)
            self.load_state_dict(torch.load(self.loadPretrained))
            print('Model parameters loaded.')

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*self.G.sample(mu.shape)

    def sample(self, label, num_samples=1):
        '''
         Here a slight workaround to get compatibility between model and integrator.
         Model is designed and trained to work on Torch tensors of size (num_samples, *), meanwhile 
         integrator works on 1-D arrays. 

         Returns: 1-D Torch Tensor of size (3)
        '''

        mean = 0
        std = 1.2

        if isinstance(label, np.ndarray):

            with torch.no_grad():
                label = torch.from_numpy(label).float()

                if self.systemType=='bistable':
                    # Extracting labels
                    if self.conditionedOn=="piri":
                        r = label[3:]
                        v = label[:3]
                        label = torch.cat((r,v))
                    elif self.conditionedOn=="piririm":
                        v = label[:3]
                        r = label[3:6]
                        r_prev = label[6:9]
                        label = torch.cat((v, r, r_prev))
                    elif self.conditionedOn=="pipimri":
                        v = label[:3]
                        v_prev = label[3:6]
                        r = label[6:9]
                        label = torch.cat((v, v_prev, r))

                elif self.systemType=='dimer':

                    if self.conditionedOn=="piri":
                        # ConditionedVars = (particle1.nextVelocity, particle2.nextVelocity, particle1.aux1, particle2.aux1)
                        v_1 = label[:3]
                        v_2 = label[3:6]
                        r_1 = label[6:9]
                        r_2 = label[9:12]
                        label = torch.cat((v_1, v_2, r_1, r_2)) 

                label = label.unsqueeze(0)
                samples = torch.normal(mean, std, (num_samples, self.latentDims))
                z_cond = torch.cat((samples, label), dim=1)
                out = np.array(self.decoder(z_cond).squeeze(0))

        else: 
            
            if label.dim()==1:
                label = label.unsqueeze(0)

            samples = torch.normal(mean, std, (num_samples, self.latentDims))
            z_cond = torch.cat((samples, label), dim=1)
            out = self.decoder(z_cond)

        return out

    def forward(self, x, y, returnLatent=False):
        x_cond = torch.cat((x,y), dim=1)
        x = self.encoder(x_cond)
        mu = self.linear1(x)
        logvar = self.linear2(x)
        z = self.reparametrize(mu, logvar)
        z_cond = torch.cat((z, y), dim=1)
        output = self.decoder(z_cond)

        if returnLatent==True:
            return output, mu, logvar, z
        
        return output, mu, logvar
    
    def getConditionDims(self):

        if self.systemType=='bistable':    
            if self.conditionedOn=="piri":
                return 6
            elif self.conditionedOn=="piririm":
                return 9
            elif self.conditionedOn=="piripim":
                return 9
            elif self.conditionedOn=="piririmpim":
                return 12
            else:
                print('Unsupported model type')
                return None

        elif self.systemType=='dimer':    
            if self.conditionedOn=="piri":
                return 12
            else:
                print('Unsupported model type')
                return None

    def getInputDims(self):

        if self.systemType=='bistable':
            return 3
        elif self.systemType=='dimer':
            return 6


class cvaeSampler_SEE(cvaeSampler):
    """
    Child model of cvaeSampler - has all the same functions and properties, but different architecture of the encoder & decoder.
    """

    def __init__(self, latentDims, loadPretrained, conditionedOn, systemType):
        print('Model type: SEE.')
        super().__init__(latentDims, loadPretrained, conditionedOn, systemType, fhl=32)

        self.encoder = nn.Sequential(
            nn.Linear(self.inputDims+self.conditionDims, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latentDims+self.conditionDims, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, self.inputDims)
        )

class cvaeSampler_SE(cvaeSampler):

    """
    Child model of cvaeSampler - has all the same functions and properties, but different architecture of the encoder & decoder.
    """

    def __init__(self, latentDims, loadPretrained, conditionedOn, systemType):
        print('Model type: SE.')
        super().__init__(latentDims, loadPretrained, conditionedOn, systemType, fhl=32)
        self.encoder = nn.Sequential(
            nn.Linear(self.inputDims+self.conditionDims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latentDims+self.conditionDims, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.inputDims)
        )


class cvaeSampler_S(cvaeSampler):

    """
    Child model of cvaeSampler - has all the same functions and properties, but different architecture of the encoder & decoder.
    """

    def __init__(self, latentDims, loadPretrained, conditionedOn, systemType):
        print('Model type: S.')
        super().__init__(latentDims, loadPretrained, conditionedOn, systemType, fhl=32)
        self.encoder = nn.Sequential(
            nn.Linear(self.inputDims+self.conditionDims, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latentDims+self.conditionDims, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.inputDims)
        )


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