import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
import os
#from torchvision.transforms import ToTensor
from torch import nn
#from torch.utils.data import DataLoader


class cvaeSampler(nn.Module):

    def __init__(self, latentDims, loadPretrained, conditionedOn):
        super().__init__()
        self.conditionedOn = conditionedOn
        self.latentDims = latentDims
        self.conditionDims = self.getConditionDims()
        self.encoder = nn.Sequential(
            nn.Linear(3+self.conditionDims, 128),
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
            nn.Linear(128, 3)
        )
        self.linear1 = nn.Linear(20, self.latentDims)
        self.linear2 = nn.Linear(20, self.latentDims)
        self.G = torch.distributions.Normal(0, 1)

        if loadPretrained==None:
            print('Untrained model initialized.')
        else:
            print('Loading pretrained model: ' + self.conditionedOn)
            self.load_state_dict(torch.load(loadPretrained))
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
        std = 1

        if isinstance(label, np.ndarray):

            with torch.no_grad():
                label = torch.from_numpy(label).float()

                if self.conditionedOn=="piri":
                    r = label[3:]
                    v = label[:3]

                label = torch.cat((r,v)).unsqueeze(0)

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
        
        if self.conditionedOn=="piri":
            return 6