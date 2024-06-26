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

    def __init__(self, latent_dims, load_model=True):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(3+6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims+6, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.linear1 = nn.Linear(20, latent_dims)
        self.linear2 = nn.Linear(20, latent_dims)
        self.G = torch.distributions.Normal(0, 1)
        if load_model==True:
            self.load_state_dict(torch.load('deepRD/noiseSampler/models/model_state.pt'))
            print('Model parameters loaded.')

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*self.G.sample(mu.shape)

    def sample(self, label, num_samples=1):
        '''
         Here a slight workaround to get compatibility between model and integrator.
         Model is designed to work on tensors of size (n_samples, *), meanwhile 
         integrator works on 1-D arrays. This function assumes input of 1-D array of 'piri'
         (later will be extended to more general functionality)

         Returns: 1-D Torch Tensor of size (3)
        '''
        with torch.no_grad():
            label = torch.from_numpy(label).float()
            r = label[3:]
            v = label[:3]
            label = torch.cat((r,v)).unsqueeze(0)
        
            mean = 0
            std = 1

            samples = torch.normal(mean, std, (num_samples,self.latent_dims))
            z_cond = torch.cat((samples, label), dim=1)
            out = np.array(self.decoder(z_cond).squeeze(0))

        return out

    def forward(self, x, y, return_latent=False):
        x_cond = torch.cat((x,y), dim=1)
        x = self.encoder(x_cond)
        mu = self.linear1(x)
        logvar = self.linear2(x)
        z = self.reparametrize(mu, logvar)
        z_cond = torch.cat((z, y), dim=1)
        output = self.decoder(z_cond)
        if return_latent==True:
            return output, mu, logvar, z
        return output, mu, logvar