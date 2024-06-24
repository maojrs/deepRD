import numpy
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader


class cvaeSampler(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()
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

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*self.G.sample(mu.shape)

    def sample(self, label, mean, std, num_samples=1):
        #mean = [0.9080, 0.3038]
        #std = [0.4723, 0.6708]
        x_g = torch.normal(mean[0], std[0], (num_samples,1))
        y_g = torch.normal(mean[1], std[1], (num_samples,1))
        z = torch.cat( (x_g, y_g), dim=1 )
        z_cond = torch.cat((z, label), dim=1)
        return self.decoder(z_cond)

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