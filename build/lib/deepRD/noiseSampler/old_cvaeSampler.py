import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
import os
import math
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
    batch_norm (bool): use batch normalisation
    dropout (0-1): dropout rate, disabled if 0
    norm_params: (mean_input, std_input, mean_cond, std_cond) - model will normalize and denormalize during training or sampling
    sampling_width (float, default=1): sets the std for generating samples in latent space
    sampling_scale (float, default=1): multiplies output at the end of sampling.
    cutoff (float, default=0): cutoff radius for latent space sampling, inactive if 0
    """

    def __init__(self, latentDims, loadPretrained, conditionedOn, systemType, hidden_dims=None, batch_norm=False, dropout_rate=0, norm_params=(0,1,0,1), sampling_width=1, cutoff=False, sampling_scale=1, scaler_cond=None, scaler_inp=None):
        super().__init__()
        self.conditionedOn = conditionedOn
        self.systemType = systemType
        self.latentDims = latentDims
        self.loadPretrained = loadPretrained
        self.conditionDims = self.getConditionDims()
        self.inputDims = self.getInputDims()
        self.G = torch.distributions.Normal(0, 1)
        self.sampling_width = sampling_width
        self.sampling_scale = sampling_scale
        self.cutoff = cutoff
        self.scaler_cond = scaler_cond
        self.scaler_inp = scaler_inp
        

        if hidden_dims==None:
            # Initialising template network architecture
            fhl=20

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
            fhl = hidden_dims[-1]

            if dropout_rate>0 and batch_norm==True:
                print('Use either BN or Dropout, not both.')

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

        self.linear1 = nn.Linear(fhl, self.latentDims)
        self.linear2 = nn.Linear(fhl, self.latentDims)
        self.load_model()

        self.mean_input, self.std_input, self.mean_cond, self.std_cond = norm_params
        print('Model loaded.')


    def load_model(self):
        if self.loadPretrained==None:
            print('Untrained model initialized. Conditioned on:', self.conditionedOn)
        else:
            print('Loading pretrained model: ' + self.loadPretrained)
            self.load_state_dict(torch.load(self.loadPretrained))
            print('Model weights loaded.')

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*self.G.sample(mu.shape)

    def normalize(self, x, mean, std):
        """
        Normalizes a tensor w.r.t a given mean and std.
        """
        #print('norm')
        return (x-mean)/std

    def denormalize(self, x, mean, std):
        """
        Denormalizes a tensor w.r.t a given mean and std.
        """
        #print('denorm')
        return x*std + mean

    def sample(self, label, num_samples=1, sim_sampling=False):
        '''
         Method to generate samples given a label of conditioning variables.
         Model is trained on Torch tensors of size (num_samples, *), meanwhile 
         integrator works on 1-D arrays. 
         
         parameters:
         label (Torch Tensor or Numpy Array): conditioning variables.
         num_samples: number of samples to generate
         sim_sampling: if set to true, 

         Returns: 1-D Torch Tensor of size (input_dims)
        '''

        # For inference, sample from Gaussian distribution with the following parameters:
        mean = 0
        std = self.sampling_width
        
        if isinstance(label, np.ndarray):
            sim_sampling = True

        if sim_sampling:
            if self.scaler_cond==None:
                # Normalizing label for inference, since model was trained on normalized data.
                label = self.normalize(label, self.mean_cond, self.std_cond)
            else:
                label = self.scaler_cond.transform(label[np.newaxis, :]).flatten()

            with torch.no_grad():
                label = torch.from_numpy(label).float()
                
                if self.conditionedOn=="piri":
                    # Switching the order because models I trained with opposite order of cond. variables
                    r = label[3:]
                    v = label[:3]
                    label = torch.cat((r,v))
                        
        if label.dim()==1:
            label = label.unsqueeze(0)
            
        # Generating latent space samples
        if not self.cutoff:
            samples = torch.normal(mean, std, (num_samples, self.latentDims))
            
        else:   
            # If using cutoff radius, resample if radius is exceeded. 
            samples = torch.zeros((num_samples, self.latentDims))
            for i in range(num_samples):
                while True:
                    zi = torch.normal(mean, std, (1, self.latentDims))
                    ri =  torch.linalg.vector_norm(zi)
                    if ri < self.cutoff:
                        break
                        
                samples[i] = zi[0]

        # Decoding from latent space and generating output
        z_cond = torch.cat((samples, label), dim=1)
        out = self.decoder(z_cond).squeeze(0).detach().numpy() if sim_sampling else self.decoder(z_cond)

        if sim_sampling:   
            
            if self.scaler_inp==None:
                # Denormalizing output
                out = self.denormalize(out, self.mean_input, self.std_input)
            else:
                out = self.scaler_inp.inverse_transform(out[np.newaxis, :]).flatten()

        return self.sampling_scale*out

    def forward(self, x, y, returnLatent=False):

        # Normalizing data 
        #x = self.normalize(x, self.mean_input, self.std_input)
        #y = self.normalize(y, self.mean_cond, self.std_cond)

        # Passing through network
        x_cond = torch.cat((x,y), dim=1)
        x = self.encoder(x_cond)
        mu = self.linear1(x)
        logvar = self.linear2(x)
        z = self.reparametrize(mu, logvar)
        z_cond = torch.cat((z, y), dim=1)
        output = self.decoder(z_cond)

        # Denormalizing output
        #output = self.denormalize(output, self.mean_input, self.std_input)

        if returnLatent==True:
            return output, mu, logvar, z
        
        return output, mu, logvar
    
    def getConditionDims(self):

        if self.systemType=='bistable':    
            if self.conditionedOn=="piri":
                return 6
            elif self.conditionedOn=="piririm":
                return 9
            elif self.conditionedOn=="pipimri":
                return 9
            elif self.conditionedOn=="piririmpim":
                return 12
            else:
                print('Unsupported model type')
                return None

        elif self.systemType=='dimer':    
            if self.conditionedOn=="piri":
                return 12
            elif self.conditionedOn=="piridqi":
                return 13
            else:
                print('Unsupported model type')
                return None

    def getInputDims(self):

        if self.systemType=='bistable':
            return 3
        elif self.systemType=='dimer':
            return 6
            

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


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the value (slope) of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return


# Alternative CVAE model using scale and shift parameters..

class cvaeSamplerScale(nn.Module):

    """
    Model for sampling of the interaction noise term in reduced simulations. 
    The model is a Conditional Variational Autoencoder (CVAE) architecture.

    Parameters:
    latentDims (int): dimensionality of latent (encoded) space
    loadPretrained (str): filepath of pre-trained model to be loaded
    conditionedOn (str): ['piri', 'pipimri', etc.] - set of the variables used for conditioning of the decoding/encoding
    systemType (str): ['bistable', 'dimer'] 
    hidden_dims (1-D array of int): e.g. [128, 64, 32] ; determines the encoder/decoder structure given as subsequent hidden layers
    batch_norm (bool): use batch normalisation
    dropout (0-1): dropout rate, disabled if 0
    norm_params: (mean_input, std_input, mean_cond, std_cond) - model will normalize and denormalize during training or sampling
    sampling_width (float, default=1): sets the std for generating samples in latent space
    sampling_scale (float, default=1): multiplies output at the end of sampling.
    cutoff (float, default=0): cutoff radius for latent space sampling, inactive if 0
    """

    def __init__(self, latentDims, loadPretrained, conditionedOn, systemType, hidden_dims=None, batch_norm=False, dropout_rate=0, norm_params=(0,1,0,1), sampling_width=1, cutoff=False):
        super().__init__()
        self.conditionedOn = conditionedOn
        self.systemType = systemType
        self.latentDims = latentDims
        self.loadPretrained = loadPretrained
        self.conditionDims = self.getConditionDims()
        self.inputDims = self.getInputDims()
        self.G = torch.distributions.Normal(0, 1)
        self.sampling_width = sampling_width
        self.cutoff = cutoff

        if hidden_dims==None:
            # Initialising template network architecture
            fhl=20

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
            fhl = hidden_dims[-1]

            if dropout_rate>0 and batch_norm==True:
                print('Use either BN or Dropout, not both.')

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

        self.linear1 = nn.Linear(fhl, self.latentDims)
        self.linear2 = nn.Linear(fhl, self.latentDims)
        self.scale_shift = ScaleShift(self.inputDims)
        self.load_model()

        self.mean_input, self.std_input, self.mean_cond, self.std_cond = norm_params
        print('Model loaded.')


    def load_model(self):
        if self.loadPretrained==None:
            print('Untrained model initialized. Conditioned on:', self.conditionedOn)
        else:
            print('Loading pretrained model: ' + self.loadPretrained)
            self.load_state_dict(torch.load(self.loadPretrained))
            print('Model weights loaded.')

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*self.G.sample(mu.shape)

    def normalize(self, x, mean, std):
        """
        Normalizes a tensor w.r.t a given mean and std.
        """
        #print('norm')
        return (x-mean)/std

    def denormalize(self, x, mean, std):
        """
        Denormalizes a tensor w.r.t a given mean and std.
        """
        #print('denorm')
        return x*std + mean

    def sample(self, label, num_samples=1, sim_sampling=False):
        '''
         Method to generate samples given a label of conditioning variables.
         Model is trained on Torch tensors of size (num_samples, *), meanwhile 
         integrator works on 1-D arrays. 
         
         parameters:
         label (Torch Tensor or Numpy Array): conditioning variables.
         num_samples: number of samples to generate
         sim_sampling: if set to true, 

         Returns: 1-D Torch Tensor of size (input_dims)
        '''

        # For inference, sample from Gaussian distribution with the following parameters:
        mean = 0
        std = self.sampling_width
        
        if isinstance(label, np.ndarray):
            sim_sampling = True

        if sim_sampling:
            
            label = self.normalize(label, self.mean_cond, self.std_cond)

            with torch.no_grad():
                label = torch.from_numpy(label).float()
                
                if self.conditionedOn=="piri":
                    # Switching the order because models I trained with opposite order of cond. variables
                    r = label[3:]
                    v = label[:3]
                    label = torch.cat((r,v))
                        
        if label.dim()==1:
            label = label.unsqueeze(0)
            
        # Generating latent space samples
        if not self.cutoff:
            samples = torch.normal(mean, std, (num_samples, self.latentDims))
            
        else:   
            # If using cutoff radius, resample if radius is exceeded. 
            samples = torch.zeros((num_samples, self.latentDims))
            for i in range(num_samples):
                while True:
                    zi = torch.normal(mean, std, (1, self.latentDims))
                    ri =  torch.linalg.vector_norm(zi)
                    if ri < self.cutoff:
                        break
                        
                samples[i] = zi[0]

        # Decoding from latent space and generating output
        z_cond = torch.cat((samples, label), dim=1)

        out = self.decoder(z_cond)

        #out = self.decoder(z_cond).squeeze(0).detach().numpy() if sim_sampling else self.decoder(z_cond)
        out = self.scale_shift(out).squeeze(0).detach().numpy() if sim_sampling else self.scale_shift(out)

        return out

    def forward(self, x, y, returnLatent=False):

        # Normalizing data 
        #x = self.normalize(x, self.mean_input, self.std_input)
        #y = self.normalize(y, self.mean_cond, self.std_cond)

        # Passing through network
        x_cond = torch.cat((x,y), dim=1)
        x = self.encoder(x_cond)
        mu = self.linear1(x)
        logvar = self.linear2(x)
        z = self.reparametrize(mu, logvar)
        z_cond = torch.cat((z, y), dim=1)
        output_norm = self.decoder(z_cond)

        # Denormalizing output
        output = self.scale_shift(output_norm)

        if returnLatent==True:
            return output, mu, logvar, z
        
        return output, mu, logvar
    
    def getConditionDims(self):

        if self.systemType=='bistable':    
            if self.conditionedOn=="piri":
                return 6
            elif self.conditionedOn=="piririm":
                return 9
            elif self.conditionedOn=="pipimri":
                return 9
            elif self.conditionedOn=="piririmpim":
                return 12
            else:
                print('Unsupported model type')
                return None

        elif self.systemType=='dimer':    
            if self.conditionedOn=="piri":
                return 12
            elif self.conditionedOn=="piridqi":
                return 13
            else:
                print('Unsupported model type')
                return None

    def getInputDims(self):

        if self.systemType=='bistable':
            return 3
        elif self.systemType=='dimer':
            return 6


class ScaleShift(nn.Module):
    '''
    Class that implements the Scale-Shift transformation.
    '''
    def __init__(self, num_features):
        super(ScaleShift, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.shift

    def inverse(self, y):
        return (y - self.shift) / self.scale.clamp(min=1e-6)