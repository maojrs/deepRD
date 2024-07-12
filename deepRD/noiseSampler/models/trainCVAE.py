import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
from deepRD.noiseSampler import cvaeSampler
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity


# Model Settings
conditionedOn = 'piri'
latentDims = 4

# Provide path to model weights file or None to load untrained model.
localModelDirectory = 'deepRD/noiseSampler/models/modelWeights/'
loadPretrained = localModelDirectory + 'model_state_' + conditionedOn + '_new.pt'
outputModelDirectory = localModelDirectory + 'model_state_' + conditionedOn + '_new.pt'


# Script Settings
trainModel = True
plotReconGen = True
plotLatentGen = True
plotKDE = True

# Training Settings
n_datasets = 10
num_epochs = 20
learning_rate = 1e-4
batch_size = 32
beta1 = 1
beta2 = 1
# beta2: 2e-5 is okay, but converges to a point in latent space for long enough training. 
# 1e-5: slightly overfits reconstruction, very slow to train a good latent space representation.

# Plot Settings
n_points = 30000
plotDirectory = 'deepRD/noiseSampler/models/modelPlots/'

# Defining loss functions
loss_1 = nn.MSELoss()
def loss_2(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim = 1), dim = 0).sum()


print(f'Loading training data ({n_datasets} datasets)...')
localDirectory = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5/benchmark/"
# Take random simulation files for training
fnums = np.random.choice(2500, n_datasets)
dataset = None

for f_num in fnums:
    try:
        ds = torch.Tensor(trajectoryTools.loadTrajectory(localDirectory + "simMoriZwanzig_", f_num)).unsqueeze(0)
    except FileNotFoundError:
        print(f'File {f_num} not available.')
        continue
              
    if dataset is None:
        dataset = ds
    else:
        dataset = torch.cat((dataset, ds), dim=0)

print('Dataset loaded.')

# Number of datasets for training
n_datasets = dataset.shape[0]

# Extract desired vectors from dataset
r_aux = dataset[:, :, -3:] 
r_nxt = torch.roll(r_aux, -1, 1)
v = dataset[:, :, 4:7]
r_v = torch.cat((r_aux, v), dim = 2)

# Cut out last datapoint for consistency
r_aux = r_aux[:, :-1].flatten(end_dim=1)
r_nxt = r_nxt[:, :-1].flatten(end_dim=1)
v = v[:, :-1].flatten(end_dim=1)
r_v = r_v[:, :-1].flatten(end_dim=1)

# Build labels tensor
if conditionedOn=="piri":
    labels = r_v

# Rebuild data for model Input: R_n+1 (3), condition: r_n, velocity_n (6)
data = torch.utils.data.TensorDataset(r_nxt, labels)

data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

VAE = cvaeSampler.cvaeSampler(latentDims, loadPretrained, conditionedOn)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
VAE = VAE.to(device)
optimizer = torch.optim.Adam(VAE.parameters(),
                             lr = learning_rate,
                             weight_decay = 1e-8)

def trainingLoop(epochs):

    losses = []
    r_norms = None
    r_new = None
    for epoch in range(epochs):
        VAE.train()
        r_epoch = None
        for (image, label) in data_loader:

            # Feed through the network
            reconstruction, mu, logvar = VAE(image, label)
            # Calculate loss function
            l1 = loss_1(reconstruction, image)
            l2 = loss_2(mu, logvar)
            loss = beta1*l1 + beta2*l2
            losses.append([loss.item(), l1.item(), l2.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if r_epoch is None:
                    r_epoch = reconstruction
                else:
                    r_epoch = torch.cat( (r_epoch, reconstruction), dim=0)
                
                if epoch==epochs-1:
                    norms = torch.cat( (torch.norm(image, dim=1).unsqueeze(0), torch.norm(reconstruction, dim=1).unsqueeze(0)) , dim=0)
                    if r_norms is None:
                        r_norms = norms
                    else:
                        r_norms = torch.cat((r_norms, norms), dim=1)
                    
        if r_new is None:
            r_new = r_epoch.unsqueeze(0)
        else:
            r_new = torch.cat( (r_new, r_epoch.unsqueeze(0)), dim=0) 
                
        print(f'E{epoch+1}:',round(losses[-1][0]*1000,4), round(losses[-1][1]*1000*beta1,4), round(losses[-1][2]*1000*beta2,4))
    return r_new 

def plotDistributionsRG(n_points):

    print('Plotting reconstructed and generated distributions..')
    
    inds = np.random.choice(len(data), n_points)
    fig = plt.figure(figsize=(12,10))
    #print(r_new[-1, :, 0].shape, torch.flatten(r_aux, end_dim=1)[:, 0].shape)

    ax1 = fig.add_subplot(2,3,1)
    ax1.scatter(r_aux[inds, 0], r_nxt[inds,0], s=0.1)
    ax1.set_xlabel('r_n')
    ax1.set_ylabel('r_n+1')
    ax1.set_title('from data')

    ax2 = fig.add_subplot(2,3,2, sharey=ax1)
    ax2.scatter(r_aux[inds,0], r_new[-1, inds, 0], s=0.1)
    #ax2.scatter(torch.flatten(r_aux, end_dim=1)[:,1], r_new[-1, :, 1], s=0.1)
    ax2.set_xlabel('r_n')
    ax2.set_ylabel('r_n+1')
    ax2.set_title('from reconstruction')

    ax3 = fig.add_subplot(2,3,3, sharey=ax1)
    ax3.scatter(r_aux[inds,0], generated[inds, 0], s=0.1)
    ax3.set_xlabel('r_n')
    ax3.set_ylabel('r_n+1')
    ax3.set_title('generated')

    ax4 = fig.add_subplot(2,3,4)
    ax4.scatter(v[inds, 0], r_nxt[inds,0], s=0.1)
    ax4.set_xlabel('v_n')
    ax4.set_ylabel('r_n+1')

    ax5 = fig.add_subplot(2,3,5, sharey=ax4)
    ax5.scatter(v[inds, 0], r_new[-1, inds, 0], s=0.1)
    ax5.set_xlabel('v_n')
    ax5.set_ylabel('r_n+1')

    ax6 = fig.add_subplot(2,3,6, sharey=ax4)
    ax6.scatter(v[inds, 0], generated[inds,0], s=0.1)
    ax6.set_xlabel('v_n')
    ax6.set_ylabel('r_n+1')

    plt.savefig(plotDirectory + 'distributions_RG.png')
    plt.clf()

def plotGenerated(n_points):

    print('Plotting latent space and generated distribution..')

    inds = np.random.choice(len(data), n_points)

    fig2 = plt.figure(figsize=(12,5))

    ax1 = fig2.add_subplot(1,2,1)
    ax1.scatter(z[inds, 0], z[inds, 1], s=0.5)
    ax1.set_title('Latent space distribution X-Y')

    ax2 = fig2.add_subplot(1,2,2)
    ax2.scatter(r_aux[inds, 0], generated[inds, 0], s=0.1)
    ax2.set_title('generated')

    plt.savefig(plotDirectory + 'distributions_latent.png')
    plt.clf()


def plotKDEs():

    print('Plotting KDE..')

    r_x = r_aux[:, 0].reshape(-1, 1)
    rx_new = r_new[-1, :, 0].reshape(-1, 1)
    gen_x = generated[:, 0].reshape(-1, 1)

    # Original data
    kde1 = KernelDensity(bandwidth=0.005)
    kde1.fit(r_aux)

    # Reconstructed data
    kde2 = KernelDensity(bandwidth=0.005)
    kde2.fit(r_new[-1])

    # Generated data
    kde3 = KernelDensity(bandwidth=0.005)
    kde3.fit(generated)

    grid = np.linspace(-0.1, 0.1, 100)
    z = np.full((100,2), 0)
    grid = np.concatenate((grid[:, np.newaxis],z), axis=1)
    pdf_data = kde1.score_samples(grid)
    pdf_rec = kde2.score_samples(grid)
    pdf_gen = kde3.score_samples(grid)

    fig3 = plt.figure(figsize=(12,4))

    ax1 = fig3.add_subplot(1,3,1)
    ax1.plot(grid[:,0], pdf_data)
    ax1.set_title('KDE data')

    ax2 = fig3.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
    ax2.plot(grid[:,0], pdf_data, label='data')
    ax2.plot(grid[:,0], pdf_gen, label='generated')
    ax2.legend()
    ax2.set_title('KDE model')

    ax3 = fig3.add_subplot(1,3,3)
    ax3.plot(grid[:,0], pdf_data, label='data')
    ax3.plot(grid[:,0], pdf_gen, label='generated')
    ax3.set_title('overlap')
    ax3.legend()
    
    plt.savefig(plotDirectory + 'KDE.png')
    plt.clf()


# Generating samples
ind = np.random.choice(len(data), len(data))
labels = data[:][1]

with torch.no_grad():
    _, _, _, z = VAE(data[ind][0], data[ind][1], returnLatent=True)
    
    generated = VAE.sample(labels, num_samples=len(labels))

if trainModel:
    print(f'Training for {num_epochs} epochs...')
    r_new = trainingLoop(num_epochs)
    print('Finished training.')

if plotReconGen:
    plotDistributionsRG(n_points)

if plotLatentGen:
    plotGenerated(n_points)

if plotKDE:
    plotKDEs()

plt.show()

print('Save new model? y/n')
saveModel = input()
if saveModel=='y':
    # Save model parameters
    torch.save(VAE.state_dict(), outputModelDirectory)
    print('Model parameters saved.')