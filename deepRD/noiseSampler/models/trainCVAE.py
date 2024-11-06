import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
import csv
from deepRD.noiseSampler import cvaeSampler
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity

# Model Settings
conditionedOn = 'piririm' # 'piri', 'piririm'
latentDims = 2
outputModelName = 'R1'

# Training Settings
train_split = 0.8
n_datasets = 80
num_epochs = 1000
learning_rate = 1e-4
batch_size = 32
beta1 = 10
beta2 = 2e-5
weight_decay = 1e-8

# Provide path to model weights file or None to load untrained model.
localModelDirectory = 'deepRD/noiseSampler/models/modelWeights/'
#loadPretrained = localModelDirectory + 'model_state_' + conditionedOn + '_M2.pt'
loadPretrained = None
outputModelDirectory = localModelDirectory + 'model_state_' + conditionedOn + '_' + outputModelName + '.pt'


# Plot Settings
n_points = 30000
plotDirectory = 'deepRD/noiseSampler/models/modelLosses/'

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
r_prev = torch.roll(r_aux, 1, 1)
v = dataset[:, :, 4:7]
v_prev = torch.roll(v, 1, 1)

# Cut out first & last datapoint for consistency
r_aux = r_aux[:, 1:-1].flatten(end_dim=1)
r_nxt = r_nxt[:, 1:-1].flatten(end_dim=1)
r_prev = r_prev[:, 1:-1].flatten(end_dim=1)
v = v[:, 1:-1].flatten(end_dim=1)
v_prev = v_prev[:, 1:-1].flatten(end_dim=1)


# Split data: first 20% test, remaining 80% test.
split_ind = int(0.2*len(r_aux))

# Build labels tensor
if conditionedOn=="piri":
    conditionalVars = torch.cat((r_aux, v), dim = 1)
elif conditionedOn=="piririm":
    conditionalVars = torch.cat((v, r_aux, r_prev), dim = 1)
else:
    print('Invalid model type.')

# Rebuild data for model Input: R_n+1 (3), condition: r_n, velocity_n (6)
test_data = torch.utils.data.TensorDataset(r_nxt[:split_ind], conditionalVars[:split_ind])
data = torch.utils.data.TensorDataset(r_nxt[split_ind:], conditionalVars[split_ind:])

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

VAE = cvaeSampler.cvaeSampler_64(latentDims, loadPretrained, conditionedOn)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
VAE = VAE.to(device)
optimizer = torch.optim.Adam(VAE.parameters(),
                             lr = learning_rate,
                             weight_decay = weight_decay)

def trainingLoop(epochs, csvfile):

    iteration_counter = 0
    writer = csv.writer(csvfile)

    for epoch in range(epochs):
        VAE.train()

        # At the start of epoch, evaluate validation error.
        val_loss = 0
        for (image_t, label_t) in test_data_loader:
            reconstruction_t, mu_t, logvar_t = VAE(image_t, label_t)
            val_loss += loss_1(reconstruction_t, image_t).item() # MSE loss only on validation set

        for (image, label) in data_loader:

            # Feed through the network
            reconstruction, mu, logvar = VAE(image, label)
            # Calculate loss function
            l1 = loss_1(reconstruction, image)
            l2 = loss_2(mu, logvar)
            loss = beta1*l1 + beta2*l2
            loss_line = [loss.item(), l1.item()*beta1, l2.item()*beta2, val_loss]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration_counter%1000==0:
                # Saving loss every one thousand iterations to save space :)
                writer.writerow(loss_line)
                iteration_counter=0

            iteration_counter += 1
                
        print(f'E{epoch+1}:',round(loss_line[0]*1000,4), round(loss_line[1]*1000,4), round(loss_line[2]*1000,4), round(loss_line[3]*1000,4))
    return None

# Writing loss file
with open(plotDirectory + 'losses_' + conditionedOn + '_' + outputModelName + '.csv', 'w', newline='') as csvfile:

    print(f'Training "{outputModelName}" for {num_epochs} epochs...')
    trainingLoop(num_epochs, csvfile)
    print('Finished training.')

# Saving model parameters
torch.save(VAE.state_dict(), outputModelDirectory)
print('Model parameters saved.')