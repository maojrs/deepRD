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
from annealing import Annealer

# Use GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Settings
systemType = 'bistable' # 'bistable', 'dimer'
conditionedOn = 'piri' # 'piri', 'piririm', 'pipimri'
latentDims = 3
outputModelNames = ['T1']
#hiddenDims = None
hiddenDims = [128, 64, 32]

# Training Settings
normalize_data = False
train_split = 0.8
n_datasets = 1500
num_epochs = 40
learning_rate = 1e-4
batch_size = 32
beta1 = 1
beta2 = 1e-5
weight_decay = 0

# Use either BN OR Dropout
batch_norm = False # Data Pre-Processing AND Batch Normalisation
dropout_rate = 0 # Dropout

# Penalizing mean of batch deviating from 0.
alpha = 0 # set to 0 to not penalise mean

# Provide path to model weights file or None to load untrained model.
localModelDirectory = 'deepRD/noiseSampler/models/modelWeights/'
#loadPretrained = localModelDirectory + 'model_state_' + conditionedOn + '_EE61.pt'
loadPretrained = None
outputModelDirectories = [localModelDirectory + 'model_state_' + conditionedOn + '_' + outputModelName + '.pt' for outputModelName in outputModelNames]


# Plot Settings
n_points = 30000
plotDirectory = 'deepRD/noiseSampler/models/modelLosses/'

# Defining loss functions
loss_1 = nn.MSELoss()
def loss_2(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim = 1), dim = 0).sum()

def reform_dataset(dataset, systemType, normalize_data):
    """
    Function to reshape dataset, extracting the physical vectors depending on systemType
    systemType: 'bistable', 'dimer'

    returns: test_data, data
    (two tensors to be fed to DataLoader)
    """
    TT_split = 0.2 # set value for test-train split

    # Number of datasets for training
    n_datasets = dataset.shape[0]

    if systemType=='bistable':

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
        split_ind = int(TT_split*len(r_aux))

        # Build labels tensor
        if conditionedOn=="piri":
            conditionalVars = torch.cat((r_aux, v), dim = 1)
        elif conditionedOn=="piririm":
            conditionalVars = torch.cat((v, r_aux, r_prev), dim = 1)
        elif conditionedOn=="pipimri":
            conditionalVars = torch.cat((v, v_prev, r_aux), dim = 1)
        else:
            print('Invalid model type.')

        inputVars = r_nxt


    elif systemType=='dimer':
        # Extracting desired vectors from dataset

        q1 = dataset[:, ::2, 1:4] # q_n
        r_aux1 = dataset[:, ::2, -3:] # r_n
        r_nxt1 = torch.roll(r_aux1, -1, 1) # r_n+1
        r_prev1 = torch.roll(r_aux1, 1, 1) # r_n-1
        v1 = dataset[:, ::2, 4:7] # v_n
        v_prev1 = torch.roll(v1, 1, 1) # v_n-1

        q2 = dataset[:, 1::2, 1:4] # q_n
        r_aux2 = dataset[:, 1::2, -3:] # r_n
        r_nxt2 = torch.roll(r_aux2, -1, 1) # r_n+1
        r_prev2 = torch.roll(r_aux2, 1, 1) # r_n-1
        v2 = dataset[:, 1::2, 4:7] # v_n
        v_prev2 = torch.roll(v2, 1, 1) # v_n-1

        # Cut out first & last datapoint for consistency
        q1 = q1[:, 1:-1].flatten(end_dim=1)
        r_aux1 = r_aux1[:, 1:-1].flatten(end_dim=1)
        r_nxt1 = r_nxt1[:, 1:-1].flatten(end_dim=1)
        r_prev1 = r_prev1[:, 1:-1].flatten(end_dim=1)
        v1 = v1[:, 1:-1].flatten(end_dim=1)
        v_prev1 = v_prev1[:, 1:-1].flatten(end_dim=1)

        q2 = q2[:, 1:-1].flatten(end_dim=1)
        r_aux2 = r_aux2[:, 1:-1].flatten(end_dim=1)
        r_nxt2 = r_nxt2[:, 1:-1].flatten(end_dim=1)
        r_prev2 = r_prev2[:, 1:-1].flatten(end_dim=1)
        v2 = v2[:, 1:-1].flatten(end_dim=1)
        v_prev2 = v_prev2[:, 1:-1].flatten(end_dim=1)

        # Computing relative distance
        dq = np.zeros(q1.shape)
        for i in range(q1.shape[0]):
            dq[i] = trajectoryTools.relativePosition(q1[i], q2[i], 'periodic', 5)
        
        dq = torch.from_numpy(np.linalg.norm(dq, axis=1)).unsqueeze(1).float()

        if conditionedOn == 'piri':
            conditionalVars = torch.cat((v1, v2, r_aux1, r_aux2), dim = 1) # pi1, pi2, ri1, ri2
        elif conditionedOn == 'piridqi':
            conditionalVars = torch.cat((v1, v2, r_aux1, r_aux2, dq), dim = 1) # pi1, pi2, ri1, ri2, dq
        else:
            print('Invalid model type')

        inputVars = torch.cat( (r_nxt1, r_nxt2), dim = 1)

    # Normalizing whole data to mean 0, std 1.
    if normalize_data:
        #mean_input, std_input = torch.mean(inputVars, dim=0), torch.std(inputVars, dim=0)
        #mean_cond, std_cond = torch.mean(conditionalVars, dim=0), torch.std(conditionalVars, dim=0)

        # Means of the distributions are on the order of e-6, so for simplicity I set 0.
        mean_input = torch.tensor(0)
        std_input = torch.tensor([0.0162, 0.0162, 0.0162])
        mean_cond = torch.tensor(0)
        if conditionedOn=='piri':
            std_cond = torch.tensor([0.0162, 0.0162, 0.0162, 0.1425, 0.1425, 0.1426])
        
    else:
        mean_input, std_input, mean_cond, std_cond = (torch.tensor(0), torch.tensor(1), torch.tensor(0), torch.tensor(1))

    mean_input, std_input, mean_cond, std_cond = mean_input.to(device), std_input.to(device), mean_cond.to(device), std_cond.to(device)

    # Split data: first 20% test, remaining 80% test.
    split_ind = int(TT_split*len(inputVars))

    # Rebuild data for model Input: R_n+1 (3), condition: r_n, velocity_n (6)
    test_data = torch.utils.data.TensorDataset(inputVars[:split_ind], conditionalVars[:split_ind])
    data = torch.utils.data.TensorDataset(inputVars[split_ind:], conditionalVars[split_ind:])

    return test_data, data, (mean_input, std_input, mean_cond, std_cond)



print(f'Loading training data ({n_datasets} datasets)...')
localDirectory = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5/benchmark/"
# Take random simulation files for training
fnums = np.random.choice(2500, n_datasets, replace=False) # sample datasets without duplicates
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

test_data, data, norm_params = reform_dataset(dataset, systemType, normalize_data)

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def trainingLoop(epochs, csvfile):

    iteration_counter = 0
    writer = csv.writer(csvfile)

    annealing_period = 10 # annealing step done once every epoch
    annealing_agent = cvaeSampler.Annealer(annealing_period, shape='logistic', baseline = 0, cyclical=True) # instantiating annealing agent

    for epoch in range(epochs):
        VAE.train()

        total_loss = 0
        total_l1 = 0
        total_l2 = 0
        total_alpha = 0
        val_loss = 0
        bc = 0

        # At the start of epoch, evaluate validation error.
        for (image_t, label_t) in test_data_loader:
            bc += 1
            reconstruction_t, mu_t, logvar_t = VAE(image_t, label_t)
            val_loss += loss_1(reconstruction_t, image_t).item() # MSE loss only on validation set

        val_loss /= bc # computing average per-batch loss
        bc=0

        for (image, label) in data_loader:
            bc += 1
            # Feed through the network
            reconstruction, mu, logvar = VAE(image, label)
            # Calculate loss function
            l1 = loss_1(reconstruction, image)
            l2 = loss_2(mu, logvar)
            l2 = annealing_agent(l2)

            # Try to enforce mean of batch of reconstructed samples deviating from 0
            batch_mean = torch.mean(reconstruction)
            mean_penalty = batch_mean**2
            
            loss = beta1*l1 + beta2*l2 + alpha*mean_penalty
            loss_line = [l1.item()*beta1, l2.item()*beta2, loss.item(), val_loss]

            total_loss += loss.item()
            total_l1 += l1.item()
            total_l2 += l2.item()
            total_alpha += mean_penalty.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration_counter%1000==0:
                # Saving loss every one thousand iterations to save space :)
                #writer.writerow(loss_line)
                iteration_counter=0

            iteration_counter += 1

        annealing_agent.step()
        loss_avg = [total_l1/bc, total_l2*beta2/bc, total_loss/bc, val_loss]
        alpha_avg = total_alpha/bc
        writer.writerow(loss_avg)

        print(f'E{epoch+1}: {round(loss_avg[0]*1000,4)}, {round(loss_avg[1]*1000,4)}, {round(loss_avg[2]*1000,4)}, {round(loss_avg[3]*1000,4)}')
    return None


for outputModelName, outputModelDirectory in zip(outputModelNames, outputModelDirectories):
    VAE = cvaeSampler.cvaeSampler(latentDims, loadPretrained, conditionedOn, systemType, 
                                    hidden_dims=hiddenDims, batch_norm=batch_norm, dropout_rate=dropout_rate, norm_params=norm_params)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay)

    # Writing loss file
    with open(plotDirectory + 'losses_' + conditionedOn + '_' + outputModelName + '.csv', 'w', newline='') as csvfile:

        print(f'Training "{outputModelName}" for {num_epochs} epochs...')
        trainingLoop(num_epochs, csvfile)
        print('Finished training.')

    # Saving model parameters
    torch.save(VAE.state_dict(), outputModelDirectory)
    print('Model parameters saved.')
    print(VAE.mean_input, VAE.std_input, VAE.mean_cond, VAE.std_cond)