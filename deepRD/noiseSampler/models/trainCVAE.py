import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
import csv
import time
from deepRD.noiseSampler import cvaeSampler
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler
from annealing import Annealer

# Use GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Default Model and Training parameters to be set:

systemType (str) - determines the dataset to be loaded - 'bistable', 'dimer', etc. 
conditionedOn (str) - set of conditioning variables. 'piri' - v_n, r_n, 'piririm' - v_n, r_n, r_n-1, etc..

latentDims (int) - number of dimensions of the latent space. 
hiddenDims (list of ints) - architecture of the Neural Network. e.g. [128,64,32] means an encoder with three subsequent layers
                             containing 128, 64, 32 neurons each, and a symmetrical decoder. 
                             [inputDim - 128 - 64 - 32 - latentDim - 32 - 64 - 128 - outputDim]

loadPretrained (str) - path to pre-trained model weights to be used for network initialisation

batch_norm, dropout_rate - [DOES NOT WORK]
normalize_data (bool) - if True, data is normalized before being fed to the network. [DOES NOT WORK WELL RIGHT NOW]

annealing_shape (str) - shape of the annealing curve - 'logistic', 'linear', 'cosine'
annealing_period (int) - annealing period

train_split (0-1) - fraction of data used for training (the rest is set aside as a test dataset)
n_datasets (int: 0-2500) - number of benchmark datasets to be loaded
num_epochs (int) - number of epochs for model training

learning_rate (float) - gradient descent learning rate
batch_size (int) - number of samples fed through the network as one batch (gradient descent is executed in batch-wise steps)
weight_decay (float) - a normalising feature [DOESN'T REALLY HELP TOO MUCH...]

Loss coefficients:
beta1 - standard MSE Loss between Input and Output. Normally kept at 1.
beta2 - KL divergence term. ranges from 0 to 1.
alpha - loss 3, in the current state is a batch mean loss [optional, normally set to 0]

modelsToTrain - dictionary containing names of models to be trained by the script with the corresponding parameters (unspecified will be set to default vals)

"""

# Model Settings
systemType = 'bistable' # 'bistable', 'dimer'
conditionedOn = 'piri' # 'piri', 'piririm', 'pipimri'
latentDims = 3
hiddenDims = [128, 64, 32] # None

# Provide path to model weights file or None to load untrained model.
localModelDirectory = 'deepRD/noiseSampler/models/modelWeights/'
loadPretrained = None # localModelDirectory + 'model_state_' + conditionedOn + '_modelName.pt'

# Use either BN OR Dropout
batch_norm = False # Data Pre-Processing AND Batch Normalisation
dropout_rate = 0 # Dropout

# Default Training Settings
normalize_data = False
annealing_shape = 'logistic'
annealing_period = 5
train_split = 0.8
n_datasets = 1500 #[40, 80, 160, 320, 640, 1260, 2500]
num_epochs = 40
learning_rate = 1e-4
batch_size = 32
weight_decay = 0

# Loss coefficients
beta1 = 1
beta2 = 1e-5
# Penalizing mean of batch deviating from 0.
alpha = 0 # set to 0 to not penalise mean

# Each model to be trained is represented by a dictionary which contains all the parameters which are to be changed.
# Parameters which are unspecified in the dictionary will be set to default values as set above.

modelsToTrain = [
    {
        'modelName': 'E81_10ep',
        'latentDims': 8,
        'n_datasets': 1500,
        'num_epochs': 10
    },
    {
        'modelName': 'D81_10ep',
        'latentDims': 8,
        'n_datasets': 2500,
        'num_epochs': 10
    },
    {
        'modelName': 'T1',
        'latentDims': 3,
        'n_datasets': 1500,
        'num_epochs': 40,
        'annealing_period': 10
    },
]

# Directory to save loss values for plotting.
plotDirectory = 'deepRD/noiseSampler/models/modelLosses/'

# Defining loss functions

# RECONSTRUCTION LOSS
loss_1 = nn.MSELoss()

# KL DIVERGENCE LOSS
def loss_2(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim = 1), dim = 0).sum()

#def loss_3(reconstruction):
#    # Try to enforce mean of batch of reconstructed samples deviating from 0
#    return torch.mean(reconstruction)**2

loss_3 = nn.MSELoss()

def compute_variance(reconstruction, image):
    """
    Penalizes the difference in variance between the input and the reconstruction.
    Assumes input and reconstruction are batched (B x D or B x C x H x W).
    """
    input_var = torch.mean(torch.var(image, dim=0, unbiased=False))
    recon_var = torch.mean(torch.var(reconstruction, dim=0, unbiased=False))
    return recon_var, input_var

def reform_dataset(dataset, systemType):
    """
    Function to reshape dataset, extracting the physical vectors depending on systemType 
    systemType: 'bistable', 'dimer'

    returns: test_data, data, (mean_input, std_input, mean_cond, std_cond)
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

    # Split data: first 20% test, remaining 80% test.
    split_ind = int(TT_split*len(inputVars))

    # Rebuild data for model Input: R_n+1 (3), condition: r_n, velocity_n (6)
    test_data = torch.utils.data.TensorDataset(inputVars[:split_ind], conditionalVars[:split_ind])
    data = torch.utils.data.TensorDataset(inputVars[split_ind:], conditionalVars[split_ind:])

    return test_data, data


def trainingLoop(data_loader, test_data_loader, epochs, csvfile, annealing_shape='logistic', annealing_period=10):

    iteration_counter = 0
    writer = csv.writer(csvfile)

    #annealing_period = 10 # annealing step done once every epoch
    annealing_agent = cvaeSampler.Annealer(annealing_period, shape=annealing_shape, baseline = 0, cyclical=True) # instantiating annealing agent

    for epoch in range(epochs):

        total_loss = 0
        total_l1 = 0
        total_l2 = 0
        total_alpha = 0
        val_loss = 0
        bc = 0

        # At the start of epoch, evaluate validation error.
        with torch.no_grad():
            VAE.eval()
            for (image_t, label_t) in test_data_loader:
                bc += 1
                reconstruction_t, mu_t, logvar_t = VAE(image_t, label_t)
                val_loss += loss_1(reconstruction_t, image_t).item() # MSE loss only on validation set

            val_loss /= bc # computing average per-batch loss
            bc=0

        VAE.train()
        for (image, label) in data_loader:
            bc += 1
            # Feed through the network
            reconstruction, mu, logvar = VAE(image, label)

            # Calculate loss function
            l1 = loss_1(reconstruction, image)
            l2 = annealing_agent(loss_2(mu, logvar))

            recon_var, image_var = compute_variance(reconstruction, image)
            l3 = loss_3(recon_var, image_var)

            loss = beta1*l1 + beta2*l2 + alpha*l3
            loss_line = [l1.item()*beta1, l2.item()*beta2, loss.item(), val_loss]

            total_loss += loss.item()
            total_l1 += l1.item()
            total_l2 += l2.item()
            total_alpha += l3.item()

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


for model in modelsToTrain:
    start=time.perf_counter()

    # Setting the variables specified in the model dictionary
    for key, val in model.items():
        print(key,' = ', val)
        exec(key + '=val')

    outputModelDirectory = localModelDirectory + 'model_state_' + conditionedOn + '_' + modelName + '.pt'
    
    print(f'Loading training data ({n_datasets} datasets)...')
    localDirectory = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5/benchmark/"
    #   Take random simulation files for training
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

    if normalize_data:
        if systemType=='bistable':
        
            mean = torch.tensor([ 4.9998e+02, -1.6760e-02,  3.3266e-03, -1.7140e-03, -1.2232e-05, 
                                 -7.2127e-05,  6.9702e-05,  1.0000e+00,  2.0080e-07,  6.3679e-06, -2.9733e-06])
            std = torch.tensor([1.4434e+02, 1.3641e+00, 7.3784e-01, 7.3927e-01, 1.4253e-01, 1.4252e-01, 
                                1.4256e-01, 0.0000e+00, 1.6200e-02, 1.6212e-02, 1.6207e-02])

            # Hard coded mean and std of the entire dataset
            mean_input = torch.tensor([2.0080e-07,  6.3679e-06, -2.9733e-06])
            std_input = torch.tensor([1.6200e-02, 1.6212e-02, 1.6207e-02])

            if conditionedOn=='piri':
                mean_cond = torch.tensor([2.0080e-07, 6.3679e-06, -2.9733e-06, -1.2232e-05, -7.2127e-05,  6.9702e-05])
                std_cond = torch.tensor([1.6200e-02, 1.6212e-02, 1.6207e-02, 1.4253e-01, 1.4252e-01, 
                            1.4256e-01])
        
        # Normalizing data
        dataset = (dataset-mean)/std

        # Robust Scaler
        #dataset = torch.flatten(dataset, end_dim=1)
        #scaler = RobustScaler().fit(dataset)
        #dataset = torch.tensor(scaler.transform(dataset))
        #dataset = torch.reshape(dataset, (n_datasets, 10000, 11)).float()

    else:
        mean_input, std_input, mean_cond, std_cond = (torch.tensor(0), torch.tensor(1), torch.tensor(0), torch.tensor(1))

    # Tuple of normalization constants to pass onto CVAE 
    norm_params = (mean_input, std_input, mean_cond, std_cond)
    test_data, data = reform_dataset(dataset, systemType)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    VAE = cvaeSampler.cvaeSampler(latentDims, loadPretrained, conditionedOn, systemType, 
                                    hidden_dims=hiddenDims, batch_norm=batch_norm, dropout_rate=dropout_rate, norm_params=norm_params)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay)

    # Writing loss file
    with open(plotDirectory + 'losses_' + conditionedOn + '_' + modelName + '.csv', 'w', newline='') as csvfile:

        print(f'Training "{modelName}" for {num_epochs} epochs...')
        trainingLoop(data_loader, test_data_loader, num_epochs, csvfile, annealing_shape, annealing_period)
        print('Finished training.')

    # Saving model parameters
    torch.save(VAE.state_dict(), outputModelDirectory)
    print('Model parameters saved.')
    print(VAE.mean_input, VAE.std_input, VAE.mean_cond, VAE.std_cond)

    end = time.perf_counter()
    training_time = end-start

    print('Training time:', int(training_time//3600), 'hours',  int(training_time%60), 'minutes', int(training_time%60), 'seconds')

    del modelName

