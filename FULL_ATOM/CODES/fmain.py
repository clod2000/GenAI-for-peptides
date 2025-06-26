import torch
import MDAnalysis as mda
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import os
import os.path as osp
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
import math
from torch.utils.tensorboard import SummaryWriter
import itertools
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import summary, VGAE
from tqdm import tqdm
import sys

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

sys.path.append("LIBS")
from LIBS.utils import *
from LIBS.FGVAE import *

import argparse 
# Create a single parser with both arguments
parser = argparse.ArgumentParser(description='Full Graph VAE with EGNN')
parser.add_argument('--config', type=str, default='config.template.in', help='Path to the configuration file')
parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode')

# Parse the command line arguments
args = parser.parse_args()
config_file = args.config
verbose = args.verbose

verbose= True  # set to debug for now, will be set to False in the future

#Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if verbose: print(f"Using device: {device}")
if verbose: print(f"Using config file: {config_file}")

# Read the parameters from the config file
try:
    config = parse_config(config_file=config_file, verbose=verbose)
    if verbose: print(f"Configuration file {config_file} read successfully.")
except Exception as e:
    print(f"Error reading config file: {e}")
    sys.exit(1)

if verbose: print(f"Configuration parameters: {config}")

#exit()


# Architecture parameters
MODEL_ARCHITECTURE = config.get('MODEL_ARCHITECTURE', 'original') # architecture of the model, can be 'original' or 'hybrid_displacement'
ENCODER_POS_PROJECTION_DIM = config.get('ENCODER_POS_PROJECTION_DIM', 64) # dimension of the position projection in the encoder, used to project the positions to a lower dimension if 'hybrid_displacement' is used
#### model parameters
# encoder
ENCODER_TYPE = config.get('ENCODER_TYPE', 'standard') # type of the encoder, can be 'standard' or TO BE DEFINED
HIDDEN_ENCODER_CHANNELS = config.get('HIDDEN_ENCODER_CHANNELS', 256)
OUT_ENCODER_CHANNELS = config.get('OUT_ENCODER_CHANNELS', 128)
NUM_ENC_LAYERS = config.get('NUM_ENC_LAYERS', 5) # number of EGNN layers in the encoder
ATTENTION_ENCODER = config.get('ATTENTION_ENCODER', True) # if True, attention is used in the encoder
LATENT_DIM = config.get('LATENT_DIM', 128) # latent dimension of the encoder, used to create the latent space
TANH_ENCODER = config.get('TANH_ENCODER', True) # if True, the output of the encoder is passed through a tanh activation function (for positions)
NORMALIZE_ENCODER = config.get('NORMALIZE_ENCODER', True) # if True, the encoder output is normalized
# decoder
MLP_DECODER_POS_SIZE = config.get('MLP_DECODER_POS_SIZE', [256,256,128]) # size of the MLP decoder for positions
HIDDEN_DECODER_CHANNELS = config.get('HIDDEN_DECODER_CHANNELS', 256)
NUM_DEC_LAYERS = config.get('NUM_DEC_LAYERS', 5)
ATTENTION_DECODER = config.get('ATTENTION_DECODER', True)
TANH_DECODER = config.get('TANH_DECODER', True) # if True, the output of the decoder is passed through a tanh activation function (for positions)
NORMALIZE_DECODER = config.get('NORMALIZE_DECODER', True) # if True, the decoder output is normalized

#### dataset parameters
INCLUDE_ATOM_TYPE = config.get('INCLUDE_ATOM_TYPE', True) # if True, the atom type is included in the dataset
SCALE_FEATURES = config.get('SCALE_FEATURES', True)
SCALE_POSITIONS = config.get('SCALE_POSITIONS', True) # if True, the positions are scaled to the range [0, 1]
INITIAL_ALIGNMENT = config.get('INITIAL_ALIGNMENT', True) # if True, the dataset is aligned to the initial positions

#### training parameters
EPOCHS = config.get('EPOCHS', 50)
BATCHSIZE = config.get('BATCHSIZE', 64)
LEARNING_RATE = config.get('LEARNING_RATE', 1E-4)
WEIGHT_DECAY = config.get('WEIGHT_DECAY', 0) # weight decay for the optimizer, set to 0 to disable weight decay ( bad idea using it for vae)

#### Scheduler parameters
USE_SCHEDULER = config.get('USE_SCHEDULER', False) # if True, the learning rate scheduler is used
SCHEDULER_PATIENCE = config.get('SCHEDULER_PATIENCE', 10) # number of epochs with no improvement after which learning rate will be reduced
SCHEDULER_FACTOR = config.get('SCHEDULER_FACTOR', 0.5) # factor by which the learning rate will be reduced. new_lr = lr * factor
SCHEDULER_TYPE = config.get('SCHEDULER_TYPE', 'ReduceLROnPlateau') # type of the scheduler, can be 'CosineAnnealingLR' or 'StepLR' or 'ReduceLROnPlateau'
SCHEDULER_THRESHOLD = config.get('SCHEDULER_THRESHOLD', 0.0001) # threshold for the scheduler, used to stop the training if the loss is below this value

# Beta annealing parameters
BETA = config.get('BETA', None)
wait_epochs = config.get('wait_epochs', 0)
annealing_epochs = config.get('annealing_epochs', 50)
beta_min = config.get('beta_min', 0.00001)
beta_max = config.get('beta_max', 0.0001)

# Other parameters
DISABLE_TQDM = config.get('DISABLE_TQDM', False) # if True, the tqdm progress bar is disabled
SEED = config.get('SEED', 42) # seed for reproducibility
NAME_SIMULATION = config.get('NAME_SIMULATION', None ) # name of the simulation, used to create a folder to save the model
NAME_FOLDER = config.get('NAME_FOLDER', 'template') # name of the folder to save the model, if None, the folder will be created with the name of the model architecture and encoder type
CONTINUE_FROM = config.get('continue_from', None) # set the path to the model to continue training from, if None, the training will start from scratch
STARTING_EPOCH = config.get('starting_epoch', 0) # if CONTINUE_FROM is not None, the training will start from this epoch
ALIGN_RECONS_LOSS = config.get('ALIGN_RECONS_LOSS', True) # if True, samples are aligned before computing the reconstruction loss, otherwise the reconstruction loss is computed without alignment
TEST_MODEL = config.get('TEST_MODEL', False) # if True, the model is tested after training
MIN_KL = config.get('MIN_KL', 0.0001) # minimum value for the KL divergence loss, if the KL divergence is below this value, the total loss is set to the reconstruction loss only


if CONTINUE_FROM is None: 
    STARTING_EPOCH = 0 # if CONTINUE_FROM is None, the training will start from epoch 0
    if verbose: print("Starting training from scratch")
else:
    if verbose: print(f"Continuing training from {CONTINUE_FROM} at epoch {STARTING_EPOCH}")
    if not os.path.exists(CONTINUE_FROM):
        raise FileNotFoundError(f"The path {CONTINUE_FROM} does not exist. Please check the path and try again.")

# Set the random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create dataset and dataloaders
dataset = get_dataset(
    include_atom_type=INCLUDE_ATOM_TYPE,
    scale_features=SCALE_FEATURES,
    scale_pos=SCALE_POSITIONS,
    initial_alignment=INITIAL_ALIGNMENT,
    verbose=verbose
)
train_loader, val_loader, test_loader = get_dataloaders(
    dataset=dataset,
    shuffle=True,
    seed=SEED,
    batch_size=BATCHSIZE,
    verbose=verbose
)

# Calculate the mean structure if needed 
pos_ref = None
if MODEL_ARCHITECTURE == 'hybrid_displacement':
    if verbose: print("Calculating mean reference structure for the hybrid model...")
    
    # Use the dataset directly instead of the DataLoader to avoid batching complications
    all_pos = torch.stack([data.pos for data in dataset], dim=0)  # Shape: (num_graphs, num_atoms, 3)
    pos_ref = all_pos.mean(dim=0).to(device)  # Shape: (num_atoms, 3)
    
    if verbose: print(f"Reference structure created with shape: {pos_ref.shape}")
    if verbose: print(f"Single graph reference shape: {pos_ref.shape}")
    if verbose: print(f"Used {len(dataset)} graphs to calculate mean structure")
    if verbose: print()


# Create the model
model = FGVAE(
        encoder=EGNN_Encoder(
            in_channels=dataset[0].num_features,
            hidden_channels_egnn=HIDDEN_ENCODER_CHANNELS,
            out_channels_egnn=OUT_ENCODER_CHANNELS,
            num_egnn_layers=NUM_ENC_LAYERS,
            latent_dim=LATENT_DIM,
            attention=ATTENTION_ENCODER,
            architecture=MODEL_ARCHITECTURE,
            pos_projection_dim=ENCODER_POS_PROJECTION_DIM,
            tanh=TANH_ENCODER,
            normalize=NORMALIZE_ENCODER,
            verbose=verbose
        ),
        decoder=EGNN_Decoder(
            latent_dim=LATENT_DIM,
            node_feature_dim_initial=dataset[0].num_features,
            hidden_nf=HIDDEN_DECODER_CHANNELS,
            num_egnn_layers=NUM_DEC_LAYERS,
            attention=ATTENTION_DECODER,
            architecture=MODEL_ARCHITECTURE,
            pos_MLP_size=MLP_DECODER_POS_SIZE,
            tanh=TANH_DECODER,
            normalize=NORMALIZE_DECODER,

            verbose=verbose
        )
    ).to(device)

#if verbose: print_model_summary(model)

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Create the scheduler
if USE_SCHEDULER:
    if SCHEDULER_TYPE == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) # never used, could need  other implementation
    elif SCHEDULER_TYPE == 'StepLR':
        scheduler = StepLR(optimizer, step_size=SCHEDULER_PATIENCE, gamma=SCHEDULER_FACTOR)
    elif SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, threshold=SCHEDULER_THRESHOLD, min_lr=1e-6)
    else: 
        raise ValueError(f"Unknown scheduler type: {SCHEDULER_TYPE}")

# Define the path to save the models, use numbers to define new simulations 
if NAME_SIMULATION is not None:
    file_path = f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/{NAME_SIMULATION}/'
    if os.path.exists(file_path):
        print(f"Directory {file_path} already exists. Adding # to the name.")
        # If the directory already exists, append a number to the name
        i = 0
        while os.path.exists(file_path):
            i += 1
            file_path = f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/{NAME_SIMULATION}_{i}/'
        print(f"Creating directory {file_path}")
        os.makedirs(file_path)
else:
    i = 1
    while True:
        if not os.path.exists(f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/simulation_{i}'):
            print(f"Creating directory for simulation {i}")
            os.makedirs(f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/simulation_{i}')
            file_path = f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/simulation_{i}/'
            break
        i += 1

if not os.path.exists(file_path):
    print(f"Creating directory {file_path}")
    os.makedirs(file_path)
# Copy the config file to the model folder
os.system(f'cp {config_file} {file_path}')



# Create a SummaryWriter to log the training process
writer = SummaryWriter(log_dir=file_path)   


# Training loop

lr = LEARNING_RATE

for epoch in range(STARTING_EPOCH, STARTING_EPOCH +EPOCHS):

    if USE_SCHEDULER:
        if lr > scheduler.get_last_lr()[0]:
            print(f"Adjusting learning rate from {lr} to {scheduler.get_last_lr()[0]}")
            lr = scheduler.get_last_lr()[0]

    if BETA is not None:
        beta = BETA
    else:
        beta = beta_annealer(epoch,beta_min, beta_max, annealing_epochs,wait_epochs )
       
    train_pbar = tqdm(train_loader, disable= DISABLE_TQDM,desc=f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} [Train]", leave=False)

    # Training
    model.train()
    train_loss = 0
    train_kl_loss = 0
    train_recon_loss = 0

    for data in train_pbar:

        data = data.to(device)
        optimizer.zero_grad()



        pos_pred, mean, log_var, batch_vec = model(data, pos_ref=pos_ref)

        kl_loss = KL_divergence(mean, log_var)
        recon_loss = reconstruction_loss(pos_pred,data.pos,data.batch, align = ALIGN_RECONS_LOSS)

        # kl loss is Nan 
        if kl_loss.isnan().any() or recon_loss.isnan().any():
            print(f"Warning: kl_loss or recon_loss is NaN. Aborting the code.")
            sys.exit(1)

        if kl_loss < MIN_KL:  # set a minimum value for the kl_divergence
            total_loss = recon_loss
        else:
            total_loss = recon_loss + beta*kl_loss
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_loss.backward()
        optimizer.step()

        if total_loss.item() > 1000:
            print(f"Warning: total_loss is too high: {total_loss.item()}. Skipping this element for loss computation.")
            # If the loss is too high, skip this element for loss computation     
        else:
            train_loss += total_loss.item()
            train_kl_loss += kl_loss.item()
            train_recon_loss += recon_loss.item()

        # Update the progress bar
        train_pbar.set_postfix(
            loss=total_loss.item(),
            recon_loss=recon_loss.item(),
            kl_loss=kl_loss.item(),
        )
        train_pbar.update(1)
    # Close the progress bar
    train_pbar.close()
  
    train_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, KL Loss: {train_kl_loss:.4f}, Recon Loss: {train_recon_loss:.4f}")
    
    if USE_SCHEDULER: #and epoch > annealing_epochs:
        scheduler.step(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:

            data = data.to(device)
            pos_pred, mean, log_var, batch_vec = model(data, pos_ref=pos_ref)
            kl_loss = KL_divergence(mean, log_var)
            recon_loss = reconstruction_loss(pos_pred,data.pos,data.batch, align = ALIGN_RECONS_LOSS)

        total_loss = recon_loss + beta * kl_loss
        val_loss += total_loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

    # Log the results in TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Loss/KL', train_kl_loss, epoch)
    writer.add_scalar('Loss/Reconstruction', train_recon_loss, epoch)
    writer.add_scalar('Beta', beta, epoch)
    if USE_SCHEDULER:
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    else:
        writer.add_scalar('Learning Rate', LEARNING_RATE, epoch)

    # save the model every 30 epochs
    if (epoch + 1) % 30 == 0 or epoch == EPOCHS - 1:
        torch.save(model.state_dict(), file_path + f'model_epoch_{epoch+1}.pth')
        if verbose: print(f"Model saved to {file_path}model_epoch_{epoch+1}.pth")


    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:

        if TEST_MODEL:
            #################################### Test the model ########################################
            print()
            print("Testing the model...", end=' ')
            #print("Using the initial position of the sample as the initial position of the decoder...")

            model.eval()

            pred_pos_list = []
            true_pos_list = []
            recon_loss_list = []

            for data in test_loader:
                data = data.to(device)
                with torch.no_grad():

                    pos_pred, mean, log_var, batch_vec = model(data, pos_ref=pos_ref)
                    kl_loss = KL_divergence(mean, log_var)
                    recon_loss = reconstruction_loss(pos_pred, data.pos, data.batch, align=ALIGN_RECONS_LOSS)
                    total_loss = recon_loss + beta * kl_loss

                    pred_pos_list.append(pos_pred.detach().cpu().numpy())
                    true_pos_list.append(data.pos.detach().cpu().numpy())
                    recon_loss_list.append(recon_loss.item())

            # Plot the loss
            plt.figure(figsize=(10, 5))
            plt.plot(recon_loss_list, label='Reconstruction Loss')
            plt.hlines(y=np.mean(recon_loss_list), xmin=0, xmax=len(recon_loss_list), color='r', linestyle='--', label='Mean Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Reconstruction Loss for Test Set')
            plt.legend()

            file_path = file_path if file_path.endswith('/') else file_path + '/'
            plt.savefig(file_path + 'recon_loss.png')

            # Extract the best reconstruction loss and corresponding predictions
            best_recon_index = np.argmin(recon_loss_list)
        
            best_coords_pred = pred_pos_list[best_recon_index]
            best_coords_true = true_pos_list[best_recon_index]
            best_coords_pred_t = torch.tensor(best_coords_pred, dtype=torch.float32).to(device)
            best_coords_true_t = torch.tensor(best_coords_true, dtype=torch.float32).to(device)

            #aligned_mse = calculate_aligned_mse_loss(best_coords_pred_t, best_coords_true_t, data.batch).item()

            R,T = find_rigid_alignment(best_coords_pred_t, best_coords_true_t)
            #print(f"Rigid Transform R: {R},\n T: {T}")

            aligned_pred_t = (R @ best_coords_pred_t.T).T + T
            aligned_pred = aligned_pred_t.detach().cpu().numpy()

            # Calculate the aligned MSE loss
            aligned_mse = np.mean((aligned_pred - best_coords_true)**2)

            # Plot the aligned predictions
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(best_coords_true[:, 0], best_coords_true[:, 1], best_coords_true[:, 2], c='b', label='True Coordinates', alpha=0.5)
            ax.scatter(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2], c='r', label='Aligned Predicted Coordinates', alpha=0.5)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.text2D(0.05, 0.95, f'(aligned) MSE: {np.round(float(aligned_mse.item()),4)}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.set_title('True vs Aligned Predicted Coordinates for Best Reconstruction')
            plt.legend()
            # Save the figure
            if not os.path.exists(file_path + 'test_reconstruction'):
                os.makedirs(file_path + 'test_reconstruction')  
            if not file_path.endswith('/'):
                file_path += '/'    
            plt.savefig(file_path + f'test_reconstruction/aligned_best_reconstruction_{epoch+1}_epochs.png')

            print("Done!")

