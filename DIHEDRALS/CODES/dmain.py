#########################################################################################
#                                                                                       #
#  Main file for the dihedral generation and reconstruction with VGAE                   #
#                                                                                       #
#  Author: Claudio Colturi                                                              #
#                                                                                       #
#########################################################################################


debug = False # set to True to print debug information

from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn.pool import global_mean_pool
import math

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm


import itertools

from sklearn.preprocessing import StandardScaler

import os
import sys

import argparse

sys.path.append('LIBS')

from LIBS.DVGAE_2 import *
from LIBS.dutils import *


# Create a single parser with both arguments
parser = argparse.ArgumentParser(description='Dihedral Graph VAE with EGNN')
parser.add_argument('--config', type=str, default='config.template.in', help='Path to the configuration file')
parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose mode')


# Parse the command line arguments
args = parser.parse_args()
config_file = args.config
verbose = args.verbose

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


# extra parameters

FREEZE_ENCODER = False # if True, the encoder parameters are frozen and only the decoder is trained

#### model parameters
# encoder
ENCODER_TYPE = config.get('ENCODER_TYPE', 'SAGE') # type of the encoder, can be GCN or SAGE
HIDDEN_ENCODER_CHANNELS = config.get('HIDDEN_ENCODER_CHANNELS', 256)
OUT_ENCODER_CHANNELS = config.get('OUT_ENCODER_CHANNELS', 128)
NUM_ENC_LAYERS = config.get('NUM_ENC_LAYERS', 5) # number of EGNN layers in the encoder
ATTENTION_ENCODER = config.get('ATTENTION_ENCODER', True) # if True, attention is used in the encoder
HEADS_ENCODER = config.get('HEADS_ENCODER', 2) # number of attention heads in the encoder
LATENT_DIM = config.get('LATENT_DIM', 128) # latent dimension of the encoder, used to create the latent space

HIDDEN_DECODER_CHANNELS = config.get('HIDDEN_DECODER_CHANNELS', 256)
NUM_DEC_LAYERS = config.get('NUM_DEC_LAYERS', 5)

#### training parameters
EPOCHS = config.get('EPOCHS', 50)
BATCHSIZE = config.get('BATCHSIZE', 64)
LEARNING_RATE = config.get('LEARNING_RATE', 1E-4)
WARMUP_EPOCHS = config.get('WARMUP_EPOCHS', 0) # number of epochs for the warm-up phase, if 0, no warm-up is applied
WEIGHT_DECAY = config.get('WEIGHT_DECAY', 0) # weight decay for the optimizer, set to 0 to disable weight decay ( bad idea using it for vae)

#### Scheduler parameters
USE_SCHEDULER = config.get('USE_SCHEDULER', False) # if True, the learning rate scheduler is used
SCHEDULER_PATIENCE = config.get('SCHEDULER_PATIENCE', 10) # number of epochs with no improvement after which learning rate will be reduced
SCHEDULER_FACTOR = config.get('SCHEDULER_FACTOR', 0.5) # factor by which the learning rate will be reduced. new_lr = lr * factor
SCHEDULER_TYPE = config.get('SCHEDULER_TYPE', 'ReduceLROnPlateau') # type of the scheduler, can be 'CosineAnnealingLR' or 'StepLR' or 'ReduceLROnPlateau'
SCHEDULER_THRESHOLD = config.get('SCHEDULER_THRESHOLD', 0.0001) # threshold for the scheduler, used to stop the training if the loss is below this value

# Beta annealing parameters
BETA = config.get('BETA', None)
CYCLIC_BETA = config.get('CYCLIC_BETA', False) # if True, the beta is cyclically annealed
CYCLE_LENGTH = config.get('CYCLE_LENGTH', 10) # number of epochs for the cyclic beta annealing
wait_epochs = config.get('wait_epochs', 0)
annealing_epochs = config.get('annealing_epochs', 50)
beta_min = config.get('beta_min', 0.00001)
beta_max = config.get('beta_max', 0.0001)

ADVANCED_KL_LOSS = config.get('advanced_kl_loss', False) # if True, the KL loss is computed using the advanced method (TC-VAE)
tc_weight = config.get('tc_weight', 1.0) # weight for the total correlation loss
mi_weight = config.get('mi_weight', 0.0) # weight for the mutual information loss
dkl_weight = config.get('dkl_weight', 1.0) #

# Other parameters
DISABLE_TQDM = config.get('DISABLE_TQDM', False) # if True, the tqdm progress bar is disabled
SEED = config.get('SEED', 42) # seed for reproducibility
NAME_SIMULATION = config.get('NAME_SIMULATION', None ) # name of the simulation, used to create a folder to save the model
NAME_FOLDER = config.get('NAME_FOLDER', 'template') # name of the folder to save the model, if None, the folder will be created with the name of the model architecture and encoder type
CONTINUE_FROM = config.get('CONTINUE_FROM', None) # set the path to the model to continue training from, if None, the training will start from scratch
STARTING_EPOCH = config.get('STARTING_EPOCH', 0) # if CONTINUE_FROM is not None, the training will start from this epoch
#TEST_MODEL = config.get('TEST_MODEL', False) # if True, the model is tested after training
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

# Load the data... for now take from a csv file, future versions will take from a database and extrat the dihedral angles from the pdb files
file_csv = "../DATA/dihedrals.csv"
dataset = create_dataset(file_csv, verbose=verbose)

# Split the dataset into train, validation and test sets
train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0, seed=SEED, train_ratio=0.7, val_ratio=0.2, verbose=verbose)




# Create the model
inchannels = dataset.num_features  # number of input features, in this case the number of dihedral angles
if verbose: print(f"Input channels: {inchannels}")

out_channels = dataset[0].num_nodes
if verbose: print(f"Output channels: {out_channels}")

encoder = encoder(inchannels, LATENT_DIM, HIDDEN_ENCODER_CHANNELS, enc_type=ENCODER_TYPE, num_layers=NUM_ENC_LAYERS, attention=ATTENTION_ENCODER, heads=HEADS_ENCODER)
decoder = MLP_Decoder(LATENT_DIM, out_channels, dataset.num_features, HIDDEN_DECODER_CHANNELS, num_layers=NUM_DEC_LAYERS)
model = DVGAE(encoder, decoder,device=device)
model = model.to(device)

if verbose: print(f"Model created with encoder type {ENCODER_TYPE} and latent dimension {LATENT_DIM}")

if CONTINUE_FROM is not None:
    print(f"Loading model from {CONTINUE_FROM}")
    if not os.path.exists(CONTINUE_FROM):
        raise FileNotFoundError(f"The path {CONTINUE_FROM} does not exist. Please check the path and try again.")
    model.load_state_dict(torch.load(CONTINUE_FROM, map_location=device))
    print("Model loaded successfully.")

# Freeze the encoder if specified
if FREEZE_ENCODER:
    for param in model.encoder.parameters():
        param.requires_grad = False
    if verbose: print("Encoder parameters are frozen. Only the decoder will be trained.")
      
if verbose: print_model_summary(model)

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Create the scheduler
if USE_SCHEDULER:
    if SCHEDULER_TYPE == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) # never used, could need  other implementation
    elif SCHEDULER_TYPE == 'StepLR':
        scheduler = StepLR(optimizer, step_size=SCHEDULER_PATIENCE, gamma=SCHEDULER_FACTOR)
    elif SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, threshold=SCHEDULER_THRESHOLD, min_lr=1e-7)
    else: 
        raise ValueError(f"Unknown scheduler type: {SCHEDULER_TYPE}")

# Define the path to save the models, use numbers to define new simulations 
if NAME_SIMULATION is not None:
    file_path = f'../RUNS/{NAME_FOLDER}/{NAME_SIMULATION}/'
    if os.path.exists(file_path):
        print(f"Directory {file_path} already exists. Adding # to the name.")
        # If the directory already exists, append a number to the name
        i = 0
        while os.path.exists(file_path):
            i += 1
            file_path = f'../RUNS/{NAME_FOLDER}/{NAME_SIMULATION}_{i}/'
        print(f"Creating directory {file_path}")
        os.makedirs(file_path)
else:
    i = 1
    while True:
        if not os.path.exists(f'../RUNS/{NAME_FOLDER}/simulation_{i}'):
            print(f"Creating directory for simulation {i}")
            os.makedirs(f'../RUNS/{NAME_FOLDER}/simulation_{i}')
            file_path = f'../RUNS/{NAME_FOLDER}/simulation_{i}/'
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

# ___________________________________ TRAINING LOOP ___________________________________ #

for epoch in range(STARTING_EPOCH, STARTING_EPOCH + EPOCHS):

    # --- LR WARM-UP LOGIC ---
    if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
        # Linearly increase the learning rate
        lr_scale = (epoch + 1) / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE * lr_scale
    # After warm-up, the LR is the standard one (and can be controlled by a scheduler)
    
    if USE_SCHEDULER and epoch >= WARMUP_EPOCHS:
        if lr > scheduler.get_last_lr()[0]:
            print(f"Adjusting learning rate from {lr} to {scheduler.get_last_lr()[0]}")
            lr = scheduler.get_last_lr()[0]

    if BETA is not None:
        beta = BETA
    elif not CYCLIC_BETA:
        beta = beta_annealer(epoch,beta_min, beta_max, annealing_epochs,wait_epochs )
    else:
        beta = cyclic_annealing(epoch, beta_min, beta_max, CYCLE_LENGTH)
        print(f"Using cyclic beta annealing: {beta}")


    train_pbar = tqdm(train_loader, disable=DISABLE_TQDM, desc=f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} [Train]", leave=False)

    # Training
    model.train()
    train_loss = 0
    train_kl_loss = 0
    train_recon_loss = 0
    train_kl_sep_loss = 0 
    train_mi_loss = 0
    train_tc_loss = 0


    for data in train_pbar:
        data = data.to(device)
        optimizer.zero_grad()

        x_recon, mu, logstd = model(data.x, data.edge_index, data.batch)

        if debug:
            print("x_recon shape:", x_recon.shape)
            print("mu shape:", mu.shape)
            print("logstd shape:", logstd.shape)
            print("x shape:", data.x.shape)

        # Compute the loss
        angle_loss_value = angle_loss(x_recon, data.x, data.batch)
        if ADVANCED_KL_LOSS:
            #total_kl_loss,kl_loss_value, mi_loss_value, tc_loss_value = compute_tc_vae_loss(x_recon, data.x, data.edge_index, mu, logstd, data.batch, beta=beta)
            kl_loss_value, mi_loss_value, tc_loss_value = compute_tc_vae_loss(x_recon, data.x, data.edge_index, mu, logstd, data.batch, beta=beta)
            total_kl_loss = dkl_weight* kl_loss_value + mi_weight * mi_loss_value + tc_weight * tc_loss_value
        else:
            kl_loss_value = kl_loss(mu, logstd)
            total_kl_loss = kl_loss_value
            mi_loss_value = torch.tensor(0.0)
            tc_loss_value = torch.tensor(0.0)

        if total_kl_loss < MIN_KL:
            loss = angle_loss_value
        else:
            loss = angle_loss_value + beta * total_kl_loss

        if debug:
            print(f"angle_loss_value: {angle_loss_value.item()}, kl_loss_value: {kl_loss_value.item()}, loss: {loss.item()}")
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_kl_loss += total_kl_loss.item()
        train_recon_loss += angle_loss_value.item()
        train_kl_sep_loss += kl_loss_value.item()
        train_mi_loss += mi_loss_value.item()
        train_tc_loss += tc_loss_value.item()

        # Update the progress bar
        if ADVANCED_KL_LOSS:
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'KL Loss': f'{kl_loss_value.item():.4f}',
                'Recon Loss': f'{angle_loss_value.item():.4f}',
                'MI Loss': f'{mi_loss_value.item():.4f}',
                'TC Loss': f'{tc_loss_value.item():.4f}',
                'Beta': f'{beta:.4f}',
                'LR': f'{lr:.4e}'
            })
        else:   
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'KL Loss': f'{kl_loss_value.item():.4f}',
                'Recon Loss': f'{angle_loss_value.item():.4f}',
                'Beta': f'{beta:.4f}',
                'LR': f'{lr:.4e}'
            })  
        # Log the losses to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('KL_Loss/train', kl_loss_value.item(), epoch)
        writer.add_scalar('Recon_Loss/train', angle_loss_value.item(), epoch)
        writer.add_scalar('Beta/train', beta, epoch)
        writer.add_scalar('Learning_Rate/train', lr, epoch)
        if ADVANCED_KL_LOSS:
            writer.add_scalar('MI_Loss/train', mi_loss_value.item(), epoch)
            writer.add_scalar('TC_Loss/train', tc_loss_value.item(), epoch)


        train_pbar.close()

    # Average the losses
    train_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_kl_sep_loss /= len(train_loader)
    train_mi_loss /= len(train_loader)
    train_tc_loss /= len(train_loader)

    if ADVANCED_KL_LOSS:
        print(f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} - Train Loss: {train_loss:.4f}, KL Loss: {train_kl_loss:.2e}, Recon Loss: {train_recon_loss:.4f}, KL sep Loss: {train_kl_sep_loss:.2e}, MI Loss: {train_mi_loss:.2e}, TC Loss: {train_tc_loss:.4f}, Beta: {beta:.2e}, LR: {lr:.2e}")
    else:
        print(f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} - Train Loss: {train_loss:.4f}, KL Loss: {train_kl_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, Beta: {beta:.4f}, LR: {lr:.4e}")

   
    # Validation
    model.eval()
    val_loss = 0
    val_kl_loss = 0
    val_recon_loss = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, disable=DISABLE_TQDM, desc=f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} [Val]", leave=False)
        for data in val_pbar:
            data = data.to(device)

            x_recon, mu, logstd = model(data.x, data.edge_index, data.batch)

            # Compute the loss
            angle_loss_value = angle_loss(x_recon, data.x,data.batch)
            kl_loss_value = kl_loss(mu, logstd)
            if kl_loss_value < MIN_KL:
                loss = angle_loss_value
            else:
                loss = angle_loss_value + beta * kl_loss_value

            val_loss += loss.item()
            val_kl_loss += kl_loss_value.item()
            val_recon_loss += angle_loss_value.item()

            # Update the progress bar
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'KL Loss': f'{kl_loss_value.item():.4f}',
                'Recon Loss': f'{angle_loss_value.item():.4f}',
                'Beta': f'{beta:.4f}',
                'LR': f'{lr:.4e}'
            })
        val_pbar.close()

    # Average the losses
    val_loss /= len(val_loader)
    val_kl_loss /= len(val_loader)
    val_recon_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} - Val Loss: {val_loss:.4f}, KL Loss: {val_kl_loss:.4f}, Recon Loss: {val_recon_loss:.4f}")

    if USE_SCHEDULER and SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler.step(val_loss)
    elif USE_SCHEDULER:
        scheduler.step()

    # Log the validation losses to TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('KL_Loss/val', val_kl_loss, epoch)
    writer.add_scalar('Recon_Loss/val', val_recon_loss, epoch)

    # save the model every 20 epochs
    if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
        torch.save(model.state_dict(), file_path + f'model_epoch_{epoch+1}.pth')
        if verbose: print(f"Model saved to {file_path}model_epoch_{epoch+1}.pth")

    # test the model on the test set
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        directory = file_path + f'test_epoch_{epoch+1}/'
        os.makedirs(directory, exist_ok=True)

        test_generation_ramachandran(model,file_csv, save_path=directory + f'test_generation_epoch_{epoch+1}.png')
        test_reconstruction_ramachandran(model, test_loader, file_csv, save_path=directory + f'test_reconstruction_epoch_{epoch+1}.png')
        analyze_latent_space(model, test_loader, device, file_path=directory, verbose=verbose)

        plt.close()

