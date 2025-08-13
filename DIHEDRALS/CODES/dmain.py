

from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn.pool import global_mean_pool
import math

from torch.utils.tensorboard import SummaryWriter

import itertools

from sklearn.preprocessing import StandardScaler


import os
import sys

import argparse


sys.path.append('LIBS')

from LIBS.DVGAE import *
from LIBS.dutils import *


# Create a single parser with both arguments
parser = argparse.ArgumentParser(description='Dihedral Graph VAE with EGNN')
parser.add_argument('--config', type=str, default='config.template.in', help='Path to the configuration file')
parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode')


# Parse the command line arguments
args = parser.parse_args()
config_file = args.config
verbose = args.verbose

#verbose= True  # set to debug for now, will be set to False in the future

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

#### model parameters
# encoder
ENCODER_TYPE = config.get('ENCODER_TYPE', 'standard') # type of the encoder, can be GCN or SAGE
HIDDEN_ENCODER_CHANNELS = config.get('HIDDEN_ENCODER_CHANNELS', 256)
OUT_ENCODER_CHANNELS = config.get('OUT_ENCODER_CHANNELS', 128)
NUM_ENC_LAYERS = config.get('NUM_ENC_LAYERS', 5) # number of EGNN layers in the encoder
#ATTENTION_ENCODER = config.get('ATTENTION_ENCODER', True) # if True, attention is used in the encoder
LATENT_DIM = config.get('LATENT_DIM', 128) # latent dimension of the encoder, used to create the latent space
#TANH_ENCODER = config.get('TANH_ENCODER', True) # if True, the output of the encoder is passed through a tanh activation function (for positions)
#NORMALIZE_ENCODER = config.get('NORMALIZE_ENCODER', True) # if True, the encoder output is normalized
# decoder

HIDDEN_DECODER_CHANNELS = config.get('HIDDEN_DECODER_CHANNELS', 256)
NUM_DEC_LAYERS = config.get('NUM_DEC_LAYERS', 5)
# ATTENTION_DECODER = config.get('ATTENTION_DECODER', True)
# TANH_DECODER = config.get('TANH_DECODER', True) # if True, the output of the decoder is passed through a tanh activation function (for positions)
# NORMALIZE_DECODER = config.get('NORMALIZE_DECODER', True) # if True, the decoder output is normalized


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

# force field parameters
# USE_FORCE_FIELD = config.get('USE_FORCE_FIELD', True) # if True, the force field is used to calculate the energy of the system
# PDB_FOR_ENERGY = config.get('PDB_FOR_ENERGY', '../DATA/raw/protein_only.pdb') # path to the PDB file for energy calculation
# LAMBDA_ENERGY = config.get('LAMBDA_ENERGY', None) # weight for the energy loss in the total loss function
# wait_lambda_epochs = config.get('wait_lambda_epochs', 10) # number of epochs to wait before starting to use the force field in the loss function
# lambda_annealing_epochs = config.get('lambda_annealing_epochs', 50) # number of epochs to anneal the lambda parameter
# lambda_min = config.get('lambda_min', 1e-15) # minimum value for the lambda parameter
# lambda_max = config.get('lambda_max', 0.001) # maximum value for the lambda parameter

# USE_LOG_FF = config.get('USE_LOG_FF', True) # if True, the force field loss is calculated using log scaling, otherwise it is calculated using the raw values
# USE_BOND_FF = config.get('USE_BOND_FF', True) # if True, the bonded energy is included in the force field loss
# USE_ANGLE_FF = config.get('USE_ANGLE_FF', True) # if True, the angle energy is included in the force field loss
# USE_LJ_FF = config.get('USE_LJ_FF', True) # if True, the Lennard-Jones energy is included in the force field loss


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


