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

sys.path.append("LIBS")
from LIBS.utils import *

import argparse 
# Create a single parser with both arguments
parser = argparse.ArgumentParser(description='Full Graph VAE with EGNN')
parser.add_argument('--config', type=str, default='config.template.in', help='Path to the configuration file')
parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode')

# Parse the command line arguments
args = parser.parse_args()
config_file = args.config
verbose = args.verbose

# Read the parameters from the config file
config = parse_config(config_file=config_file)


# TO BE UPDATED IN A CLEAR WAY
# model parameters
LATENT_DIM = config.get('LATENT_DIM', 128)
HIDDEN_ENCODER_CHANNELS = config.get('HIDDEN_ENCODER_CHANNELS', 256)
HIDDEN_DECODER_CHANNELS = config.get('HIDDEN_DECODER_CHANNELS', 256)
NUM_DEC_LAYERS = config.get('NUM_DEC_LAYERS', 5)
NUM_EGNN_LAYERS = config.get('NUM_EGNN_LAYERS', 5)

INCLUDE_ATOM_TYPE = True # if True, the first feature of the input is the atom type and ohe is applied
SCALE_POS_FACTOR = config.get('SCALE_POS_FACTOR', 10.0)
SCALE_FEATURES = config.get('SCALE_FEATURES', True)

# training parameters
EPOCHS = config.get('EPOCHS', 50)
BATCHSIZE = config.get('BATCHSIZE', 64)
LEARNING_RATE = config.get('LEARNING_RATE', 1E-4)
WEIGHT_DECAY = 0 # 1E-5 # weight decay for the optimizer, set to 0 to disable weight decay ( bad idea for vae)

# Scheduler parameters
SCHEDULER = True # if True, a scheduler is used to reduce the learning rate
                 # SCHEDULER IS ON TRAINING LOSS, NOT VALIDATION LOSS
TRESHOLD = 0.005 # threshold for the scheduler
PATIENCE = 5 # number of epochs with no improvement after which learning rate will be reduced
FACTOR = 0.5 # factor by which the learning rate will be reduced
DISABLE_TQDM = config.get('DISABLE_TQDM', False) # if True, the tqdm progress bar is disabled

# Beta annealing parameters
BETA = config.get('BETA', None)
wait_epochs = config.get('wait_epochs', 0)
annealing_epochs = config.get('annealing_epochs', 50)
beta_min = config.get('beta_min', 0.00001)
beta_max = config.get('beta_max', 0.0001)

NAME_SIMULATION = config.get('NAME_SIMULATION', None ) # name of the simulation, used to create a folder to save the model
TYPE_SIM = config.get('TYPE_SIM', 2) # type of the simulation, used to create a folder to save the model

CONTINUE_FROM = config.get('continue_from', None)
STARTING_EPOCH = 30 # if CONTINUE_FROM is not None, the training will start from this epoch
if CONTINUE_FROM is None: 
    STARTING_EPOCH = 0 # if CONTINUE_FROM is None, the training will start from epoch 0

SEED = 42



# Create dataset and dataloaders
dataset = get_dataset(initial_alignment=True, verbose=verbose)
train_loader, val_loader, test_loader = get_dataloaders(dataset,verbose=verbose, batch_size=BATCHSIZE)

